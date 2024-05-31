import torch
from torch import nn
from torchinfo import summary


# Part 1: Image Embedding
class ImageEmbedding(nn.Module):
    def __init__(self, 
                 input_channels: int=3, 
                 embedding_size: int=768, 
                 patch_size: int=16, 
                 batch_size: int=32,
                 embedding_dropout: float=0.1,
                 img_size: int=224):
        super().__init__()
        self.input_channels = input_channels
        self.embedding_size = embedding_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_patches = int(img_size * img_size / patch_size ** 2)
        self.projection = nn.Conv2d(in_channels=self.input_channels,
                                    out_channels=self.embedding_size,
                                    kernel_size=self.patch_size,
                                    stride=self.patch_size)
        
        self.flatten = nn.Flatten(start_dim=2)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

    def add_positional_embedding(self, x):
        # Add positional embedding
        positional_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, self.embedding_size),
                                            requires_grad=True).to(x.device)
        patch_and_positional_embedding = x + positional_embedding
        return patch_and_positional_embedding

    def add_class_token(self, x):
        # Add class token
        class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.embedding_size),
                                requires_grad=True).to(x.device)
        patch_embedding_with_class_token = torch.cat((class_token, x), dim=1)
        return patch_embedding_with_class_token

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Image resolution must be divisible by patch size"
        x = self.projection(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1) # (batch_size, num_patches, embedding dimension)

        x = self.add_class_token(x)

        x = self.add_positional_embedding(x)

        x = self.embedding_dropout(x)

        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, 
                  embedding_dimension:int=768,
                  num_heads:int=12,
                  attn_dropout:int = 0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dimension)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dimension,
                                                         num_heads=num_heads,
                                                         dropout=attn_dropout,
                                                         batch_first=True) # Is batch_first? (batch, seq, feature) -> (batch_size, number of patches, embedding dimension)

    def forward(self, x):
        x = self.layer_norm(x)
        attention_output, _ = self.multihead_attention(query=x,
                                                      key=x,
                                                      value=x,
                                                      need_weights=False)
        return attention_output

class MultiLayerPerceptron(nn.Module):
    def __init__(self, 
                  embedding_dimension:int=768,
                  mlp_size:int=3072, # Given in the paper Table 1
                  dropout:int = 0.1): # Given in the paper
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dimension)
        # The MLP contains two layers with a GELU non-linearity (from the paper)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dimension, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dimension),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, 
                  embedding_dimension:int=768,
                  num_heads:int=12,
                  mlp_size:int=3072,
                  mlp_dropout:int = 0.1,
                  attn_dropout:int = 0):
        super().__init__()

        self.msa_block = MultiHeadAttentionBlock(embedding_dimension=embedding_dimension,
                                                 num_heads=num_heads,
                                                 attn_dropout=attn_dropout)
        self.mlp_block = MultiLayerPerceptron(embedding_dimension=embedding_dimension,
                                              mlp_size=mlp_size,
                                              dropout=mlp_dropout)
        
    def forward(self, x):
        x = self.msa_block(x) + x # Add skip connection, kind of like residual connection (as shown in the encoder block)
        x = self.mlp_block(x) + x # Add skip connection, kind of like residual connection (as shown in the encoder block)
        return x

class ViT(nn.Module):
    def __init__(self,
                img_size:int=224,# Table 3 from the ViT paper
                in_channels:int=3,
                patch_size:int=16,
                num_transformer_layers:int=12,
                embedding_dim:int=768,
                mlp_size:int=3072,
                num_heads:int=12,
                attn_dropout:int=0,
                mlp_dropout:int=0.1,
                embedding_dropout:int=0.1,
                num_classes:int=1000):
        
        super().__init__()

        # Check Image size is compatible with patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size"

        # Image embeddings
        self.image_embedding = ImageEmbedding(input_channels=in_channels,
                                              embedding_size=embedding_dim,
                                              patch_size=patch_size,
                                              batch_size=32,
                                              embedding_dropout=embedding_dropout)
                                              
        
        # Transformer Encoder Layers
        # Initialize an empty list to hold the TransformerEncoder layers
        transformer_layers = []

        # Loop over the number of transformer layers we want to create
        for _ in range(num_transformer_layers):
            # Create a new TransformerEncoder layer
            layer = TransformerEncoder(
                embedding_dimension=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
                attn_dropout=attn_dropout
            )
            transformer_layers.append(layer)

        # Create sequential model from transformer layers
        self.transformer_encoder = nn.Sequential(*transformer_layers)

        """ Same above code (Transformer Encoder Layers) in compact form
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(embedding_dimension=embedding_dim,
                                                                      num_heads=num_heads,
                                                                      mlp_size=mlp_size,
                                                                      mlp_dropout=mlp_dropout,
                                                                      attn_dropout=attn_dropout) for _ in range(num_transformer_layers)]).to(device)
        """        

        # MLP head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        # Get the batch size
        batch_size = x.shape[0]

        # Image embeddings
        self.image_embedding.batch_size = batch_size
        x = self.image_embedding(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # MLP head
        # Put 0th index logit through the classifier (equation 4)
        x = self.classifier(x[:, 0])

        return x

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ViT().to(device)

    # Summary using torchinfo
    summary(model=model,
            input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
if __name__ == "__main__":
    main()
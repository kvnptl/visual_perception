import os
import shutil

def copy_png_files(src_dir, dest_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through all the subdirectories in the source directory
    for subdir, _, files in os.walk(src_dir):
        for file in files:
            # Check if the file is a png file
            if file.endswith('.png'):
                # Generate full paths for source and destination files
                src_path = os.path.join(subdir, file)
                # dest_path = os.path.join(dest_dir, file)
                dest_path = os.path.join(dest_dir, subdir.split('/')[-1] + "_" + os.path.basename(src_path))
                
                # # Handle the case if a file with the same name already exists in the destination directory
                # counter = 1
                # while os.path.exists(dest_path):
                #     # dest_path = os.path.join(dest_dir, f"{file.split('.')[0]}_{counter}.png")
                #     # Destination path is src_path + counter
                #     dest_path = os.path.join(dest_dir, os.path.basename(src_path) + "_" + str(counter) + ".png")
                #     counter += 1
                
                # Copy the file
                shutil.copy2(src_path, dest_path)
                print(f"Copied {file} from {src_path} to {dest_path}")

if __name__ == "__main__":
    src_dir = "/dir/source"  # Replace with the path to your source directory
    dest_dir = "/dir/destination"  # Replace with the path to your destination directory

    copy_png_files(src_dir, dest_dir)

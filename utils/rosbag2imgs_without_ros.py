"""
Script to extract images from rosbag without using ROS installation.

Requirements:
pip3 install bagpy opencv-python tqdm numpy os

"""

import os
import cv2
import rosbag
import numpy as np
from tqdm import tqdm

def print_topics_in_rosbag(bag_file):
    with rosbag.Bag(bag_file, 'r') as bag:
        topics = bag.get_type_and_topic_info()[1].keys()
        for topic in topics:
            print(topic)

def extract_images_from_rosbag(bag_file, output_dir, image_topic):
    count = 0
    saved_count = 0

    # Check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in tqdm(bag.read_messages(topics=[image_topic])):
            if count % 10 == 0:  # Process every 10th frame
                if msg._type == 'sensor_msgs/Image':
                    # Decode the image message
                    if msg.encoding == 'rgb8':
                        dtype = np.uint8
                    elif msg.encoding == 'mono8':
                        dtype = np.uint8
                    elif msg.encoding == '16UC1':
                        dtype = np.uint16
                    else:
                        raise ValueError(f"Unsupported encoding: {msg.encoding}")

                    img_data = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, msg.width, -1))
                    if msg.encoding == 'rgb8':
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    elif msg.encoding == 'mono8':
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                    
                    # Save the image
                    cv2.imwrite(os.path.join(output_dir, f'frame{saved_count:06d}.png'), img_data)
                    saved_count += 1
            count += 1

    print(f"Processed {count} frames, saved {saved_count} frames")

if __name__ == '__main__':
    root_dir = "/bagfiles"
    file_name = "filename.bag"
    bag_file = os.path.join(root_dir, file_name)
    output_dir = os.path.join(root_dir, file_name.split(".")[0])
    image_topic = "/image_raw"

    # print_topics_in_rosbag(bag_file)
    extract_images_from_rosbag(bag_file, output_dir, image_topic)
    

import os
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tqdm import tqdm

def main(bag_file, output_dir, image_topic):
    bridge = CvBridge()
    count = 0

    # check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for topic, msg, t in rosbag.Bag(bag_file, "r").read_messages(topics=[image_topic]):
        if "rgb" in msg.encoding.lower():
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f'_frame{count:06d}.png'), cv_img)
            count += 1

    print(f"Processed {count} frames")

def main_limit_frame_rate(bag_file, output_dir, image_topic):
    bridge = CvBridge()
    count = 0  # Total frames processed
    saved_count = 0  # Total frames saved
    frame_skip = 6  # Only process every 6th frame

    # check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for topic, msg, t in tqdm(rosbag.Bag(bag_file, "r").read_messages(topics=[image_topic])):
        if count % frame_skip == 0:
            if "rgb" in msg.encoding.lower():
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, f'frame{saved_count:06d}.png'), cv_img)
                saved_count += 1
        count += 1

    print(f"Processed {count} frames")

if __name__ == '__main__':
    root_dir = "/rosbags"
    file_name = "filename.bag"
    bag_file = os.path.join(root_dir, file_name)
    output_dir = os.path.join(root_dir, file_name.split(".")[0])
    image_topic = "/image_raw"

    # Uncomment the one you want to use
    main(bag_file, output_dir, image_topic)
    # main_limit_frame_rate(bag_file, output_dir, image_topic)

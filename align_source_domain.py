import os
import cv2
import numpy as np
from tqdm import tqdm

# Function to align the channels
def align_channels(channel, mean_target, std_dev_target, mean_source, std_dev_source):
    # Z-score normalization
    channel = ((channel - mean_source) / std_dev_source) * std_dev_target + mean_target
    # Clip the values to be in the valid range [0, 255]
    channel = np.clip(channel, 0, 255).astype(np.uint8)
    return channel

# Function to align an image from source domain to target domain
def align_image(image, mean_a_target, std_a_target, mean_b_target, std_b_target, mean_a_source, std_a_source, mean_b_source, std_b_source):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_image)
    aligned_a = align_channels(a, mean_a_target, std_a_target, mean_a_source, std_a_source)
    aligned_b = align_channels(b, mean_b_target, std_b_target, mean_b_source, std_b_source)
    aligned_lab = cv2.merge([l, aligned_a, aligned_b])
    aligned_image = cv2.cvtColor(aligned_lab, cv2.COLOR_Lab2BGR)
    return aligned_image

if __name__ == '__main__':
    # Mean and standard deviation values for target and source domains (replace with your actual values)
    mean_a_target = 127.66211763313609
    std_a_target = 2.9876052041537933
    mean_b_target = 137.06047654585797
    std_b_target = 8.431086753716576

    mean_a_source = 126.6375984907259
    std_a_source = 1.7286320877961807
    mean_b_source = 136.27915872199804
    std_b_source = 7.060126031814528

    # Source and destination folders (replace with your actual folder paths)
    source_folder = "./domain_A"
    destination_folder = "./domain_A_normalized"
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Process each image in the source folder
    for filename in tqdm(os.listdir(source_folder)):
        if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
            filepath = os.path.join(source_folder, filename)
            image = cv2.imread(filepath)

            aligned_image = align_image(image, mean_a_target, std_a_target, mean_b_target, std_b_target,
                                        mean_a_source, std_a_source, mean_b_source, std_b_source)

            save_path = os.path.join(destination_folder, filename)
            cv2.imwrite(save_path, aligned_image)
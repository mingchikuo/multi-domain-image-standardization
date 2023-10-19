import os
import cv2
import numpy as np
from tqdm import tqdm

# Function to process images and calculate average and standard deviation for A and B channel
def calculate_channel_stats(folder_path):
    a_channel_avg_list = []
    b_channel_avg_list = []
    
    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab_image)
            
            # Calculate mean and standard deviation for A channel
            a_channel_avg = np.mean(a)
            a_channel_avg_list.append(a_channel_avg)
            
            # Calculate mean and standard deviation for B channel
            b_channel_avg = np.mean(b)
            b_channel_avg_list.append(b_channel_avg)

    # Calculate global mean and standard deviation for A channel
    global_a_channel_mean = np.mean(a_channel_avg_list)
    global_a_channel_std = np.std(a_channel_avg_list)
    
    # Calculate global mean and standard deviation for B channel
    global_b_channel_mean = np.mean(b_channel_avg_list)
    global_b_channel_std = np.std(b_channel_avg_list)
    
    return global_a_channel_mean, global_a_channel_std, global_b_channel_mean, global_b_channel_std

if __name__ == "__main__":
    folder_path = "./domain_A"  # Replace with the path to your image folder
    global_a_channel_mean, global_a_channel_std, global_b_channel_mean, global_b_channel_std = calculate_channel_stats(folder_path)
    
    print(f"Global A channel average: {global_a_channel_mean}")
    print(f"Global A channel standard deviation: {global_a_channel_std}")
    print(f"Global B channel average: {global_b_channel_mean}")
    print(f"Global B channel standard deviation: {global_b_channel_std}")

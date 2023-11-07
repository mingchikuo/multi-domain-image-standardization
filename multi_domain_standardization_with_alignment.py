import cv2
import numpy as np
import os
import math
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import albumentations as A

class MultiDomainStandardization(ImageOnlyTransform):
    def __init__(self, p=1.0, always_apply=False):
        super(MultiDomainStandardization, self).__init__(always_apply, p)
        
    def apply(self, img, **params):
        # Step 1: Color Balance
        balanced_img = self.dynamic_white_balance(img)
        # Step 2: Normalize Brightness
        normalized_img = self.normalize_brightness(balanced_img)
        # Step 3: Normalize Contrast
        contrast_normalized_img = self.normalize_contrast(normalized_img)
        # Step 4: Align to Training Set
        target_a_mean = 127.78945087199409
        target_a_std = 2.3812908711197913
        target_b_mean = 133.73629809548993
        target_b_std = 7.459165608749118
        standardized_img = self.align_to_target_domain(contrast_normalized_img, target_a_mean, target_a_std, target_b_mean, target_b_std)
    
        return standardized_img

    def dynamic_white_balance(self, img):
        # Your implementation here
        r, g, b = cv2.split(img)
        r_avg = np.mean(r)
        g_avg = np.mean(g)
        b_avg = np.mean(b)
        avg = (r_avg + g_avg + b_avg) / 3.0

        r_gain = min(avg / r_avg, 255 / np.max(r))
        g_gain = min(avg / g_avg, 255 / np.max(g))
        b_gain = min(avg / b_avg, 255 / np.max(b))

        r = np.clip(r * r_gain, 0, 255).astype(np.uint8)
        g = np.clip(g * g_gain, 0, 255).astype(np.uint8)
        b = np.clip(b * b_gain, 0, 255).astype(np.uint8)
        return cv2.merge((r, g, b))

    def normalize_brightness(self, img, target_mean=64):
        # Your implementation here
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l_channel = img_lab[:, :, 0].astype(np.float32)
        mean_val = np.mean(l_channel)

        gamma_val = math.log10(target_mean / 255) / math.log10(mean_val / 255)
        gamma_table = [np.power(x / 255.0, gamma_val) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        corrected_l_channel = cv2.LUT(l_channel.astype(np.uint8), gamma_table)

        img_lab[:, :, 0] = corrected_l_channel
        return cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    
    def normalize_contrast(self, img, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Normalize the contrast of an image using CLAHE.
        
        :param img: Input image to be processed.
        :param clip_limit: Threshold for contrast limiting.
        :param tile_grid_size: Size of grid for histogram equalization.
        :return: Contrast normalized image.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        contrast_normalized_img = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
        return contrast_normalized_img
    
    def align_to_target_domain(self, img, target_a_mean, target_a_std, target_b_mean, target_b_std):
        """
        Align the A and B channels of the LAB image to the target domain statistics.

        :param img: Input image to be processed.
        :param target_a_mean: Target A channel mean.
        :param target_a_std: Target A channel standard deviation.
        :param target_b_mean: Target B channel mean.
        :param target_b_std: Target B channel standard deviation.
        :return: Image aligned to the target domain.
        """
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab_image)
        source_a_mean, source_a_std = np.mean(a), np.std(a)
        source_b_mean, source_b_std = np.mean(b), np.std(b)
        
        # Align A channel
        a = self.align_channels(a, target_a_mean, target_a_std, source_a_mean, source_a_std)
        # Align B channel
        b = self.align_channels(b, target_b_mean, target_b_std, source_b_mean, source_b_std)
        
        aligned_lab = cv2.merge([l, a, b])
        aligned_image = cv2.cvtColor(aligned_lab, cv2.COLOR_Lab2BGR)
        return aligned_image

    def align_channels(self, channel, mean_target, std_target, mean_source, std_source):
        # Z-score normalization
        channel_normalized = ((channel - mean_source) / std_source) * std_target + mean_target
        # Clip the values to be in the valid range [0, 255]
        channel_clipped = np.clip(channel_normalized, 0, 255).astype(np.uint8)
        return channel_clipped
    
def main(source_folder, destination_folder, image_processing_function):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(source_folder, filename)
            img = cv2.imread(filepath)

            processed_img = image_processing_function(img)

            destination_filepath = os.path.join(destination_folder, filename)
            cv2.imwrite(destination_filepath, processed_img)

if __name__ == "__main__":
    source_folder = 'image_288_image_288_ROI'
    destination_folder = 'image_288_image_288_ROI_s_a'
    
    augmentation_pipeline = A.Compose([
        MultiDomainStandardization(p=1)
    ])

    def albumentation_function(img):
        augmented = augmentation_pipeline(image=img)
        return augmented['image']

    main(source_folder, destination_folder, albumentation_function)
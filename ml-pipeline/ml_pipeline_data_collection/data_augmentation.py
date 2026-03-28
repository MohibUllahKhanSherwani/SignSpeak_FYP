"""
Data Augmentation Module for Sign Language Landmark Sequences

This module provides augmentation techniques specifically designed for
MediaPipe landmark data (spatial-temporal sequences).

Key Features:
- Time warping (speed variations)
- Spatial transformations (scaling, rotation, translation)
- Noise injection
- Horizontal flipping
- Temporal cropping
"""

import numpy as np
from scipy.interpolate import interp1d


class SignLanguageAugmenter:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        
    def time_warp(self, sequence, speed_range=(0.9, 1.1)):
        
        speed_factor = np.random.uniform(*speed_range)
        seq_len = len(sequence)
        
        original_times = np.linspace(0, 1, seq_len)
        
        new_length = int(seq_len / speed_factor)
        new_times = np.linspace(0, 1, new_length)
        
        augmented = np.zeros((seq_len, sequence.shape[1]))
        for i in range(sequence.shape[1]):
            interpolator = interp1d(original_times, sequence[:, i], 
                                   kind='cubic', fill_value='extrapolate')
            resampled = interpolator(np.linspace(0, 1, seq_len))
            augmented[:, i] = resampled
            
        return augmented
    
    def spatial_scale(self, sequence, scale_range=(0.95, 1.05)):
        scale_factor = np.random.uniform(*scale_range)
        
        seq_len = len(sequence)
        landmarks_3d = sequence.reshape(seq_len, -1, 3)
        
        landmarks_3d[:, :, 0] *= scale_factor  # x
        landmarks_3d[:, :, 1] *= scale_factor  # y
        
        return landmarks_3d.reshape(seq_len, -1)
    
    def spatial_translate(self, sequence, translate_range=0.05):
        
        tx = np.random.uniform(-translate_range, translate_range)
        ty = np.random.uniform(-translate_range, translate_range)
        
        seq_len = len(sequence)
        landmarks_3d = sequence.reshape(seq_len, -1, 3)
        
        landmarks_3d[:, :, 0] += tx
        landmarks_3d[:, :, 1] += ty
        
        return landmarks_3d.reshape(seq_len, -1)
    
    def spatial_rotate(self, sequence, angle_range=(-15, 15)):
        angle_deg = np.random.uniform(*angle_range)
        angle_rad = np.deg2rad(angle_deg)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        seq_len = len(sequence)
        landmarks_3d = sequence.reshape(seq_len, -1, 3)
        
        center_x = landmarks_3d[:, :, 0].mean()
        center_y = landmarks_3d[:, :, 1].mean()
        
        landmarks_3d[:, :, 0] -= center_x
        landmarks_3d[:, :, 1] -= center_y
        
        x_new = landmarks_3d[:, :, 0] * cos_a - landmarks_3d[:, :, 1] * sin_a
        y_new = landmarks_3d[:, :, 0] * sin_a + landmarks_3d[:, :, 1] * cos_a
        
        landmarks_3d[:, :, 0] = x_new + center_x
        landmarks_3d[:, :, 1] = y_new + center_y
        
        return landmarks_3d.reshape(seq_len, -1)
    
    def add_noise(self, sequence, noise_std=0.01):
        noise = np.random.normal(0, noise_std, sequence.shape)
        return sequence + noise
    
    def horizontal_flip(self, sequence):
        seq_len = len(sequence)
        landmarks_3d = sequence.reshape(seq_len, -1, 3)
        
        mask = np.any(landmarks_3d != 0, axis=2)
        
        flipped_x = landmarks_3d.copy()
        flipped_x[mask, 0] = 1.0 - landmarks_3d[mask, 0]
        
        final_flipped = flipped_x.copy()
        
        left_hand = flipped_x[:, 0:21]
        right_hand = flipped_x[:, 21:42]
        
        final_flipped[:, 0:21] = right_hand
        final_flipped[:, 21:42] = left_hand
        
        return final_flipped.reshape(seq_len, -1)
    
    def temporal_crop(self, sequence, crop_ratio=0.1):
        seq_len = len(sequence)
        max_crop = int(seq_len * crop_ratio)
        
        start_crop = np.random.randint(0, max_crop + 1)
        end_crop = np.random.randint(0, max_crop + 1)
        
        cropped = sequence[start_crop:seq_len - end_crop]
        
        original_times = np.linspace(0, 1, len(cropped))
        new_times = np.linspace(0, 1, seq_len)
        
        resized = np.zeros((seq_len, sequence.shape[1]))
        for i in range(sequence.shape[1]):
            interpolator = interp1d(original_times, cropped[:, i], 
                                   kind='linear', fill_value='extrapolate')
            resized[:, i] = interpolator(new_times)
        
        return resized
    
    def augment(self, sequence, techniques=None, probabilities=None):
        if techniques is None:
            techniques = [
                'time_warp', 'spatial_scale', 'spatial_translate',
                'spatial_rotate', 'add_noise', 'temporal_crop'
            ]
        
        if probabilities is None:
            probabilities = {
                'time_warp': 0.5,
                'spatial_scale': 0.5,
                'spatial_translate': 0.5,
                'spatial_rotate': 0.3,
                'add_noise': 0.3,
                'temporal_crop': 0.3,
                # 'horizontal_flip': 0.5, # DISABLED for PSL (Non-symmetric)
            }
        
        augmented = sequence.copy()
        
        for technique in techniques:
            if np.random.random() < probabilities.get(technique, 0.0):
                if technique == 'time_warp':
                    augmented = self.time_warp(augmented)
                elif technique == 'spatial_scale':
                    augmented = self.spatial_scale(augmented)
                elif technique == 'spatial_translate':
                    augmented = self.spatial_translate(augmented)
                elif technique == 'spatial_rotate':
                    augmented = self.spatial_rotate(augmented)
                elif technique == 'add_noise':
                    augmented = self.add_noise(augmented)
                elif technique == 'temporal_crop':
                    augmented = self.temporal_crop(augmented)
                elif technique == 'horizontal_flip':
                    augmented = self.horizontal_flip(augmented)
        
        return augmented
    
    def augment_batch(self, sequences, multiplier=2):
        augmented_list = []
        
        for seq in sequences:
            augmented_list.append(seq)
            
            for _ in range(multiplier - 1):
                aug_seq = self.augment(seq)
                augmented_list.append(aug_seq)
        
        return np.array(augmented_list)



def create_augmented_dataset(X, y, augmentation_multiplier=3):
    augmenter = SignLanguageAugmenter()
    
    X_augmented = []
    y_augmented = []
    
    for i, (seq, label) in enumerate(zip(X, y)):
        X_augmented.append(seq)
        y_augmented.append(label)
        
        for _ in range(augmentation_multiplier - 1):
            aug_seq = augmenter.augment(seq)
            X_augmented.append(aug_seq)
            y_augmented.append(label)
    
    return np.array(X_augmented), np.array(y_augmented)



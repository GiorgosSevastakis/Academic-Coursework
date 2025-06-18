#### Imports ####
import numpy as np
import matplotlib.pyplot as plt
import os

import skimage
from skimage.io import imread
from skimage.transform import resize

import hashlib

## Load whole images 
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        img = imread(image_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading image {filename}")
    return images

## Read masks from Cellpose output
def load_npy_files(base_dir):
    subfolders = ['control', 'drug']
    outputs = []
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_dir, subfolder)
        
        if os.path.exists(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(subfolder_path, filename)
                    data = np.load(file_path)
                    outputs.append(data)
        else:
            print(f"Subfolder {subfolder_path} does not exist.")
    return outputs
## Get masks from Cellpose output
def get_masks(base_path):
    segments = load_npy_files(base_path)
    segments_nuclei = []
    for segment in segments:
        segments_nuclei.append(segment[:, :, 1])
    return segments_nuclei[:10],segments_nuclei[10:]

##Find median image, return size
def get_median_image(ctrl_masks,pntr_masks):
    from skimage import measure
    ctrl_temp = []
    pntr_temp = []
    for k,i in enumerate(ctrl_masks):
        regions = measure.regionprops(i)
        for j,reg in enumerate(regions):
            bi_mask = reg.image
            ctrl_temp.append(bi_mask.shape)
    for k,i in enumerate(pntr_masks):
        regions = measure.regionprops(i)
        for j,reg in enumerate(regions):
            bi_mask = reg.image
            pntr_temp.append(bi_mask.shape)

    ctrl_temp = np.array(ctrl_temp)
    pntr_temp = np.array(pntr_temp)
    
    max_ctrl = np.median(ctrl_temp, axis=0)
    max_pntr = np.median(pntr_temp, axis=0)
    
    return int(np.median([max_ctrl, max_pntr]))

# Get cells from images using masks
def get_cells_re(image_list,mask_list):
    from skimage import measure
    temp = []
    for k,i in enumerate(mask_list):
        regions = measure.regionprops(i)
        for j,reg in enumerate(regions):
            min_row, min_col, max_row, max_col = reg.bbox

            region_of_interest = image_list[k][min_row:max_row,min_col:max_col]
            
            temp.append(region_of_interest)
    return temp

##Read the ID of images to discard and put them in a list
##The IDs are stored in text files ctrl_dis.txt and pntr_dis.txt
def get_dis():
    with open('ctrl_dis.txt', 'r') as file:
        text_values_ctrl = file.read().strip()
    values_list_ctrl = text_values_ctrl.split(',')
    values_list_ctrl = [int(value) for value in values_list_ctrl]
    ctrl_dis= np.array(values_list_ctrl)

    with open('pntr_dis.txt', 'r') as file:
        text_values_pntr = file.read().strip()
    values_list_pntr = text_values_pntr.split(',')
    values_list_pntr = [int(value) for value in values_list_pntr]
    pntr_dis= np.array(values_list_pntr)
    return ctrl_dis,pntr_dis

##Discard the unwanted images
def discard(image_list,indices):
    indices_to_remove_set_ctrl = set(indices)
    cell_images = [image for idx, image in enumerate(image_list) if idx not in indices_to_remove_set_ctrl]
    return cell_images

def pad_and_resize_to_square(original_image, target_size):
    
    height, width = original_image.shape[:2]
    if width > height:
        pad_top = (width - height) // 2
        pad_bottom = (width - height) - pad_top
        pad_left = pad_right = 0
    else:
        pad_left = (height - width) // 2
        pad_right = (height - width) - pad_left
        pad_top = pad_bottom = 0
    
    #Padding
    padded_image = np.pad(original_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
    
    #Resizing
    resized_image = resize(padded_image, (target_size, target_size), anti_aliasing=False)
    
    #Computing the scale factor
    larger_dim = max(width, height)
    scale_factor = target_size / larger_dim
    
    return resized_image, scale_factor

##Make sure the augmeneted data/images are unique
def image_to_hash(image):
    image_bytes = image.tobytes()
    return hashlib.md5(image_bytes).hexdigest()

##Function to augment and add to respective lists if unique
def process_and_add_images(images):
    image_hashes = set()
    new_rot = []
    new_flip = []
    for i in images:
        
        original_hash = image_to_hash(i)
        if original_hash not in image_hashes:
            image_hashes.add(original_hash)
            new_rot.append(i)

        #Rotating
        for angle in [90, 180, 270]:
            rotated_image = skimage.transform.rotate(np.copy(i), angle=angle, resize=True, preserve_range=True).astype(i.dtype)
            rotated_hash = image_to_hash(rotated_image)
            if rotated_hash not in image_hashes:
                image_hashes.add(rotated_hash)
                new_rot.append(rotated_image)
        
        #Flipping
        flipped_lr = np.fliplr(np.copy(i)).astype(i.dtype)
        flipped_ud = np.flipud(np.copy(i)).astype(i.dtype)
        flipped_lr_hash = image_to_hash(flipped_lr)
        flipped_ud_hash = image_to_hash(flipped_ud)
        
        if flipped_lr_hash not in image_hashes:
            image_hashes.add(flipped_lr_hash)
            new_flip.append(flipped_lr)
        
        if flipped_ud_hash not in image_hashes:
            image_hashes.add(flipped_ud_hash)
            new_flip.append(flipped_ud)
    
    return new_rot+new_flip

##Shuffle and split to training, validation and test set
def shuffle_and_split(images,factors,labels):
    np.random.seed(seed=49999)
    indices = np.random.permutation(len(images))
    shuffled_images = images[indices]
    shuffled_scaling_factors = factors[indices]
    shuffled_labels = labels[indices]

    total_images = len(shuffled_images)
    train_split_index = int(total_images * 0.8)
    validation_split_index = int(total_images * 0.9)
    
    train_images = shuffled_images[:train_split_index]
    train_scaling_factors = shuffled_scaling_factors[:train_split_index]
    train_labels = shuffled_labels[:train_split_index]

    validation_images = shuffled_images[train_split_index:validation_split_index]
    validation_scaling_factors = shuffled_scaling_factors[train_split_index:validation_split_index]
    validation_labels = shuffled_labels[train_split_index:validation_split_index]

    test_images = shuffled_images[validation_split_index:]
    test_scaling_factors = shuffled_scaling_factors[validation_split_index:]
    test_labels = shuffled_labels[validation_split_index:]

    return train_images,train_scaling_factors,train_labels,validation_images,validation_scaling_factors,validation_labels,test_images,test_scaling_factors,test_labels



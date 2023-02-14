import os
import pandas as pd
import numpy as np
import json
#from bunch import Bunch
from torch.nn import L1Loss

import gc


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config = json.load(config_file)
    #config = Bunch(config)
    return config


def window_search_MIP_2D(image, mask, patch_size, stride, stripe_label):
    img_h, img_w = image.shape
    patch_h, patch_w = patch_size if len(patch_size)==2 else [patch_size[0], patch_size[0]]
    stride_h, stride_w = stride if len(stride)==2 else [stride[0], stride[0]]
    stripe_label_2d = np.ones_like(image) * stripe_label[:,np.newaxis]

    # Stack [image, mask, stripe] together so we can crop simultaneously.
    # `image_fused` in shape (img_h, img_w, 3).
    image_fused = np.array([image, mask, stripe_label_2d]).transpose((1,2,0))
    image_fused_patch_array = np.zeros((0, patch_h, patch_w, 3))

    # The number of patches cropped from the image.
    patch_num_h = int(np.ceil((img_h-patch_h) / stride_h)) + 1
    patch_num_w = int(np.ceil((img_w-patch_w) / stride_w)) if img_w<=800 else int(np.ceil((img_w-patch_w) / stride_w)) + 1

    for idx_h in range(patch_num_h):
        # If cropping outside the boundary, shifting the window back to the boundary.
        start_h = (stride_h*idx_h) if (stride_h*idx_h + patch_h) <= img_h else (img_h-patch_h)
        for idx_w in range(patch_num_w):
            if  idx_w == (patch_num_w-1) and img_w<=800:
                # If image witdh is too short, the last window is shifted to the end so more pixels are used.
                start_w = (img_w-patch_w)
            else:
                start_w = (stride_w*idx_w) if (stride_w*idx_w + patch_w) <= img_w else (img_w-patch_w)

            image_fused_patch = image_fused[start_h : (start_h+patch_h), start_w : (start_w+patch_w), :]
            # image_fused_patch_array in shape (N,Ph,Pw,3), where 3 is [image, mask, stripe].
            image_fused_patch_array = np.append(image_fused_patch_array, image_fused_patch[np.newaxis,...], axis=0)

    return image_fused_patch_array


def window_search_3D(img, masks, BMAmask, imgshape, patch_size, stride):
    '''
    Sliding window for 3D patches extraction.
    img in shape (Pz,Py,Px). 
    masks in shape (3,Pz,Py,Px) (FG, MG, BG).
    BMAmask in shape (Pz,Px).

    Each volume: 30G RAM, 3.5s --> 6G, 4s
    '''
    img_z, img_y, img_x = imgshape
    patch_z, patch_y, patch_x = patch_size
    stride_z, stride_x = stride if len(stride)==2 else [stride, stride]

    # The number of patches cropped from the image.
    #patch_num_z = int(np.ceil((img_z-patch_z) / stride_z)) # Incorrect.
    #patch_num_x = int(np.ceil((img_x-patch_x) / stride_x))
    patch_num_z = int(np.ceil((img_z-patch_z) / stride_z)) + 1
    patch_num_x = int(np.ceil((img_x-patch_x) / stride_x)) + 1

    # Placeholder.
    img_patch_array = np.zeros((patch_num_z*patch_num_x, patch_z, patch_y, patch_x), dtype=np.float32)
    masks_patch_array = np.zeros((patch_num_z*patch_num_x, 3, patch_z, patch_y, patch_x), dtype=np.bool8)
    BMAmask_patch_array = np.zeros((patch_num_z*patch_num_x, patch_z, 1, patch_x), dtype=np.bool8)

    for idx_z in range(patch_num_z):
        # If cropping outside the boundary, shifting the window back to the boundary.
        start_z = (stride_z*idx_z) if (stride_z*idx_z + patch_z) <= img_z else (img_z-patch_z)
        for idx_x in range(patch_num_x):
            start_x = (stride_x*idx_x) if (stride_x*idx_x + patch_x) <= img_x else (img_x-patch_x)
            img_patch_array[idx_z*patch_num_x + idx_x] = img[
                start_z : (start_z+patch_z), :, start_x : (start_x+patch_x)]
            masks_patch_array[idx_z*patch_num_x + idx_x] = masks[
                :, start_z : (start_z+patch_z), :, start_x : (start_x+patch_x)]
            BMAmask_patch_array[idx_z*patch_num_x + idx_x] = BMAmask[
                start_z : (start_z+patch_z), np.newaxis, start_x : (start_x+patch_x)]

    return img_patch_array, masks_patch_array, BMAmask_patch_array


def patch_reconstruction_3D(img_patch_array, imgshape, patch_size, stride):
    '''
    The inverse of sliding window for 3D patches.
    Reconstruct the origial 3D volume from patches.

    img_patch_array in shape (N,Pz,Py,Px).
    imgshape in shape (img_z, img_y, img_x)
    Return img_stack in original image shape.

    Time cost: 9.45s.
    '''
    img_z, img_y, img_x = imgshape
    img_n = img_patch_array.shape[0]
    patch_z, patch_y, patch_x = patch_size
    stride_z, stride_x = stride if len(stride)==2 else [stride, stride]

    # The number of patches cropped from the image.
    patch_num_z = int(np.ceil((img_z-patch_z) / stride_z)) + 1
    patch_num_x = int(np.ceil((img_x-patch_x) / stride_x)) + 1

    assert img_n == patch_num_z*patch_num_x, 'Input patch number inconsistent.'

    # Placeholder.
    img_stack = np.zeros((img_z, img_y, img_x), dtype=np.float32)
    img_stack_count = np.zeros((img_z, img_y, img_x), dtype=np.float32)

    for idx_z in range(patch_num_z):
        # If cropping outside the boundary, shifting the window back to the boundary.
        start_z = (stride_z*idx_z) if (stride_z*idx_z + patch_z) <= img_z else (img_z-patch_z)
        for idx_x in range(patch_num_x):
            start_x = (stride_x*idx_x) if (stride_x*idx_x + patch_x) <= img_x else (img_x-patch_x)
            img_stack[start_z : (start_z+patch_z), :, start_x : (start_x+patch_x)] += img_patch_array[idx_z*patch_num_x + idx_x]
            img_stack_count[start_z : (start_z+patch_z), :, start_x : (start_x+patch_x)] += 1

    assert img_stack_count.min() != 0, 'Every voxel should be predicted at least once.'
    # Normalization.
    img_stack = img_stack / img_stack_count

    return img_stack


class WeightedL1Loss(L1Loss):
    def __init__(self, threshold=0.091554, weight=0.05):
        super().__init__(reduction="none")
        # 0.091554 = 6000/65535
        self.threshold = threshold
        # background weight [0.3, 0.5].
        self.weight = weight

    def forward(self, input, target):
        l1 = super().forward(input, target)
        mask = target < self.threshold
        l1[mask] = l1[mask] * self.weight

        return l1.mean()

def maskpatch_reconstruction(predpatches, original_size, patch_size, stride):
    '''
    The inverse operation of window_search_MIP_2D(), 
    which reconstructs the whole prediciton from a set of prediction patches.

    predpatches in shape (patch_num_h*patch_num_w, patch_h, patch_w).
    return predmask_whole in shape (img_h, img_w).
    '''

    img_h, img_w = original_size
    patch_h, patch_w = patch_size if len(patch_size)==2 else [patch_size, patch_size]
    stride_h, stride_w = stride if len(stride)==2 else [stride, stride]

    # Strore predict values.
    predmask_whole = np.zeros((img_h,img_w))
    predmask_whole_count = np.zeros((img_h,img_w))

    # The number of patches cropped from the image.
    patch_num_h = int(np.ceil((img_h-patch_h) / stride_h)) + 1
    patch_num_w = int(np.ceil((img_w-patch_w) / stride_w)) if img_w<=800 else int(np.ceil((img_w-patch_w) / stride_w)) + 1

    for idx_h in range(patch_num_h):
        # If cropping outside the boundary, shifting the window back to the boundary.
        start_h = (stride_h*idx_h) if (stride_h*idx_h + patch_h) <= img_h else (img_h-patch_h)
        for idx_w in range(patch_num_w):
            if  idx_w == (patch_num_w-1) and img_w<=800:
                # If image witdh is too short, the last window is shifted to the end so more pixels are used.
                start_w = (img_w-patch_w)
            else:
                start_w = (stride_w*idx_w) if (stride_w*idx_w + patch_w) <= img_w else (img_w-patch_w)

            predmask_whole[start_h : (start_h+patch_h), start_w : (start_w+patch_w)] += predpatches[idx_h*patch_num_w + idx_w]
            predmask_whole_count[start_h : (start_h+patch_h), start_w : (start_w+patch_w)] += 1

    predmask_whole = predmask_whole/predmask_whole_count

    return predmask_whole

def input_single_img(path,label,threshold=0.5):
    artifacted_img = cv2.imread(path,0)
    gapmask =np.zeros(artifacted_img.shape)
    ori_img = np.copy(artifacted_img)
    for i in range(artifacted_img.shape[0]):
        artifacted_img[i,:]=artifacted_img[i,:]* (1-int(label[i]>threshold))
        gapmask[i,:]=int(label[i]>threshold)
    return artifacted_img,ori_img,gapmask

def single_img_reconstruct(img_array,width,height,patch_size,stride):
    # width is image height while height is width...
    img = np.zeros([width,height])
    mask =np.zeros([width,height])
    pw= (patch_size-stride)-(width-stride)%(patch_size-stride)
    ph=(patch_size-stride)-(height-stride)%(patch_size-stride)

    for i in range((width+pw-stride)//(patch_size-stride)):
        for j in range((height+ph-stride)//(patch_size-stride)):
            index_x = i*(patch_size-stride)+patch_size
            index_y = j*(patch_size-stride)+patch_size
            if index_x>= width:
                index_x = width
            if index_y>= height:
                index_y = height
            img[index_x - patch_size : index_x, index_y - patch_size : index_y] += img_array[
                    int(i * (height+ph-stride) // (patch_size-stride) + j),:,:,0]
            mask[index_x - patch_size : index_x, index_y - patch_size : index_y] += 1
    img = np.divide(img, mask)
    return img

def locate_cont_stripe(label_bin, invert=False):
    '''
    Given an binary array, locate and count continuous '1's.
    '''
    stripe_start_list = []
    stripe_list = []

    last_label = -1
    noise_num = 0
    for label_idx, label in enumerate(label_bin):
        # Last label check
        if (label_idx == len(label_bin)-1) and label==1:
            noise_num = noise_num + 1
            stripe_list.append(noise_num)
            stripe_start_list.append(label_idx-noise_num)
            break

        if label == 0:
            # Count as one stripe. Then reset last_label and noise_num.
            if last_label == 1:
                stripe_list.append(noise_num)
                stripe_start_list.append(label_idx-noise_num)
                last_label = 0
                noise_num = 0
        elif label == 1:
            noise_num = noise_num + 1
            last_label = 1
        else:
            raise ValueError('label_bin should be {0, 1}.')

    stripe_list = np.array(stripe_list)
    hist = np.histogram(stripe_list, range=(min(stripe_list), max(stripe_list)+1), 
                 bins=(max(stripe_list)-min(stripe_list)) + 1)
    
    return stripe_list, stripe_start_list, hist

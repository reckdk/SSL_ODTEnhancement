import os
import random
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, binary_closing
from glob import glob
import cv2
import skimage.io as skio
import torch
import h5py

from utils import window_search_3D

# DataLoader(dataset_train, batch_size=None, shuffle=False, num_workers=4)
# For future multiple GPUs training.
def collate_batch(batchdata, config):
    if config.stripeloss_only:
        img_mask_patch = batchdata[0][0][0].copy()
        loss_mask = batchdata[0][0][1].copy()
        mask_patch_gt = batchdata[0][1].copy()
        for [img_mask_patch_cur, loss_mask_cur], mask_patch_gt_cur in batchdata[1:]:
            img_mask_patch = np.concatenate((img_mask_patch, img_mask_patch_cur), axis=0)
            loss_mask = np.concatenate((loss_mask, loss_mask_cur), axis=0)
            mask_patch_gt = np.concatenate((mask_patch_gt, mask_patch_gt_cur), axis=0)
        return torch.tensor(img_mask_patch), torch.tensor(loss_mask), torch.tensor(mask_patch_gt)
    else:
        img_mask_patch = batchdata[0][0].copy()
        mask_patch_gt = batchdata[0][1].copy()
        for img_mask_patch_cur, mask_patch_gt_cur in batchdata[1:]:
            img_mask_patch = np.concatenate((img_mask_patch, img_mask_patch_cur), axis=0)
            mask_patch_gt = np.concatenate((mask_patch_gt, mask_patch_gt_cur), axis=0)
        return torch.tensor(img_mask_patch), torch.tensor(mask_patch_gt)

class ODT3D_Dataset(torch.utils.data.Dataset):
    def __init__(self, config, shuffle=True):
        self.shuffle = shuffle
        self.dataset_dir = config['dataset_dir']
        '''
        self.imglist_train = glob(os.path.join(self.dataset_dir, 'train/*_shear.tif'))
        self.imglist_train.sort()
        self.imglist_val = glob(os.path.join(self.dataset_dir, 'val/*_shear.tif'))
        self.imglist_val.sort()
        # Due to data insufficiency, the samples in val and test are same but evaluated differently.
        self.imglist_test = glob(os.path.join(self.dataset_dir, 'val/*_shear.tif'))
        self.imglist_test.sort()
        '''

        self.hf = h5py.File(os.path.join(self.dataset_dir, 'ODT3D_Dataset.h5'), 'r')
        #self.hf = h5py.File(os.path.join(self.dataset_dir, 'ODT3D_Dataset_small.h5'), 'r')
        #self.hf = h5py.File(os.path.join(self.dataset_dir, 'ODT3D_Dataset_tiny.h5'), 'r',  driver='core')
        #self.hf = h5py.File(os.path.join(self.dataset_dir, 'ODT3D_Dataset_tiny_val.h5'), 'r')
        self.imglist_train = ['train/' + x for x in list(self.hf['train'].keys())]
        self.imglist_train.sort()
        self.imglist_val = ['val/' + x for x in list(self.hf['val'].keys())]
        self.imglist_val.sort()
        self.imglist_test = ['test/' + x for x in list(self.hf['test'].keys())]
        self.imglist_test.sort()
        
        self.batch_size = config['batch_size']
        self.img_perbatch = 2
        self.patch_perimg = self.batch_size // self.img_perbatch 
        assert self.batch_size/self.img_perbatch == self.batch_size//self.img_perbatch, (
            'The reminder should be zero.')
        [self.patch_z, self.patch_y, self.patch_x] = config['patch_size'] # (N,H,W) or axis (z,y,x).
        self.patch_y_bottom = 300

        # Augmentation initializaiton.
        self.aug_intensity = True
        self.aug_flip = True
        ## Flip flags. [1,2,3] --> [z,y,x].
        #self.flip_op = [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
        ## Flip flags. [2,3,4] --> [z,y,x].
        #self.flip_op_masks = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        # Do not flip y-axis because syn-process is y-based.
        self.flip_op = [[], [1], [3], [1, 3]]
        self.flip_op_masks = [[], [2], [4], [2, 4]]
        # Morph setting.
        
        self.struct_dil = np.zeros((3,2,2,2), dtype=np.bool8)
        self.struct_dil[1] = np.ones((2,2,2), dtype=np.bool8)
        self.broken_factor = 0.10
        '''
        self.struct_dil = np.zeros((3,3,3,3), dtype=np.bool8)
        self.struct_dil[1] = np.ones((3,3,3), dtype=np.bool8)
        self.broken_factor = 0.03 # 0.10 # 0.15 too much.
        '''

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.imglist_train)
    
    def __len__(self):
        return len(self.imglist_train) // self.img_perbatch

    def __getitem__(self, index):
        imglist_cur = self.imglist_train[index*self.img_perbatch : (index+1) * self.img_perbatch]
        X, loss_mask, Y = self.getbatch(imglist_cur)

        # Padding for UNet (32x).
        input_paddding = np.zeros((self.batch_size, self.patch_z, 4, self.patch_x), dtype=np.float32)
        X = np.concatenate((X, input_paddding), axis=2)
        loss_mask = np.concatenate((loss_mask, input_paddding), axis=2)
        Y = np.concatenate((Y, input_paddding), axis=2)

        return (torch.from_numpy(np.expand_dims(X,1)),
                torch.from_numpy(np.expand_dims(loss_mask,1)),
                torch.from_numpy(np.expand_dims(Y,1)))

    def getbatch(self, imglist):
        img_patch = np.zeros((self.batch_size, self.patch_z, self.patch_y, self.patch_x), dtype=np.float32) # placeholder.
        # Masked loss. 0: voxel not for training; 1: for training. 
        loss_mask = np.zeros((self.batch_size, self.patch_z, self.patch_y, self.patch_x), dtype=np.float32)
        img_patch_gt = np.zeros((self.batch_size, self.patch_z, self.patch_y, self.patch_x), dtype=np.float32)

        # Extract image patches from each image sample.
        for img_idx, img_path in enumerate(imglist):

            '''
            # Folder dataset.
            imgname = os.path.basename(img_path)
            mask_path = os.path.join(self.dataset_dir, 'masks', imgname.replace('.tif', '_masks.npy'))
            BMAmask_path = os.path.join(self.dataset_dir, 'masks', imgname.replace('.tif', '_BMA.npy'))
            img = skio.imread(img_path, plugin='tifffile')
            img = img.astype(np.float32)/65535 # Normalization.
            masks = np.load(mask_path) 
            BMAmask = np.load(BMAmask_path) 
            '''
            # HDF5 dataset.
            img = np.array(self.hf.get('{}/volume'.format(img_path)))
            img = img.astype(np.float32)/65535 # Normalization.
            # masks: binary FG/MG/BG in shape (3,Pz,Py,Px).
            masks = np.array(self.hf.get('{}/masks'.format(img_path)))
            # BMAmask: binary in shape (Pz,Px).
            BMAmask = np.array(self.hf.get('{}/BMAmask'.format(img_path)))
            img_z, img_y, img_x = img.shape

            # Randomized sampling parameters.
            # Extract 2*self.patch_perimg patches. The additional one for synthesis.
            patches_start_x = np.random.randint(0, img_x-self.patch_x, (2, self.patch_perimg))
            patches_start_z = np.random.randint(0, img_z-self.patch_z, (2, self.patch_perimg))
            # Extract a patch of [150,250] in height and randomly insert into the top-300 area (y).
            # To be efficient, all systhesis patches in a batch have the same size and location.
            syn_upper = np.random.randint(400, 450)
            syn_down = np.random.randint(600, 650)
            # Y-coordinates of synthsis signals.
            syn_y = np.random.randint(0, 300 - (syn_down-syn_upper))

            img_patch_perimg, loss_mask_perimg, img_patch_gt_perimg = self.getimagepatch(
                    img, masks, BMAmask, patches_start_x, patches_start_z,syn_upper, syn_down, syn_y)

            img_patch[img_idx*self.patch_perimg : (img_idx+1)*self.patch_perimg] = img_patch_perimg
            img_patch_gt[img_idx*self.patch_perimg : (img_idx+1)*self.patch_perimg] = img_patch_gt_perimg
            loss_mask[img_idx*self.patch_perimg : (img_idx+1)*self.patch_perimg] = loss_mask_perimg

        ''' Flip is fast enough so no need to be put here.
        # For efficiency, put flip augmentation here.
        # All samples of a batch have the same flips.
        flip_num = np.random.randint(4)
        if flip_num != 0:
            flip_idx = np.random.choice(3, size=flip_num, replace=False)
            img_patch = np.flip(img_patch, axis=flip_idx)
            loss_mask = np.flip(loss_mask, axis=flip_idx)
            img_patch_gt = np.flip(img_patch_gt, axis=flip_idx)
        '''

        # GT is masked here while prediction is masked in model.
        return img_patch, loss_mask, img_patch_gt*loss_mask
        #return img_patch, loss_mask, img_patch_gt # For debug.

    def getimagepatch(self, img, masks, BMAmask, patches_start_x,
            patches_start_z, syn_upper, syn_down, syn_y):
        '''
        Process an ODT volume and masks into input patches.
        '''
        patch_perimg = patches_start_x.shape[1]
        synpatches_height = syn_down - syn_upper
        # Placeholder.
        img_patch_perimg = np.zeros((patch_perimg, self.patch_z, self.patch_y, self.patch_x), dtype=np.float32)
        masks_patch_perimg = np.zeros((patch_perimg, 3, self.patch_z, self.patch_y, self.patch_x), dtype=np.bool8)
        BMAmask_patch_perimg = np.zeros((patch_perimg, self.patch_z, 1, self.patch_x), dtype=np.bool8)

        loss_mask_perimg = np.ones((patch_perimg, self.patch_z, self.patch_y, self.patch_x), dtype=np.bool8)
        img_patch_gt_perimg = np.zeros((patch_perimg, self.patch_z, self.patch_y, self.patch_x), dtype=np.float32)

        img_patch_exbot_perimg = np.zeros((patch_perimg, self.patch_z, synpatches_height, self.patch_x), dtype=np.float32)
        masks_patch_exbot_perimg = np.zeros((patch_perimg, 3, self.patch_z, synpatches_height, self.patch_x), dtype=np.bool8)
        BMAmask_patch_exbot_perimg = np.zeros((patch_perimg, self.patch_z, 1, self.patch_x), dtype=np.bool8)

        # Extract patches.
        for i in range(patch_perimg):
            img_patch_perimg[i] = img[patches_start_z[0,i] : patches_start_z[0,i]+self.patch_z,
                                      :, patches_start_x[0,i] : patches_start_x[0,i]+self.patch_x].copy()
            masks_patch_perimg[i] = masks[:, patches_start_z[0,i] : patches_start_z[0,i]+self.patch_z,
                                          :, patches_start_x[0,i] : patches_start_x[0,i]+self.patch_x]
            BMAmask_patch_perimg[i,:,0,:] = BMAmask[patches_start_z[0,i] : patches_start_z[0,i]+self.patch_z,
                                                    patches_start_x[0,i] : patches_start_x[0,i]+self.patch_x]
             # Extract extra bottom-patches for synthesis.                                       
            img_patch_exbot_perimg[i] = img[patches_start_z[1,i] : patches_start_z[1,i]+self.patch_z,
                                            syn_upper:syn_down,
                                            patches_start_x[1,i] : patches_start_x[1,i]+self.patch_x]
            masks_patch_exbot_perimg[i] = masks[:, patches_start_z[1,i] : patches_start_z[1,i]+self.patch_z,
                                                syn_upper:syn_down,
                                                patches_start_x[1,i] : patches_start_x[1,i]+self.patch_x]
            BMAmask_patch_exbot_perimg[i,:,0,:] = BMAmask[patches_start_z[1,i] : patches_start_z[1,i]+self.patch_z,
                                                            patches_start_x[1,i] : patches_start_x[1,i]+self.patch_x]
            #print(i, img_patch_perimg.max(), masks_patch_perimg[i].sum((1,2,3)))

        if self.aug_intensity:
            # Intensity jittering augmentation.
            # All voxels jittering between [0.8, 1.2].
            ## For efficiency, only jittering on extracted patches, not full volume.
            jitter_factor = np.random.rand(
                patch_perimg, self.patch_z, self.patch_y, self.patch_x)*0.4 + 0.8
            img_patch_perimg *= jitter_factor

            # Prev: Syn-voxels suppression between [0.3, 0.7].
            # Syn-voxels suppression between [0.05, 0.45].
            supp_max = np.random.rand(
                patch_perimg, self.patch_z, synpatches_height, self.patch_x)*0.4 + 0.05
            # Although suppress all voxels but only FG' is used during syn.
            img_patch_exbot_perimg *= supp_max

        if self.aug_flip:
            # Random flip on image patches.
            flip_idx = np.random.randint(4)
            if flip_idx != 0:
                img_patch_perimg = np.flip(img_patch_perimg, axis=self.flip_op[flip_idx])
                masks_patch_perimg = np.flip(masks_patch_perimg, axis=self.flip_op_masks[flip_idx])
                BMAmask_patch_perimg = np.flip(BMAmask_patch_perimg, axis=self.flip_op[flip_idx])
            # Random flip on syn-image patches.
            flip_idx = np.random.randint(4)
            if flip_idx != 0:
                img_patch_exbot_perimg = np.flip(img_patch_exbot_perimg, axis=self.flip_op[flip_idx])
                masks_patch_exbot_perimg = np.flip(masks_patch_exbot_perimg, axis=self.flip_op_masks[flip_idx])
                BMAmask_patch_exbot_perimg = np.flip(BMAmask_patch_exbot_perimg, axis=self.flip_op[flip_idx])

        # Process the surface part (y-->700).
        # Input_B := I_B
        # GT_B := I_B*FG_B
        img_patch_gt_perimg[:,:,-self.patch_y_bottom:,:] = (img_patch_perimg[:,:,-self.patch_y_bottom:,:]
                                                            * masks_patch_perimg[:,0,:,-self.patch_y_bottom:,:])
        # Lossmask_B := ~MG_B*~BMA_B
        loss_mask_perimg[:,:,-self.patch_y_bottom:,:] = (~masks_patch_perimg[:,1,:,-self.patch_y_bottom:,:]
                                                         * ~BMAmask_patch_perimg)

        # Process the deep part (y-->0).
        # Input_T :=
        #   for syn-area: I'_B*FG'_B + I_T*~FG'_B
        #   for original: I_T
        synimg_FG = img_patch_exbot_perimg*masks_patch_exbot_perimg[:,0,...]
        img_patch_perimg[:,:, syn_y : syn_y+synpatches_height,:] *= ~masks_patch_exbot_perimg[:,0,...]
        img_patch_perimg[:,:, syn_y : syn_y+synpatches_height,:] += synimg_FG
        # GT_T := I'_B*FG'_B if synarea else 0 (assuming no signal here).
        img_patch_gt_perimg[:,:, syn_y : syn_y+synpatches_height,:] = synimg_FG.copy()
        # Lossmask_T [300,400) is 0. (Due to uncertainty, loss here is ignored.)
        loss_mask_perimg[:,:,self.patch_y_bottom:400,:] = 0
        # Lossmask_T for syn-area: ~BMA'_B (No MG'_B is injected so only ~BMA'_B.)
        loss_mask_perimg[:,:, syn_y : syn_y+synpatches_height,:] *= ~BMAmask_patch_exbot_perimg
        #print('--> ', img_patch_perimg.max(), loss_mask_perimg.sum(), img_patch_gt_perimg.max())

        # Augmentation to simulate broken vessels for all FG.
        # Todo: jietter but train on original value (to boost smooth?)
        if self.aug_intensity:
            # FG mask of img_patch_perimg.
            img_FG_masks = np.zeros_like(img_patch_perimg, dtype=np.bool8)
            # Bottom FG := FG_B
            img_FG_masks[:,:,-self.patch_y_bottom:,:] = masks_patch_perimg[:,0,:,-self.patch_y_bottom:,:]
            # Top (syn. area) FG := FG'_B
            img_FG_masks[:,:,syn_y : syn_y+synpatches_height,:] = masks_patch_exbot_perimg[:,0,...]

            '''
            # AP-1: random suppression.
            # Change 10~40% FG voxels into [0,0.15] [20000-->3000]
            supp_ratio = np.random.rand()*0.3 + 0.4
            #supp_ratio = np.random.rand()*0.2 + 0.7
 
            #aug_max = np.random.rand()*0.1 + 0.1 # jit.15
            #aug_max = np.random.rand()*0.15 # jitmax.15
            aug_max = np.random.rand()*0.40 # jitmax.30
            aug_mask = np.random.rand(
                patch_perimg, self.patch_z, self.patch_y, self.patch_x)>supp_ratio
            '''
            # AP-2: Morph-based suppresion. Like real broken vessels.
            aug_max = np.random.rand()*0.30 # jitmax.30, jitmax.15
            aug_mask = np.random.rand(
                patch_perimg, self.patch_z, self.patch_y, self.patch_x)<self.broken_factor
            # Morphological op.
            aug_mask = binary_dilation(aug_mask, self.struct_dil)
            #aug_mask = binary_closing(aug_mask, self.struct_dil) # 650ms

            # Only suppress FG voxels.
            aug_mask *= img_FG_masks
            aug_factor = np.random.rand(
                patch_perimg, self.patch_z, self.patch_y, self.patch_x)*aug_max
            img_patch_perimg = img_patch_perimg*(~aug_mask) + img_patch_perimg*aug_mask*aug_factor

        # May be ignored for efficiency.
        img_patch_perimg = np.clip(img_patch_perimg, a_min=0, a_max=1)
        img_patch_gt_perimg = np.clip(img_patch_gt_perimg, a_min=0, a_max=1)

        return img_patch_perimg, loss_mask_perimg, img_patch_gt_perimg


class ODT3D_Dataset_val(ODT3D_Dataset):
    def __init__(self, config):
        super().__init__(config)
        self.aug_intensity = False
        self.aug_flip = False
    
    def __len__(self):
        return len(self.imglist_val)

    def __getitem__(self, index):
        img_path = self.imglist_val[index]
        X, loss_mask, Y = self.getbatch(img_path)

        # Padding for UNet (32x).
        input_paddding = np.zeros((self.batch_size, self.patch_z, 4, self.patch_x), dtype=np.float32)
        X = np.concatenate((X, input_paddding), axis=2)
        loss_mask = np.concatenate((loss_mask, input_paddding), axis=2)
        Y = np.concatenate((Y, input_paddding), axis=2)

        return (torch.from_numpy(np.expand_dims(X,1).astype(np.float32)),
                torch.from_numpy(np.expand_dims(loss_mask,1)),
                torch.from_numpy(np.expand_dims(Y,1)))

    # Override.
    def getbatch(self, img_path):
        img_patch = np.zeros((self.batch_size, self.patch_z, self.patch_y, self.patch_x), dtype=np.float32) # placeholder.
        # Masked loss. 0: voxel not for training; 1: for training. 
        loss_mask = np.zeros((self.batch_size, self.patch_z, self.patch_y, self.patch_x), dtype=np.float32)
        img_patch_gt = np.zeros((self.batch_size, self.patch_z, self.patch_y, self.patch_x), dtype=np.float32)

        # Each batch for each sample.
        # HDF5 dataset.
        img = np.array(self.hf.get('{}/volume'.format(img_path)))
        img = img.astype(np.float32)/65535 # Normalization.
        # masks: binary FG/MG/BG in shape (3,Pz,Py,Px).
        masks = np.array(self.hf.get('{}/masks'.format(img_path)))
        # BMAmask: binary in shape (Pz,Px).
        BMAmask = np.array(self.hf.get('{}/BMAmask'.format(img_path)))
        img_z, img_y, img_x = img.shape

        # Fixed sampling parameters. For UNet bz20 on Awake_07102021 (500*1000).
        # Extract 2*self.batch_size patches. The additional one for synthesis.
        # TODO: this may be too small for evaluation.

        # patches_start_x in shape (2, 20).
        patches_start_x = np.array([
            np.repeat(np.arange(400, 560, 32), repeats=4),
            np.repeat(np.arange(600, 760, 32), repeats=4)])
        patches_start_z = np.array([
            np.repeat(np.arange(200, 328, 32)[np.newaxis,], repeats=5, axis=0).flatten(),
            np.repeat(np.arange(200, 328, 32)[np.newaxis,], repeats=5, axis=0).flatten()])
        syn_upper = 450
        syn_down = 650
        syn_y = 100

        # Fixed valset and no augmentation.
        img_patch_perimg, loss_mask_perimg, img_patch_gt_perimg = self.getimagepatch(
                img, masks, BMAmask, patches_start_x, patches_start_z,
                syn_upper, syn_down, syn_y)

        return img_patch_perimg, loss_mask_perimg, img_patch_gt_perimg*loss_mask_perimg

class ODT3D_Dataset_vis(ODT3D_Dataset_val):
    ''' Dataset only used for visualization.
    '''
    def __init__(self, config):
        super().__init__(config)
        self.aug_intensity = True # Augmentation when fetching data.
        self.aug_flip = False


class ODT3D_Dataset_test(ODT3D_Dataset):
    def __init__(self, config):
        super().__init__(config)
    
    def __len__(self):
        return len(self.imglist_test)

    def __getitem__(self, index):
        img_path = self.imglist_test[index]
        X, loss_mask, Y = self.getbatch(img_path)

        # Padding for UNet (32x).
        input_paddding = np.zeros((self.batch_size, self.patch_z, 4, self.patch_x), dtype=np.float32)
        X = np.concatenate((X, input_paddding), axis=2)
        loss_mask = np.concatenate((loss_mask, input_paddding), axis=2)
        Y = np.concatenate((Y, input_paddding), axis=2)

        return (torch.from_numpy(np.expand_dims(X,1)),
                torch.from_numpy(np.expand_dims(loss_mask,1)),
                torch.from_numpy(np.expand_dims(Y,1)))

    def getbatch(self, img_path):
        # HDF5 dataset.
        img = np.array(self.hf.get('{}/volume'.format(img_path)))
        img = img.astype(np.float32)/65535 # Normalization.
        # masks: binary FG/MG/BG in shape (3,Pz,Py,Px).
        masks = np.array(self.hf.get('{}/masks'.format(img_path)))
        # BMAmask: binary in shape (Pz,Px).
        BMAmask = np.array(self.hf.get('{}/BMAmask'.format(img_path)))

        img_patch, loss_mask, img_patch_gt = self.getimagepatch_test(img, masks, BMAmask)

        return img_patch, loss_mask, img_patch_gt*loss_mask

    def getimagepatch_test(self, img, masks, BMAmask):
        '''
        Process an ODT volume and masks into input patches.
        For testset, the GT is 
        '''
        # img_patch_array in shape (N, patch_z, patch_y, patch_x).
        # masks_patch_array in shape (N, 3, patch_z, patch_y, patch_x).
        # BMAmask_patch_array in shape (N, patch_z, 1, patch_x).
        img_patch_array, masks_patch_array, BMAmask_patch_array = window_search_3D(
            img,
            masks, 
            BMAmask,
            img.shape
            [self.patch_z, self.patch_y, self.patch_x],
            [self.patch_z//2, self.patch_x//2])
        # For efficiency, only evaluate on the z-middle part.
        patchnum_all = BMAmask_patch_array.shape[0]
        # TODO: may be too many. Chunk into batches.
        img_patch_array = img_patch_array[patchnum_all//4 : patchnum_all - patchnum_all//4]
        masks_patch_array = masks_patch_array[patchnum_all//4 : patchnum_all - patchnum_all//4]
        BMAmask_patch_array = BMAmask_patch_array[patchnum_all//4 : patchnum_all - patchnum_all//4]
        patchnum_perimg = img_patch_array.shape[0]

        # Placeholder.
        img_patch_perimg = np.zeros((patchnum_perimg, self.patch_z, self.patch_y, self.patch_x), dtype=np.float32)
        img_patch_gt_perimg = np.zeros((patchnum_perimg, self.patch_z, self.patch_y, self.patch_x), dtype=np.float32)
        loss_mask_perimg = np.ones((patchnum_perimg, self.patch_z, self.patch_y, self.patch_x), dtype=np.bool8)

        # Input := I
        img_patch_perimg = img_patch_array.copy() # Keep it simple.
        # GT := I*FG + 0*BG
        img_patch_gt_perimg = img_patch_array * masks_patch_array[:,0,...]
        # Lossmask_T := ~MG_T
        loss_mask_perimg = ~masks_patch_array[:,1,...]
        # Lossmask_B := ~MG_B*~MBA_B
        loss_mask_perimg[:,:,-self.patch_y_bottom:,:] *= ~BMAmask_patch_array

        return img_patch_perimg, loss_mask_perimg, img_patch_gt_perimg


if __name__ == "__main__":
    pass
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from skimage.transform import AffineTransform
import cv2
import numpy as np
import torch
def to_multi_input(A_img,n_inputs,transform):
    scales = np.linspace(0.999, 1.001, 30)
    scale = np.random.choice(scales)
    rots = np.linspace(-1 * (np.pi / 300), np.pi / 300, 30)
    rot = np.random.choice(rots)
    shears = np.linspace(-1 * (np.pi / 300), np.pi / 300, 30)
    shear = np.random.choice(shears)
    shifts = np.linspace(-1 * 2, 2, 30)
    shift = np.random.choice(shifts)
    tform = AffineTransform(scale=scale, rotation=rot, shear=shear, translation=shift)
    matrix = np.linalg.inv(tform.params)[:2]
    matrixinv = tform.params[:2]
    A_img = np.asarray(A_img)
    A_all = []
    for n in range(n_inputs):
        if (len(A_all) == 0):
            A_img_transformed = cv2.warpAffine(A_img, matrix, (A_img.shape[0], A_img.shape[1]))
        else:
            A_img_transformed = cv2.warpAffine(A_img_prev, matrix, (A_img.shape[0], A_img.shape[1]))
        A_img_prev = A_img_transformed.copy()
        A_img_transformed = Image.fromarray(A_img_transformed)
        # A_img_transformed.show()
        A = transform(A_img_transformed)
        A_all.append(A)
    A_all = torch.cat(A_all, 0)
    return A_all
class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation



        n_inputs = int(self.opt.input_nc/3)

        A = to_multi_input(A_img,n_inputs,self.transform_A)
        B = to_multi_input(B_img,n_inputs,self.transform_B)


        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

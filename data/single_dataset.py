from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from skimage.transform import AffineTransform
import cv2
import numpy as np
import torch
class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        n_inputs = int(self.opt.input_nc/3)






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
            A_img_transformed=Image.fromarray(A_img_transformed)
            # A_img_transformed.show()
            A = self.transform(A_img_transformed)
            A_all.append(A)
        A_all = torch.cat(A_all,0)






        return {'A': A_all, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

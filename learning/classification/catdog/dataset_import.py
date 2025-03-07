import cv2
import glob
import torch
from torch.utils.data.dataset import Dataset


class dataset_import(Dataset):

    def __init__(self, folder_path):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.image_list = glob.glob(folder_path + "*")
        # Calculate len
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        # Get image name from the path
        single_image_path = self.image_list[index]
        # Open image
        # im_as_im = Image.open(single_image_path) #original
        img_dim = 50
        im_gray = cv2.imread(single_image_path, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (img_dim, img_dim))
        # Do some operations on image
        # Convert to numpy, dim = img_dimximg_dim

        # im_as_np = np.asarray(im_as_im)/255 #original
        # sudo apt-get install fdupes

        # Add channel dimension, dim = 1ximg_dimximg_dim
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension
        # im_as_np = np.expand_dims(im_as_np, 0)

        # Transform image to tensor, change data type
        # im_as_ten = torch.from_numpy(im_as_np).float() #original
        im_as_ten = torch.from_numpy(im_gray / 255).float()
        # Get label(class) of the image based on the file name
        # Let cat = 0, dog = 1

        # class_indicator_location = single_image_path.rfind('dog')
        # label = int(single_image_path[class_indicator_location+2:class_indicator_location+3])
        if "cat" in single_image_path:
            label = 0
        else:
            label = 1
        return (im_as_ten, label)

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import torch.nn as nn
    import torch
    from torch.utils.data import DataLoader
    from dataset_import import dataset_import
    from model import Model

    import matplotlib.pyplot as plt

    dataset_train = dataset_import("./dataset/train/")
    dataset_test = dataset_import("./dataset/test/")

    print(dataset_train.data_len)
    print(dataset_train.image_list[0])
    print(dataset_test.data_len)
    print(dataset_test.image_list[0])

    imt, label = dataset_train.__getitem__(7520)
    print(imt.size())
    print(label)
    plt.imshow(imt)

    # fake data
    f_data = torch.rand((8, 100, 100)).to("cuda")
    outp = Model(f_data)
    print(f_data.shape)
    print(outp.shape)

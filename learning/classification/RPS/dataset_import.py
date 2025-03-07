import cv2
import glob
import torch
from torch.utils.data.dataset import Dataset


class dataset_import(Dataset):
    def __init__(self, folder_path):
        # Get image list
        self.image_list = glob.glob(folder_path + "*/*")

        # Calculate len
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        # Get image name from the path
        single_image_path = self.image_list[index]

        # Open image
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
        im_as_ten = torch.from_numpy(im_gray / 255).float()

        # Get label(class) of the image based on the file name
        # Let cat = 0, dog = 1

        if "rock" in single_image_path:
            label = 0
        elif "paper" in single_image_path:
            label = 1
        elif "scissors" in single_image_path:
            label = 2

        return (im_as_ten, label)

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    import glob
    from dataset_import import dataset_import
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # img_path = glob.glob('./data/test/*/'+'*')
    # print(img_path)

    data_test = dataset_import("./data/train/")
    dataloader_test = DataLoader(dataset=data_test, batch_size=16, shuffle=True)

    for img, label in dataloader_test:
        print(img.shape)
        print(label)
        break

    plt.imshow(img[2])
    plt.show()

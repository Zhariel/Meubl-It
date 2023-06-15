import torchvision
from diffusion import load_env_variables
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import os
import torch
from PIL import Image


# def custom_dataset(IMG_SIZE=64, train_split=80, image_paths=[(os.path.join('assets', 'test3.png'), 0)]):
def custom_dataset(IMG_SIZE=64, train_split=80, image_paths=[(os.path.join('assets', 'sample.png'), 0)]):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
    data_transform = transforms.Compose(data_transforms)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, transform=None):
            self.dataset_directory = os.path.join("D:", "PA_Data", "segmented", "ade20k", "training",
                                                  "shopping_and_dining")
            self.image_paths = image_paths
            self.transform = transform
            self.test = self.gather_links()

        def __getitem__(self, index):
            image_path, annotation = self.image_paths[index]

            # samplebox = (50, 50, 250, 250)  # Temporary
            samplebox = (0, 285, 440, 725) # Temporary
            image = Image.open(image_path).convert('RGB')
            coord, newbox = self.crop_largest_square_around_point(*image.size, samplebox, IMG_SIZE)
            image = image.crop(coord)
            # add finding new hole coordinates after resizing
            if self.transform is not None:
                image = self.transform(image)
            return image, annotation, newbox

        def __len__(self):
            return len(self.image_paths)

        # @jit
        def crop_largest_square_around_point(self, width, height, box, IMG_SIZE):
            box_side = abs(box[0] - box[2])
            point = (box[0] + box_side // 2, box[1] + box_side // 2)
            square_size = min(width, height)

            left = max(0, point[0] - square_size // 2)
            top = max(0, point[1] - square_size // 2)
            right = min(width, left + square_size)
            bottom = min(height, top + square_size)

            if right - left < square_size:
                left = max(0, right - square_size)
            if bottom - top < square_size:
                top = max(0, bottom - square_size)

            scale_factor = IMG_SIZE / square_size

            nleft = int((box[0] - left) * scale_factor)
            ntop = int((box[1] - top) * scale_factor)
            nright = int((box[2] - left) * scale_factor)
            nbottom = int((box[3] - top) * scale_factor)

            return (left, top, right, bottom), (nleft, ntop, nright, nbottom)

        def gather_links(self):
            links = []
            dirs = os.listdir(self.dataset_directory)

            for dir in dirs:
                subdirs_folder = os.path.join(self.dataset_directory, dir)

                items = os.listdir(subdirs_folder)
                for i in items:
                    if not '.' in i:
                        links.append((i + '.json', i + '.jpg'))

            return links

    dataset = CustomDataset(image_paths, transform=data_transform)
    return dataset
    # test_split = 100 - train_split
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_split, test_split])

    # return train_set, val_set

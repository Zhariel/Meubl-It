import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from numba import jit
from PIL import Image
import numpy as np
import random
import torch
import json


class ListDataset(Dataset):
    def __init__(self, images, masks, labels, targets, time, device):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.targets = targets
        self.time = time
        self.device = device

    def shuffle(self):
        indices = list(range(len(self.targets)))
        random.shuffle(indices)
        self.images = [self.images[i] for i in indices]
        self.masks = [self.masks[i].to(self.device) for i in indices]
        self.labels = [self.labels[i].to(self.device) for i in indices]
        self.targets = [self.targets[i].to(self.device) for i in indices]
        self.time = [self.time[i].to(self.device) for i in indices]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        i = self.images[index].to(self.device)
        m = self.masks[index].to(self.device)
        l = self.labels[index].to(self.device)
        y = self.targets[index].to(self.device)
        t = self.time[index].to(self.device)

        return i, m, l, y, t


# @jit
def gather_links(folder="shopping_and_dining"):
    path = os.path.join("D:", "PA_Data", "segmented", "ade20k", "training", folder)
    links = []
    dirs = os.listdir(path)

    for dir in dirs:
        subdirs_folder = os.path.join(path, dir)

        items = os.listdir(subdirs_folder)
        for i in items:
            if not '.' in i:
                links.append((os.path.join(path, dir, i + '.jpg'),
                              os.path.join(path, dir, i + '.json')))

    return links


# @jit
def crop_largest_square_around_point(width, height, box, IMG_SIZE):
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

    # returns coord after crop, coords after crop and resize
    return (left, top, right, bottom), (nleft, ntop, nright, nbottom)


# @jit
def prepare_training_sample(img, steps, labels, normalize, x_lis, y_lis, mask_lis, label_lis, time_lis, x1, y1, x2, y2):
    '''
    Takes one image, and produces steps amount of training samples
    '''
    mask = np.zeros_like(img)
    mask[y1:y2 + 1, x1:x2 + 1, :] = 1
    noise = np.random.randint(0, 256, size=img.shape)
    inter = np.linspace(img, noise, steps + 1)

    for i in range(steps):
        clone_x = np.copy(img)
        clone_y = np.copy(img)

        clone_x[y1:y2, x1:x2, :] = inter[i + 1, y1:y2, x1:x2, :]
        clone_y[y1:y2, x1:x2, :] = inter[i, y1:y2, x1:x2, :]

        x_lis.append(normalize(torch.from_numpy(clone_x).float()))
        y_lis.append(normalize(torch.from_numpy(clone_y).float()))
        mask_lis.append(torch.from_numpy(mask[:, :, 0:1]).float())
        label_lis.append(torch.from_numpy(labels).float())
        time_lis.append(torch.tensor([i]).float())

    # from utils import show_img
    # show_img(x_lis[0], permute=False)
    # show_img(x_lis[1], permute=False)
    # show_img(x_lis[2], permute=False)


# @jit
def one_hot_labels(label_list, selected):
    return np.array([1 if selected == elt else 0 for elt in label_list])


# @jit
def load_images_and_labels(links, labels, resolution):
    # ls = [(os.path.join('assets', 'adesample', 'a.jpg'), os.path.join('assets', 'adesample', 'a.json'))]
    resize = transforms.Resize((resolution, resolution))

    img_array = []
    encoded_labels = []
    cropped_and_resized_box_coords = []

    for imgfile, metafile in links:
        img = Image.open(imgfile)
        objects = json.load(open(metafile))["annotation"]["object"]

        for object in objects:
            if object["name"] in labels:
                np_labels = one_hot_labels(labels, object["name"])
                box = find_box_from_polygon(object["polygon"])
                coords, new_coords = crop_largest_square_around_point(*img.size, box, resolution)

                img_array.append(np.array(resize(img.crop(coords))))
                encoded_labels.append(np_labels)
                cropped_and_resized_box_coords.append(new_coords)

    return img_array, cropped_and_resized_box_coords, encoded_labels


def find_box_from_polygon(polygon):
    return min(polygon['x']), min(polygon['y']), max(polygon['x']), max(polygon['y'])

# def load_images_and_labels(links, labels, resolution):
#     links = [
#         os.path.join('assets', 'pics', '1.PNG'),
#         os.path.join('assets', 'pics', '2.PNG'),
#         os.path.join('assets', 'pics', '3.PNG'),
#         os.path.join('assets', 'pics', '4.PNG'),
#         os.path.join('assets', 'pics', '5.PNG'),
#     ]
#     boxes = [
#         (540, 360, 540+175, 360+175),
#         (0, 390, 0+320, 390+320),
#         (495, 415, 495+230, 415+230),
#         (150, 310, 150+300, 310+300),
#         (355, 1100, 355 + 200, 1100 + 200),
#     ]
#     images = [Image.open(link) for link in links]
#     labels = [one_hot_labels(labels, np.random.choice(labels)) for _ in range(len(boxes))]
#
#     #####################
#
#     resize = transforms.Resize((resolution, resolution))
#
#     img_array = []
#     cropped_box_coords = []
#     cropped_and_resized_box_coords = []
#
#     for i, b in zip(images, boxes):
#         coords, new_coords = crop_largest_square_around_point(*i.size, b, resolution)
#
#         i.crop(coords).show()
#         i.crop(b).show()
#         test_img = resize(i.crop(coords))
#         test_img.show()
#         test_img.crop(new_coords).show()
#
#         img_array.append(np.array(resize(i.crop(coords))))
#         cropped_and_resized_box_coords.append(new_coords)
#
#     return img_array, cropped_and_resized_box_coords, labels

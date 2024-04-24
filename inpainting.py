
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import os
from random import randint, seed
import itertools
from PIL import Image
import neuralgym as ng
from inpaint_ops import random_bbox, bbox2mask

class MaskGenerator():
    def __init__(self, height, width, channels=3, rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for inpainting training

        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width

        Keyword Arguments:
            channels {int} -- Channels to output (default: {3})
            rand_seed {[type]} -- Random seed (default: {None})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath
        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if any(
                filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(
                len(self.mask_files), self.filepath))
        # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self):
        """Generates a random irregular mask with lines, circles and elipses"""
        img = np.zeros((self.height, self.width, self.channels), np.uint8)
        # Set size scale
        size = int((self.width + self.height) * 0.03)
        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")
        # Draw random lines
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)
        # Draw random circles
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), thickness=1)
        # Draw random ellipses
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3,
                        (1, 1, 1), thickness=thickness)  # Change here

        return img

    def _load_mask(self, rotation=True, dilation=True, cropping=True):
        """Loads a mask from disk, and optionally augments it"""
        # Read image
        mask = cv2.imread(os.path.join(self.filepath, np.random.choice(
            self.mask_files, 1, replace=False)[0]))
        # Random rotation
        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D(
                (mask.shape[1]/2, mask.shape[0]/2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        # Random dilation
        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        # Random cropping
        if cropping:
            x = np.random.randint(0, mask.shape[1] - self.width)
            y = np.random.randint(0, mask.shape[0] - self.height)
            mask = mask[y:y+self.height, x:x+self.width]
        return (mask > 1).astype(np.uint8)

    def sample(self, random_seed=None):
        """Retrieve a random mask"""
        if random_seed:
            seed(random_seed)
        if self.filepath and len(self.mask_files) > 0:
            return self._load_mask()
        else:
            return self._generate_mask()


config = ng.Config('inpaint.yml')


# if os.path.exists('./mask'):
#     os.system('rm -rf mask')
# os.makedirs('./mask/train')
# os.makedirs('./mask/val')

# mask_generator = MaskGenerator(256, 256, 1, rand_seed=42)
# for i in range(45000):
#     m = mask_generator.sample()
#     cv2.imwrite(f'./mask/train/{i}.jpg', m*255)
# for i in range(5000):
#     # m = mask_generator.sample()
#     # cv2.imwrite(f'./mask/val/{i}.jpg', m*255)

#     bbox = random_bbox(config)
#     mask = bbox2mask(bbox, config)
#     mask = tf.Session().run(mask)[0]  # Convert tensor to numpy array
#     cv2.imwrite(f'./mask/val/{i}.jpg', mask*255)


# Now, combine the masks with the images to create the training and validation datasets.
# image will be in the same order as the masks, but instead mask/, it is imagenet/
# Read through all the directory
training_directory = "imagenet/train"
validation_directory = "imagenet/val"
training_images = [f for f in os.listdir(training_directory)]
validation_images = [f for f in os.listdir(validation_directory)]

# Combine the masks with the images
# if exist, delete the data directory
if os.path.exists("data"):
    os.system("rm -rf data")
os.makedirs("data/train")
os.makedirs("data/val")
os.makedirs("data/train_mask")
os.makedirs("data/val_mask")
os.makedirs("data/train_image_grouth_truth")
os.makedirs("data/val_image_grouth_truth")

"""
for i in range(45000):
    image = cv2.imread(f"{training_directory}/{training_images[i]}")
    image = cv2.resize(image, (256, 256))
    mask = cv2.imread(f"mask/train/{i}.jpg")
    # Remake the mask to the same size as the image variable
    mask = cv2.resize(mask, (256, 256))
    # masked_image = cv2.bitwise_and(image, mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f"data/train/{i}.jpg", masked_image)
    cv2.imwrite(f"data/train_mask/{i}.jpg", mask)
    cv2.imwrite(f"data/train_image_grouth_truth/{i}.jpg", image)
"""
mask_percentage_list = []
for i in range(5000):
    image = cv2.imread(f"{validation_directory}/{validation_images[i]}")
    image = cv2.resize(image, (256, 256))
    mask = cv2.imread(f'./mask/val/{i}.jpg', cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))
    # masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_image = image.copy()
    for x, y in itertools.product(range(256), range(256)):
        if mask[x, y] > 200:
            masked_image[x, y] = mask[x, y]
    cv2.imwrite(f"data/val/{i}.jpg", masked_image)
    cv2.imwrite(f"data/val_mask/{i}.jpg", mask)
    cv2.imwrite(f"data/val_image_grouth_truth/{i}.jpg", image)
    
    mask_percentage = np.sum(mask > 0) / (256 * 256)
    mask_percentage_list.append(mask_percentage)

np.save("mask_percentage_list.npy", mask_percentage_list)

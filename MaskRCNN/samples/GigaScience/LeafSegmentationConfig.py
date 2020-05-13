"""
Mask R-CNN
Train on the Arabidopsis dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modified by Gytis Bernotas
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib
import matplotlib.pyplot as plt
import glob
import imgaug

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################
class ArabidopsisLeafSegmentationConfig(Config):
    """Configuration for training on the Arabidopsis  dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "leaf"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    TRAIN_ROIS_PER_IMAGE = 128
    DETECTION_MAX_INSTANCES = 50

    STEPS_PER_EPOCH = 135 
    VALIDATION_STEPS = 22 
    
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    DETECTION_MIN_CONFIDENCE = 0.5
    USE_MINI_MASK = False
    BACKBONE = "resnet50"
    
    LEARNING_RATE = 0.01
    LEARNING_MOMENTUM = 0.9

############################################################
#  Dataset
############################################################
class ArabidopsisLeafSegmentationDataset(utils.Dataset):

    def __init__(self, images):
        self.images = images
        super().__init__()

    def load_plant(self, dataset_dir, subset):
        self.add_class("leaf", 1, "leaf")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        self.images = glob.glob(dataset_dir + "/*grayscale.png")
        
        for idx, img in enumerate(self.images):
            self.add_image("leaf", image_id=idx, path=img) 
               
    def load_image(self, image_id):
        img = cv2.imread(self.images[image_id])
        height, width = img.shape[:2]
        
        info = self.image_info[image_id]
        return img
        
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        
        pth = self.images[image_id].split("_")
        pth.pop()
        s = "_"
        mask = cv2.imread(s.join(pth) + "_label.png", 0)
        
        width = np.size(mask, 1)
        height = np.size(mask, 0) 
        
        masks = []
        for i in range(1,255):
            binary_mask = np.zeros([np.size(mask,0), np.size(mask,1)], dtype=np.uint8)
            binary_mask[mask == i] = 1
            if(np.sum(binary_mask) != 0):
                masks.append(binary_mask)
        masks = np.stack(masks, axis=-1)

        return masks, np.ones([masks.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "leaf":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
def train(model):
    """Train the model."""
    images = []
    dataset_train = ArabidopsisLeafSegmentationDataset(images)
    dataset_train.load_plant(args.dataset, "train")
    dataset_train.prepare()

    dataset_val = ArabidopsisLeafSegmentationDataset(images)
    dataset_val.load_plant(args.dataset, "val")
    dataset_val.prepare()
    
    print("Training network: heads") # example
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads', augmentation = imgaug.augmenters.OneOf([
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.Flipud(0.5),
                    imgaug.augmenters.Affine(rotate=(-90, 90)),
                    imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                    imgaug.augmenters.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                    imgaug.augmenters.Affine(shear=(-16, 16))
                ]))

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Configurations
    if args.command == "train":
        config = ArabidopsisLeafSegmentationConfig()
        
    else:
        class InferenceConfig(ArabidopsisLeafSegmentationConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        if(args.mode == "tune"):
            model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
            pass
        elif(args.mode == "scratch"):
            pass    
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

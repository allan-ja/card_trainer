import os
import math
import random
from collections import Counter
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import pickle
import tensorflow as tf
from pillow import Image
from sklearn.model_selection import train_test_split

from .config import Config
from . import model as modellib, utils

CSV_FILEPATH = 'data/full_dataset.csv'
LABELS_PATH = 'data/labels.txt'
COCO_WEIGHTS_PATH = os.environ.get('COCO_WEIGHTS_PATH', 
                    'data/mask_rcnn_coco.h5')
DEFAULT_LOGS_DIR = 'logs/'
IMAGE_HEIGHT = IMAGE_WIDTH = 1024


############################################################
#  Configurations
############################################################

class ChampopConfig(Config):
    '''Configuration for training on Playing Card Dataset'''
    NAME = 'champop'

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1
    # GPU_COUNT = 3
    IMAGES_PER_GPU = 2
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    IMAGE_SHAPE = np.array([IMAGE_HEIGHT, IMAGE_WIDTH, 3])


############################################################
#  Dataset
############################################################

class ChampopDataset(utils.Dataset):
    """Generates the image with playing cards synthetic dataset. The dataset consists of a background
    from textured images and two card images placed randomly on it.
    The images are generated on the fly. No file access required.
    """

    def load_scenes(self, dataset_info, labels, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        for i, card in enumerate(labels):
            self.add_class("cards", i+1, card)
        
        for index, row in dataset_info.iterrows():
            self.add_image("cards", image_id=index, path=row['im_filename'],
                           width=width, height=height, mask_path=row['mask_filename'],
                           class_ids=['card', 'card'])



    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]

        image = np.array(Image.open('images/' + info['path']))
        return image


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cards":
            return info["cards"]
        else:
            super(self.__class__).image_reference(self, image_id)


    def load_mask(self, image_id):
        """Load mask image and class_ids
        """
        info = self.image_info[image_id]
        mask = np.array(Image.open('masks_images/' + info['mask_path']))
        class_ids = np.array([self.class_names.index(c) for c in info['class_ids']])
        return mask[:, :, :2].astype(np.bool), class_ids.astype(np.int32)


def train(args):
    """Train the model."""

    # Configurations
    config = ChampopConfig()
    config.display()

    labels = ['card']

    full_dataset = pd.read_csv(CSV_FILEPATH)
    train_df, test_df = train_test_split(full_dataset, test_size=0.15)

    # Training dataset.

    # dataset_train = ChampopDataset()
    # dataset_train.load_scenes(10, card_pickle_path=card_pickle_path, 
    #                 backgrounds_folder=background_folder,
    #                 labels=labels, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    # dataset_train.prepare()

    # # Validation dataset
    # dataset_val = ChampopDataset()
    # dataset_val.load_scenes(2, card_pickle_path=CARDS_PICKLE, 
    #                 backgrounds_folder=BACKGROUNDS_FOLDER,
    #                 labels=labels, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    # dataset_val.prepare()


    # # Create model
    # model = modellib.MaskRCNN(mode="training", config=config,
    #                             model_dir=args.job_dir)

    # weights_path = os.path.join(args.job_dir, COCO_WEIGHTS_PATH)

    # # Load weights
    # print("Loading weights ", weights_path)

    # # Exclude the last layers because they require a matching
    # # number of classes
    # model.load_weights(weights_path, by_name=True, exclude=[
    #         "mrcnn_class_logits", "mrcnn_bbox_fc",
    #         "mrcnn_bbox", "mrcnn_mask"])



    # # *** This training schedule is an example. Update to your needs ***
    # # Since we're using a very small dataset, and starting from
    # # COCO trained weights, we don't need to train too long. Also,
    # # no need to train all layers, just the heads should do it.
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=30,
    #             layers='heads')

    # # Export model
    # export_path = os.path.join(args.job_dir, 'keras_export')
    # # tf.contrib.saved_model.save_keras_model(model.keras_model, export_path)


def get_args():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--job-dir', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()
    return args


############################################################
#  Training / Evaluate
############################################################

if __name__ == '__main__':

    args = get_args()
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.job_dir)

    train(args)


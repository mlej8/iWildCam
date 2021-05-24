import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import os
import json
import datetime
import copy
import logging

from config import *

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
        )
logger = logging.getLogger(__name__)

class iWildCam(Dataset):
	def __init__(self, dataset_file: str, img_dir: str, test=False,transform=None):
		"""
		Create dataset for iWildCam.

		:param dataset_file: location of the iWildCam dataset file
		:param img_dir: directory containing images
		:param transform: (optional) transform to be applied on an image sample.
		"""
		# directory where images are stored
		self.img_dir = img_dir

        logger.info(f'Loading {"test" if test else "train"} iWildCam dataset')
		time_t = datetime.datetime.utcnow()
		with open(dataset_file, 'r') as f:
            self.dataset = json.load(f)
		logger.info("Done in {}".format(datetime.datetime.utcnow() - time_t))
		logger.info(f"The dataset contains the following keys: {self.dataset.keys()}")

		# dictionary mapping image id to the image object - image object is a dict with {seq_num_frames, location, datetime, id, seq_id, width, height, file_name, seq_frame_num} as keys
		self.images = {image['id']: image for image in self.dataset['images']}
		self.image_ids = self.images.keys()
		
		# annotations and labels are only available for train dataset
		if not test: 
			# dictionary mapping image_id to an annotation that contains {id, image_id, category_id} as keys
			self.annotations = {annotation['image_id']: annotation for annotation in self.dataset['annotations']}

			# dictionary mapping category id (int) to category name (string)
			self.categories = {category['id']: category["name"] for category in self.dataset['categories']}

		# loading the pre-computed bounding boxes
		with open(MEGATRON_BOUNDING_BOXES, encoding='utf-8') as json_file:
    		self.megadetector_results = json.load(json_file)
		
		# building a dictionary mapping image id to detection
		self.bbox = {detection["id"]: detection["detections"] for detection in self.megadetector_results}

		if transform:
			logger.info("Transform: %s", transform)
		logger.info(f"There are {len(self.images)} samples in the dataset.")
		logger.info(f"Images from {self.img_dir} are used for the dataset.")

	def bb_to_point(self, bb, img_width, img_height):
		""" Helper method transforming list of bounding boxes for an image to points. Since the bounding boxes from MegaDetector are scaled between [0,1], we use the width and height of the original image to scale back. """
		return [(round(((2*box[0] + box[2])/2) * img_width),round(((2*box[1] + box[3])/2) * img_height)) for box in bb]

	def __len__(self):
		""" Return length of the dataset based on the number of images. """
		return len(self.image_ids)

	def __getitem__(self, idx):
		""" Each sample consist of (question, image, answer). """
		img_id = self.image_ids[idx]

		# image annotation
		image_ann = self.images[img_id]
		
		# get the image from disk
		img_path = os.path.join(self.img_dir, image_ann["file_name"])

		# read image from disk
		if os.path.isfile(img_path):
			image = self.preprocess_image(img_path)
		else:
			logger.error(f"{img_path} is not a valid file.")
			exit(1)

		# transform bounding boxes into points
		points = self.bb_to_point(self.bbox[img_id], image_ann["width"], image_ann["height"])

		return {"image":image, "points": points}

	def preprocess_image(self, img_path):
		""" Helper method to preprocess an image """	
		# always opening images in rgb - so that greyscale images are copied through all 3 channels
		image = Image.open(img_path).convert("RGB")

		# apply transformation on the image
		if self.transform:
			image = self.transform(image)
		
		return image
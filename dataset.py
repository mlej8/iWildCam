import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from collections import defaultdict
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
	@classmethod
	def iWildCam_collate(cls, batch):
		""" Custom collate function for iWildCam """
		# filter out samples that don't have bounding boxes
		mini_batch = [sample for sample in batch if sample["target"]]		
		images = [sample["image"] for sample in mini_batch]		
		targets = [sample["target"] for sample in mini_batch]		
		return images, targets

	def __str__(self):
		return f"iWildCam Dataset"

	@property
	def num_classes(self):
		return len(self.categories)

	def __init__(self, dataset_file: str, img_dir: str, test=False, transforms=None):
		"""
		Create dataset for iWildCam.

		:param dataset_file: location of the iWildCam dataset file
		:param img_dir: directory containing images
		:param transforms: (optional) transforms to be applied on an image sample.
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
		self.image_ids = [*self.images.keys()]
		
		# dictionary mapping category id (int) to category name (string)
		with open(TRAIN_DATASET, "r") as f:
			self.categories = {category['id']: category["name"] for category in json.load(f)['categories']}
			self.cat_to_idx = {category: i for i,category in enumerate(self.categories.keys())}

		# annotations and labels are only available for train dataset
		if not test: 
			# dictionary mapping image_id to an annotation that contains {id, image_id, category_id} as keys
			self.annotations = {annotation['image_id']: annotation for annotation in self.dataset['annotations']}

		# loading the pre-computed bounding boxes
		with open(MEGATRON_BOUNDING_BOXES, encoding='utf-8') as json_file:
			self.megadetector_results = json.load(json_file)
		
		# building a dictionary mapping image id to detection
		self.bbox = defaultdict(lambda: [])
		for detection in self.megadetector_results["images"]:
			if detection["detections"]:
				self.bbox[detection["id"]] += detection["detections"]
			
		# path to instance masks
		self.masks_dir = INSTANCE_MASK_DIR

		# transforms
		self.transforms = transforms
		
		if self.transforms:
			logger.info("Transforms: %s", transforms)
		logger.info(f"There are {len(self.images)} samples in the dataset.")
		logger.info(f"Images from {self.img_dir} are used for the dataset.")

	def bb_to_point(self, bb, img_width, img_height):
		""" 
		Helper method transforming list of bounding boxes for an image to points. 
		Since the bounding boxes from MegaDetector are scaled between [0,1], we use the width and height of the original image to scale back. 
		"""
		return [(round(((2*box[0] + box[2])/2) * img_width),round(((2*box[1] + box[3])/2) * img_height)) for box in bb]

	def __len__(self):
		""" Return length of the dataset based on the number of images. """
		return len(self.image_ids)

	def __getitem__(self, idx):
		""" 
		image: a PIL Image of size (H, W)
		target: a dict containing the following fields
			boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
			labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
			image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
			area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
			iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
			(optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
			(optionally) keypoints (FloatTensor[N, K, 3]): For each one of the N objects, it contains the K keypoints in [x, y, visibility] format, defining the object. visibility=0 means that the keypoint is not visible. Note that for data augmentation, the notion of flipping a keypoint is dependent on the data representation, and you should probably adapt references/detection/transforms.py for your new keypoint representation
		"""
		# get image id
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

		# get target for training faster RCNN
		target = {}
		
		if self.bbox[img_id] and [detection["bbox"] for detection in self.bbox[img_id] if ((detection["bbox"][2] - detection["bbox"][0])* (detection["bbox"][3]-detection["bbox"][1])) > 0]:
			
			""" 
			Bounding boxes from MegaDetector are in normalized, floating-point coordinates, with the origin at the upper-left.
			Note that the categories returned by the detector are not the categories in the iWildCam dataset
			detection { 
				'bbox' : [x, y, width, height],
				'category': str,
				'conf': float
			}
			"""
			# get the bounding boxes (given by MegaDetector) for the image and use if label
			target["boxes"] = torch.tensor([detection["bbox"] for detection in self.bbox[img_id] if ((detection["bbox"][2] - detection["bbox"][0])* (detection["bbox"][3]-detection["bbox"][1])) > 0])

			# scale bounding boxes back to their original scale
			target["boxes"][:,[1,3]] *= image_ann["height"]
			target["boxes"][:,[0,2]] *= image_ann["width"] 

			# assuming detections all come from same class
			category = self.annotations[img_id]['category_id']
			target["labels"] = torch.full(size=(target["boxes"].size(0),), fill_value=self.cat_to_idx[category])

			# include all instances for evaluation
			target["iscrowd"] = torch.zeros(size=(target["boxes"].size(0),))

			# image id 
			target["image_id"] = torch.tensor([idx])

			# area covered by bounding boxes 
			target["area"] = (target["boxes"][:,2] - target["boxes"][:,0]) * (target["boxes"][:,3] - target["boxes"][:,1])

		return {"image": image, "target": target}

		# # transform bounding boxes into points (for LCFCN)
		# points = self.bb_to_point(self.bbox[img_id], image_ann["width"], image_ann["height"])

	def preprocess_image(self, img_path):
		""" Helper method to preprocess an image """	
		# always opening images in rgb - so that greyscale images are copied through all 3 channels
		image = Image.open(img_path).convert("RGB")

		# apply transformation on the image
		if self.transforms:
			image = self.transforms(image)
		
		return image

	def write_results_files(self, results, output_dir):
		"""Write the detections in the format for MOT17Det sumbission

		all_boxes[image] = N x 5 array of detections in (x1, y1, x2, y2, score)

		Each file contains these lines:
		<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
		"""

		files = {}
		for image_id, res in results.items():
			path = self._img_paths[image_id]
			img1, name = osp.split(path)
			# get image number out of name
			frame = int(name.split('.')[0])
			# smth like /train/MOT17-09-FRCNN or /train/MOT17-09
			tmp = osp.dirname(img1)
			# get the folder name of the sequence and split it
			tmp = osp.basename(tmp).split('-')
			# Now get the output name of the file
			out = tmp[0]+'-'+tmp[1]+'.txt'
			outfile = osp.join(output_dir, out)

			# check if out in keys and create empty list if not
			if outfile not in files.keys():
				files[outfile] = []

			for box, score in zip(res['boxes'], res['scores']):
				x1 = box[0].item()
				y1 = box[1].item()
				x2 = box[2].item()
				y2 = box[3].item()
				files[outfile].append(
					[frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

		for k, v in files.items():
			with open(k, "w") as of:
				writer = csv.writer(of, delimiter=',')
				for d in v:
					writer.writerow(d)

	def print_eval(self, results, ovthresh=0.5):
		"""Evaluates the detections (not official!!)

		all_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)
		"""

		if 'test' in self.root:
			print('No GT data available for evaluation.')
			return

		# Lists for tp and fp in the format tp[cls][image]
		tp = [[] for _ in range(len(self._img_paths))]
		fp = [[] for _ in range(len(self._img_paths))]

		npos = 0
		gt = []
		gt_found = []

		for idx in range(len(self._img_paths)):
			annotation = self._get_annotation(idx)
			bbox = annotation['boxes'][annotation['visibilities'].gt(self._vis_threshold)]
			found = np.zeros(bbox.shape[0])
			gt.append(bbox.cpu().numpy())
			gt_found.append(found)

			npos += found.shape[0]

		# Loop through all images
		for im_index, (im_gt, found) in enumerate(zip(gt, gt_found)):
			
			# Loop through dets an mark TPs and FPs
			im_det = results[im_index]['boxes'].cpu().numpy()

			im_tp = np.zeros(len(im_det))
			im_fp = np.zeros(len(im_det))
			for i, d in enumerate(im_det):
				ovmax = -np.inf

				if im_gt.size > 0:
					# compute overlaps
					# intersection
					ixmin = np.maximum(im_gt[:, 0], d[0])
					iymin = np.maximum(im_gt[:, 1], d[1])
					ixmax = np.minimum(im_gt[:, 2], d[2])
					iymax = np.minimum(im_gt[:, 3], d[3])
					iw = np.maximum(ixmax - ixmin + 1., 0.)
					ih = np.maximum(iymax - iymin + 1., 0.)
					inters = iw * ih

					# union
					uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
							(im_gt[:, 2] - im_gt[:, 0] + 1.) *
							(im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)

					overlaps = inters / uni
					ovmax = np.max(overlaps)
					jmax = np.argmax(overlaps)

				if ovmax > ovthresh:
					if found[jmax] == 0:
						im_tp[i] = 1.
						found[jmax] = 1.
					else:
						im_fp[i] = 1.
				else:
					im_fp[i] = 1.

			tp[im_index] = im_tp
			fp[im_index] = im_fp

		# Flatten out tp and fp into a numpy array
		i = 0
		for im in tp:
			if type(im) != type([]):
				i += im.shape[0]

		tp_flat = np.zeros(i)
		fp_flat = np.zeros(i)

		i = 0
		for tp_im, fp_im in zip(tp, fp):
			if type(tp_im) != type([]):
				s = tp_im.shape[0]
				tp_flat[i:s+i] = tp_im
				fp_flat[i:s+i] = fp_im
				i += s

		tp = np.cumsum(tp_flat)
		fp = np.cumsum(fp_flat)
		rec = tp / float(npos)
		# avoid divide by zero in case the first detection matches a difficult
		# ground truth (probably not needed in my code but doesn't harm if left)
		prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
		tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)

		# correct AP calculation
		# first append sentinel values at the end
		mrec = np.concatenate(([0.], rec, [1.]))
		mpre = np.concatenate(([0.], prec, [0.]))

		# compute the precision envelope
		for i in range(mpre.size - 1, 0, -1):
			mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

		# to calculate area under PR curve, look for points
		# where X axis (recall) changes value
		i = np.where(mrec[1:] != mrec[:-1])[0]

		# and sum (\Delta recall) * prec
		ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

		tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap

		print(f"AP: {ap:.2f} Prec: {prec:.2f} Rec: {rec:.2f} TP: {tp} FP: {fp}")

		return {'AP': ap, 'precision': prec, 'recall': rec, 'TP': tp, 'FP': fp}

# def get_transforms():
#     transforms = []
#     # converts the image, a PIL image, into a PyTorch Tensor
#     transforms.append(T.ToTensor())
#     if not test:
#         # during training, randomly flip the training images
#         # and ground-truth for data augmentation
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)

preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ])

# use our dataset and defined transformations
train_dataset = iWildCam(dataset_file=TRAIN_DATASET, img_dir=TRAIN_DATA_DIR, test=False, transforms=preprocess)
test_dataset = iWildCam(dataset_file=TEST_DATASET, img_dir=TEST_DATA_DIR, test=True, transforms=preprocess)
    
# define training and validation data loaders
train_loader = torch.utils.data.DataLoader(
    							train_dataset,
								batch_size=BATCH_SIZE,
								shuffle=SHUFFLE,
								num_workers=NUM_WORKERS,
								collate_fn=iWildCam.iWildCam_collate
								)
test_loader = torch.utils.data.DataLoader(
								test_dataset, 
								batch_size=BATCH_SIZE, 
								shuffle=False,
								num_workers=NUM_WORKERS,
								collate_fn=iWildCam.iWildCam_collate
								)

# TODO: transforms
# TODO: add instance masks if it is helpful ? 

import os

# setting the seed for reproducability (it is important to set seed when using DPP mode)
import torch
torch.manual_seed(0)

SCRATCH_DIR = "/scratch/mlej8/"
DATA_DIR = os.path.join(SCRATCH_DIR, "iWildCam2021")

# path to datasets
METADATA = os.path.join(DATA_DIR, "metadata")
MEGATRON_BOUNDING_BOXES =os.path.join(METADATA, "iwildcam2021_megadetector_results.json")
TRAIN_DATASET = os.path.join(METADATA, "iwildcam2021_train_annotations.json")
TEST_DATASET = os.path.join(METADATA, "iwildcam2021_test_information.json")

# path to images
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

# instance masks
INSTANCE_MASK_DIR = os.path.join(DATA_DIR, "instance_masks", "instance_masks")

SUBMISSION_EXAMPLE_PATH = os.path.join(DATA_DIR,"sample_submission.csv")

# dataloader settings
BATCH_SIZE = 2
SHUFFLE = False
NUM_WORKERS = 4

# training settings
NUM_EPOCHS = 30
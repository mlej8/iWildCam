import os
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

SUBMISSION_EXAMPLE_PATH = os.path.join(DATA_DIR,"sample_submission.csv")
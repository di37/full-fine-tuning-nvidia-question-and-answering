import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from glob import glob
import torch 

from dotenv import load_dotenv

_ = load_dotenv()

# RTX-4000 doesnt support P2P and IB. Therefore, the settings
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES")
os.environ["NCCL_P2P_DISABLE"] = os.getenv("NCCL_P2P_DISABLE")
os.environ["NCCL_IB_DISABLE"] = os.getenv("NCCL_IB_DISABLE")

KAGGLE_CONFIG_DIR = os.getenv("KAGGLE_CONFIG_DIR")
DATASET_ID_OR_URL = os.getenv("DATASET_ID_OR_URL")

# Dataset paths
DATA_DIR = os.getenv("DATA_DIR")
DATA_PATH = glob(os.path.join(DATA_DIR, os.path.split(DATASET_ID_OR_URL)[-1], "*.csv"))[0]
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model directory
MODEL_DIR = os.getenv("MODEL_DIR")

# Base Model
MODEL_ID = os.getenv("MODEL_ID")
MODEL_NAME = os.getenv("MODEL_NAME")
BASE_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

## Output Model
TRAINED_MODEL_NAME = os.getenv("TRAINED_MODEL_NAME")
TRAINING_PATH = os.path.join(MODEL_DIR, TRAINED_MODEL_NAME)
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model")

## Sentence Embeddings Model
SENTENCE_EMBEDDING_MODEL = os.getenv("SENTENCE_EMBEDDING_MODEL")

## Hyperparameters
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 4
SAVE_TOTAL_LIMIT = 2
EVALUATION_STRATEGY = "epoch"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
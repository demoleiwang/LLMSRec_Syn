import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from utils import get_model
from trainer import SelectedUserTrainer
from openai_parallel_toolkit import ParallelToolkit, Prompt
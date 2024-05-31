import os
import numpy as np
from tqdm import tqdm
import torch
from recbole.trainer import Trainer
from recbole.utils import EvaluatorType, set_color
from recbole.data.interaction import Interaction

from sklearn.metrics.pairwise import cosine_similarity
import random

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist

import json



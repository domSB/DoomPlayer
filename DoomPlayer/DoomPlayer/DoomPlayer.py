import tensorflow as tf
import numpy as np  
from vizdoom import *

import random                
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore') 

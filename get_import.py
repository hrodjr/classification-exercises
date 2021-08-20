import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import get_db
import warnings
warnings.filterwarnings("ignore")

import graphviz
from graphviz import Graph

from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
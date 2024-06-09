import pandas
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import glob



# Define a path where EMG data is stored into folders belonging to each class
MAIN_PATH = ""
# Define a path to save generated features
FEATURES_PATH = ""


# List of each folder with a class
MOTOR_TASKS = ["Task1", "Task2"]

# Creates a dictionary with a key-value pair for each item in MOTOR_TASKS
MOTOR_TASKS_DIC = {label.split('.')[0]:num+1 for num, label in enumerate(MOTOR_TASKS)}
"""
MOTOR_TASKS_DIC = {}
for num, label in enumerate(MOTOR_TASKS):
    MOTOR_TASKS_DIC[label.split('.')[0]] = num+1 
# num: a counter
# label: the value in the ith iteration of the loop
"""

# List of features:
FEATURES_LIST = ["MeanAbsoluteValue", "NumSlopeChanges", "NumZeroCrossings"]

# Define EMG sensors used in experiment:
EMG_SENSORS = ["MW1"]


# Features
def mean_abs_value(data):
    mean_abs_val = abs((data)).mean()
    return mean_abs_val

def num_slope_changes(data):
    num_slope_change = 0
    return num_slope_change

def num_zero_crossings(data):
    num_zero_cross = 0
    return num_zero_cross
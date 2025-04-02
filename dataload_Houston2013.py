


# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math

from einops import rearrange
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from scipy import io
import torch.utils.data
import scipy.io as sio
import mat73
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from tensorflow import keras
from keras.utils import to_categorical

from utils.auxiliary import applyPCA
from utils.hyper_pytorch import HyperData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path
# path='../Datasets/'
# C:/Users/admin/Desktop/Transformer_code/HSITransformer/Datasets/

# 2.1 Loads Data
# Load hyperpsectral data
hsi_2013_data=sio.loadmat('D:/lyh/data/Datasets/2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat')['ans']
print('hsi_2013_data shape:', hsi_2013_data.shape)
# hsi_2013_data = applyPCA(hsi_2013_data, 20)
data_hsi = hsi_2013_data

# Loader Lidar  data
import mat73
lidar_2013_data = sio.loadmat('D:/lyh/data/Datasets/2013_IEEE_GRSS_DF_Contest_LiDAR.mat')['LiDAR_data']

print('Lidar_2013_data shape:', lidar_2013_data.shape)

#Load ground truth labels
gt_2013_data=sio.loadmat('D:/lyh/data/Datasets/GRSS2013.mat')['name']
print('gt_2013_data.shape:', gt_2013_data.shape)
print(np.max(gt_2013_data))
data_lidar = lidar_2013_data
# # 2.0 Data Preprocessing & Dataloader Preparation

# 2.1 Define the class information
# class_info = [(0, "Healthy grass", 'training_sample', 198, 'test_sample', 1053,  'total', 1251),
#     (1, "Stressed grass",'training_sample', 190, 'test_sample', 1064,  'total', 1254),
#     (2, "Synthetic grass", 'training_sample', 192, 'test_sample', 505,  'total', 697),
#     (3, "Trees", 'training_sample', 188, 'test_sample', 1058,  'total', 1244),
#     (4, "Soil",'training_sample', 186, 'test_sample', 1056,  'total', 1242),
#     (5, "Water", 'training_sample', 182, 'test_sample', 141,  'total', 325),
#     (6, "Residential", 'training_sample', 196, 'test_sample', 1072,  'total', 1268),
#     (7, "Commercial", 'training_sample', 191, 'test_sample', 1053,  'total', 1244),
#     (8, "Road", 'training_sample', 193, 'test_sample', 1059,  'total', 1252),
#     (9, "Highway", 'training_sample', 191, 'test_sample', 1036,  'total', 1227),
#     (10, "Railway", 'training_sample', 181, 'test_sample', 1054,  'total', 1235),
#     (11, "Parking lot 1", 'training_sample', 192, 'test_sample', 1041,  'total', 1233),
#     (12, "Parking lot 2", 'training_sample', 184, 'test_sample',285,  'total', 469),
#     (13, "Tennis court",'training_sample', 181, 'test_sample', 247,  'total', 428),
#     (14, "Running track", 'training_sample', 187, 'test_sample', 473,  'total', 660)]

class_info = [(1, "Healthy grass", 'training_sample', 198, 'test_sample', 1053,  'total', 1251),
    (2, "Stressed grass",'training_sample', 190, 'test_sample', 1064,  'total', 1254),
    (3, "Synthetic grass", 'training_sample', 192, 'test_sample', 505,  'total', 697),
    (4, "Trees", 'training_sample', 188, 'test_sample', 1058,  'total', 1244),
    (5, "Soil",'training_sample', 186, 'test_sample', 1056,  'total', 1242),
    (6, "Water", 'training_sample', 182, 'test_sample', 141,  'total', 325),
    (7, "Residential", 'training_sample', 196, 'test_sample', 1072,  'total', 1268),
    (8, "Commercial", 'training_sample', 191, 'test_sample', 1053,  'total', 1244),
    (9, "Road", 'training_sample', 193, 'test_sample', 1059,  'total', 1252),
    (10, "Highway", 'training_sample', 191, 'test_sample', 1036,  'total', 1227),
    (11, "Railway", 'training_sample', 181, 'test_sample', 1054,  'total', 1235),
    (12, "Parking lot 1", 'training_sample', 192, 'test_sample', 1041,  'total', 1233),
    (13, "Parking lot 2", 'training_sample', 184, 'test_sample',285,  'total', 469),
    (14, "Tennis court",'training_sample', 181, 'test_sample', 247,  'total', 428),
    (15, "Running track", 'training_sample', 187, 'test_sample', 473,  'total', 660)]

# [41,47,125],[]

# Create a dictionary to store class number, class name, and class samples
class_dict = {class_number: {"class_name": class_name,
                             'training_sample': training_sample,
                             'test_sample': test_sample,
                             "total_samples": total}
              for class_number, class_name, _, training_sample, _, test_sample, _, total in class_info}

print(class_dict)


# ### 2.1  Samples Extraction
# 2.2 Samples Extraction
# Define patch size and stride
patch_size = 11
stride = 1

# Create an empty list to store patches and labels
hsi_samples = []
lidar_samples = []
labels = []

# Initialize a dictionary to store class count
class_count = {i: 0 for i in class_dict.keys()}

# Function to check if all classes have the required number of samples
def all_classes_completed(class_count, class_dict):
    return all(class_count[class_num] == class_dict[class_num]["total_samples"] for class_num in class_dict.keys())

while not all_classes_completed(class_count, class_dict):
    # Loop through the ground truth data
    for label in class_dict.keys():
        # Get the coordinates of the ground truth pixels
        #coords = np.argwhere((gt_2013_data == label) & (mask > 0))
        coords = np.argwhere(gt_2013_data == label)
        # coords = np.argwhere(gt_2013_data)

        # Shuffle the coordinates to randomize the patch extraction
        np.random.shuffle(coords)
        for coord in coords:
            i, j = coord
            # Calculate the patch indices
            i_start, i_end = i - patch_size // 2, i + patch_size // 2 + 1
            j_start, j_end = j - patch_size // 2, j + patch_size // 2 + 1

            # Check if the indices are within the bounds of the HSI data
            if i_start >= 0 and i_end <= hsi_2013_data.shape[0] and j_start >= 0 and j_end <= hsi_2013_data.shape[1]:
                # Extract the patch
                hsi_patch = hsi_2013_data[i_start:i_end, j_start:j_end, :]

                # Extract the LiDAR patch
                lidar_patch = lidar_2013_data[i_start:i_end, j_start:j_end, :]

                # If the class count is less than the required samples
                if class_count[label] < class_dict[label]["total_samples"]:
                    # Append the patch and its label to the list
                    hsi_samples.append(hsi_patch)
                    lidar_samples.append(lidar_patch)
                    labels.append(label)
                    class_count[label] += 1

                    # If all classes have the required number of samples, exit the loop
                    if all_classes_completed(class_count, class_dict):
                        break

# Convert the list of patches and labels into arrays
hsi_samples = np.array(hsi_samples)
lidar_samples = np.array(lidar_samples)
labels = np.array(labels) # GT
print('hsi_samples shape:', hsi_samples.shape)
print('lidar_samples shape:', lidar_samples.shape)
print('labels shape:', labels.shape)


# ### 2.2 Training samples extraction

#Avoid overlap of train and test
# Extracting training samples
hsi_training_samples, lidar_training_samples, training_labels = [], [], []
used_indices = []  # To keep track of indices already taken for training samples

for label, class_data in class_dict.items():
    # Get indices of the current class
    class_indices = np.where(labels == label)[0]

    # Randomly shuffle the indices
    np.random.shuffle(class_indices)

    # Take the required number of training samples
    train_indices = class_indices[:class_data["training_sample"]]
    used_indices.extend(train_indices)  # Add these to the used_indices list

    # Append training samples
    hsi_training_samples.extend(hsi_samples[train_indices])
    lidar_training_samples.extend(lidar_samples[train_indices])
    training_labels.extend(labels[train_indices])

# Extracting test samples
hsi_test_samples, lidar_test_samples, test_labels = [], [], []

for label, class_data in class_dict.items():
    class_indices = np.where(labels == label)[0]

    # Exclude indices which were used for training
    test_indices = np.setdiff1d(class_indices, used_indices)

    # Append test samples
    hsi_test_samples.extend(hsi_samples[test_indices])
    lidar_test_samples.extend(lidar_samples[test_indices])
    test_labels.extend(labels[test_indices])

# Convert lists back to numpy arrays
hsi_training_samples = np.array(hsi_training_samples)
lidar_training_samples = np.array(lidar_training_samples)
training_labels = np.array(training_labels)

hsi_test_samples = np.array(hsi_test_samples)
lidar_test_samples = np.array(lidar_test_samples)
test_labels = np.array(test_labels)

# Print shapes to verify
print('hsi_training_samples shape:', hsi_training_samples.shape)
print('lidar_training_samples shape:', lidar_training_samples.shape)
print('training_labels shape:', training_labels.shape)

print('hsi_test_samples shape:', hsi_test_samples.shape)
print('lidar_test_samples shape:', lidar_test_samples.shape)
print('test_labels shape:', test_labels.shape)

# hsi_train=np.transpose(hsi_training_samples, (0, 3, 1, 2))
hsi_train=np.transpose(hsi_training_samples, (0, 3, 1, 2)).astype("float32")
lidar_train=np.transpose(lidar_training_samples, (0, 3, 1, 2)).astype("float32")
y_train=training_labels
print('hsi_train_samples shape:', hsi_train.shape)
print('lidar_train_samples shape:', lidar_train.shape)
print('train_labels shape:', y_train.shape)
hsi_test=np.transpose(hsi_test_samples, (0, 3, 1, 2)).astype("float32")
lidar_test=np.transpose(lidar_test_samples, (0, 3, 1, 2)).astype("float32")
y_test=test_labels
print('hsi_test_samples shape:', hsi_test.shape)
print('lidar_test_samples shape:', lidar_test.shape)
print('y_test shape:', y_test.shape)

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(hsi_train).type(torch.FloatTensor),
                                               torch.from_numpy(lidar_train).type(torch.FloatTensor),
                                               torch.from_numpy(y_train).type(torch.LongTensor))

test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(hsi_test).type(torch.FloatTensor),
                                               torch.from_numpy(lidar_test).type(torch.FloatTensor),
                                               torch.from_numpy(y_test).type(torch.LongTensor))


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,
                                               shuffle=True,
                                               num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=32,
                                               shuffle=True,
                                               num_workers=0)
print("data is ok")


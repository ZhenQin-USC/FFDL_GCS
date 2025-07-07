import numpy as np
import sys
import os
import json
import torch
sys.path.append('/project/jafarpou_227/Storage_Folder/Zhen/Research/Code/PhysicsGuided/FFDL/src')
from sys import argv
from os.path import join
from source import RSE_loss, memory_usage_psutil, MyDataset, train, data_loader_experiment_1
from source import ffdl3M as Model
from torch.utils.data import DataLoader

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Inputs
argv = ["", "model1", "2"]
model_folder, batch_size = argv[1], argv[2]
root_to_data = '/project/jafarpou_227/Storage_Folder/Zhen/Data/CO2_Dataset/grid120'
root_to_model = '/scratch1/zhenq/GCS_FFDL/checkpoint_perno9/exp2_both'
path_to_model = join(root_to_model, model_folder)
print(model_folder)

# Hyperparameter
nstep = 15
learning_rate, step_size, gamma, gradient_clip_val = 1e-4, 4500, 0.9, 40
batch_size, EPOCHS, loss_fn = int(batch_size), 100, RSE_loss()

# Build Model
print('Build Model')
model = Model().to(device) 
print('Summarize Model')
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable Parameters: {:.4f} M".format(trainable_params/1e6))
mem = memory_usage_psutil()

# Define Dataset
trainingset_folders = [
    'onewell_g20_z2', 'onewell_g60_z2', 'onewell_g100_z2', 'onewell_g20_z5', 'onewell_g60_z5', 'onewell_g100_z5',
    'twowell_tworange_g20_z2', 'twowell_tworange_g60_z2', 'twowell_tworange_g100_z2', 'twowell_tworange_g20_z5', 'twowell_tworange_g60_z5', 'twowell_tworange_g100_z5',]
validateset_folders = ['twowell_tworange_g20_z5']
testingset_folders = ['twowell_tworange_g60_z5', 'twowell_tworange_g100_z5']
for folder in validateset_folders: trainingset_folders.remove(folder)
for folder in testingset_folders: trainingset_folders.remove(folder)
print("Train: ", trainingset_folders)
print("Valid: ", validateset_folders)
print("Test:  ", testingset_folders)

# Collect settings
settings = {'nstep': nstep, 'batch_size': batch_size, 'EPOCHS': EPOCHS, 'step_size': step_size, 
            'gamma': gamma, 'learning_rate': learning_rate, 'gradient_clip_val': gradient_clip_val, 
            'trainingset_folders': trainingset_folders, 'validateset_folders': validateset_folders}
if not os.path.exists(path_to_model):
    os.makedirs(path_to_model)
with open(join(path_to_model,'settings.json'), 'w') as file:
    json.dump(settings, file)
print(settings)
mem = memory_usage_psutil()

# Load Dataset
def training_data_loader(): return data_loader_experiment_1(trainingset_folders, root_to_data=root_to_data)
def validate_data_loader(): return data_loader_experiment_1(validateset_folders, sample_index=range(5), root_to_data=root_to_data)
print('Training Dataset')
training_set = MyDataset(training_data_loader, nstep)       
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
print(training_set.contrl.shape, training_set.states.shape, training_set.static.shape)
print(training_set.time_index.shape, training_set.real_index.shape)
print('Validate Dataset')
validate_set = MyDataset(validate_data_loader, nstep)  
valid_loader = DataLoader(validate_set, batch_size=1, shuffle=False)
print(validate_set.contrl.shape, validate_set.states.shape, validate_set.static.shape)
print(validate_set.time_index.shape, validate_set.real_index.shape)
mem = memory_usage_psutil()

# Train Loop
print('Training Loop')
train_loss_list, valid_loss_list = train(
    model, device, EPOCHS, train_loader, valid_loader, path_to_model, verbose=1, 
    learning_rate=learning_rate, step_size=step_size, gamma=gamma, loss_fn=loss_fn, 
    gradient_clip=True, gradient_clip_val=gradient_clip_val)

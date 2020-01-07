
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import scipy.io as sio
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt
from tqdm import tqdm

patch_h, patch_w = 32, 32

# %% Prepare dataset

# List all files
pascal_path = './images/pascal/JPEGImages'
f_list = [f for f in listdir(pascal_path) if isfile(join(pascal_path, f)) and f.endswith('.jpg')]

# Save
save_path = './images/pascal'
if os.path.exists(save_path):
    os.makedirs(save_path)

all_patches = []
c = 0

for f in tqdm(f_list):
    im = cv2.imread(os.path.join(pascal_path, f))
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_h, im_w = im_gray.shape
    max_patches = (im_h * im_w) // (patch_h * patch_w)
    patches = extract_patches_2d(im_gray, (patch_h, patch_w), max_patches=max_patches)
    all_patches.append(patches)

concat_patches = np.concatenate(all_patches, axis=0)
print('Saving mat file...')
sio.savemat('./images/pascal_patches.mat', {'patches': concat_patches})
print('[!] First Mat file is ready.')

x_train = sio.loadmat('./images/pascal/pascal_patches.mat')['patches']

means = []
img_std = []
smooth_img_std = []

num_img_per_bin = 3000
max_std = 110
n_bins = max_std + 1
selected_img = []
std_hist = np.zeros(n_bins)

c = 0
for i in tqdm(range(x_train.shape[0])):
    cur_std = np.std(x_train[i])
    if cur_std > max_std:
        cur_std = max_std
    cur_bin_idx = int(np.floor(cur_std))
    
    if std_hist[cur_bin_idx] < num_img_per_bin:
        std_hist[cur_bin_idx] += 1
        selected_img.append(x_train[i])
        img_std.append(cur_std)
    
    
plt.plot(std_hist)
plt.figure()
plt.hist(img_std)

# Save new dataset
print('Saving mat file...')
train_imgs = np.array(selected_img)
sio.savemat('./images/pascal/pascal_resampled.mat', {'patches':train_imgs})

print('[!] Second Mat file is ready.')

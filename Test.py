# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:18:20 2018

@author: Test
"""

import os
import time
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io as sio
import cv2
import skimage.transform as imgTrans
from skimage.measure import compare_ssim, compare_psnr
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw  
from tqdm import tqdm
import tensorflow as tf
keras = tf.keras
layers = keras.layers
from include.my_circular_layer import Conv2D_circular
import include.various_Functions as vf
from scipy.ndimage.filters import convolve, median_filter
from scipy.ndimage.filters import gaussian_filter


# Utilities
def multiply_255(x):
    return x*255.0   

def divide_255(x):
    return x/255.0  

def multiply_scalar(x, scalar):
    return x * scalar #tf.convert_to_tensor(scalar, tf.float32)

def buildModel(model_path, patch_rows=32, patch_cols=32, channels=1, block_size=8, use_circular=True):
    
    conv2d_layer = layers.Conv2D if use_circular == False else Conv2D_circular
    
    w_rows = int((patch_rows) / block_size)
    w_cols = int((patch_cols) / block_size)
    
    input_img = layers.Input(shape=(patch_rows, patch_cols, 1), name='input_img')
    input_strenght_alpha = layers.Input(shape=(1,), name='strenght_factor_alpha')
    input_watermark = layers.Input(shape=(w_rows, w_cols, 1), name='input_watermark')
    
    # Rearrange input 
    rearranged_img = l1 = layers.Lambda(tf.space_to_depth, arguments={'block_size':block_size}, name='rearrange_img')(input_img)
    
    
    dct_layer = layers.Conv2D(64, (1, 1), activation='linear', padding='same', use_bias=False, trainable=False, name='dct1')
    dct_layer2 = layers.Conv2D(64, (1, 1), activation='linear', padding='same', use_bias=False, trainable=False, name='dct2')
    idct_layer = layers.Conv2D(64, (1, 1), activation='linear', padding='same', use_bias=False, trainable=False, name='idct')
    dct_layer_img = dct_layer(rearranged_img)
    
    # Concatenating The Image's dct coefs and watermark
    encoder_input = layers.Concatenate(axis=-1, name='encoder_input')([dct_layer_img, input_watermark])
    
    # Encoder
    encoder_model = layers.Conv2D(64, (1, 1), dilation_rate=1, activation='elu', padding='same', name='enc_conv1')(encoder_input)
    encoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv2')(encoder_model)
    encoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv3')(encoder_model)
    encoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv4')(encoder_model)
    encoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv5')(encoder_model)
    encoder_model = idct_layer(encoder_model)
    
    # Strength
    encoder_model = layers.Lambda(multiply_scalar, arguments={'scalar':input_strenght_alpha}, name='strenght_factor')(encoder_model)
    
    encoder_model = layers.Add(name='residual_add')([encoder_model, l1])
    encoder_model = x = layers.Lambda(tf.depth_to_space, arguments={'block_size':block_size}, name='enc_output_depth2space')(encoder_model)
    
    # Attack (The attacks occure in test phase)
    
    # Watermark decoder
    input_attacked_img = layers.Input(shape=(patch_rows, patch_cols, 1), name='input_attacked_img')
    decoder_model = layers.Lambda(tf.space_to_depth, arguments={'block_size':block_size}, name='dec_input_space2depth')(input_attacked_img)
    decoder_model = dct_layer2(decoder_model)
    decoder_model = layers.Conv2D(64, (1, 1), dilation_rate=1, activation='elu', padding='same', name='dec_conv1')(decoder_model)
    decoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='dec_conv2')(decoder_model)
    decoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='dec_conv3')(decoder_model)
    decoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='dec_conv4')(decoder_model)
    decoder_model = layers.Conv2D(1, (1, 1), dilation_rate=1, activation='sigmoid', padding='same', name='dec_output_depth2space')(decoder_model)
    
    # Whole model
    embedding_net = tf.keras.models.Model(inputs=[input_img, input_watermark, input_strenght_alpha], outputs=[x])
    extractor_net = tf.keras.models.Model(inputs=[input_attacked_img], outputs=[decoder_model])
    
    # Set weights
    DCT_MTX = sio.loadmat('./transforms/DCT_coef.mat')['DCT_coef']
    dct_mtx = np.reshape(DCT_MTX, [1,1,64,64])
    embedding_net.get_layer('dct1').set_weights(np.array([dct_mtx]))
    extractor_net.get_layer('dct2').set_weights(np.array([dct_mtx]))
    
    IDCT_MTX = sio.loadmat('./transforms/IDCT_coef.mat')['IDCT_coef']
    idct_mtx = np.reshape(IDCT_MTX, [1,1,64,64])
    embedding_net.get_layer('idct').set_weights(np.array([idct_mtx]))
    
    embedding_net.load_weights(model_path,by_name = True)
    extractor_net.load_weights(model_path,by_name = True)
    return embedding_net, extractor_net

# %%
# Test images
img_rows, img_cols = 512, 512
test_folder = './images/{}x{}'.format(img_rows, img_cols)
test_imgs_files = [f for f in listdir(test_folder) if isfile(join(test_folder, f)) and (f.endswith('.bmp') or f.endswith('.gif'))]

# Exp Info
exp_id = 'MT-Net'
use_circular = True
save_samples = True
patch_rows, patch_cols = 32, 32
block_size = 8
Is_mean_normalized = True
mean_normalize = 128.0
std_normalize = 255.0

assert patch_rows == patch_cols, 'Patches must have same rows and columns'
assert img_rows % patch_rows == 0 and img_cols % patch_cols == 0, 'Image size must be dividable by the patch size'

print('Analyzing Experiment {}...'.format(exp_id))
time.sleep(1)

# Log folder
analysis_folder = './logs/{}/Analysis'.format(exp_id)
if os.path.exists(analysis_folder) == False:
    os.mkdir(analysis_folder)

sampled_embeded_folder = os.path.join(analysis_folder, 'Sampled Embeddings')
if os.path.exists(sampled_embeded_folder) == False:
    os.mkdir(sampled_embeded_folder)
    
sampled_attack_folder = os.path.join(analysis_folder, 'Sampled Attacks')
if os.path.exists(sampled_attack_folder) == False:
    os.mkdir(sampled_attack_folder)

# Model Definition
model_path = './logs/{}/Weights/weights_final.h5'.format(exp_id)
embedding_net, extractor_net = buildModel(model_path,use_circular=use_circular)

# List of Attacks
# Rotation
def rotate(img, Q): # Q = rotation degree
    im = PIL.Image.fromarray(img)
    rotated_img = im.rotate(Q, resample=Image.BILINEAR)
    return np.array(rotated_img)

#JPEG
def jpg(img, Q):
    global exp_id
    cv2.imwrite('temporary_files/temp_{}.jpg'.format(exp_id), img, [cv2.IMWRITE_JPEG_QUALITY, int(Q)])
    Iw_attacked = cv2.imread('temporary_files/temp_{}.jpg'.format(exp_id), cv2.IMREAD_GRAYSCALE)
    return Iw_attacked

# Salt and Pepper Noise
def salt_pepper_noise(img, Q): # Q = percentage of noise
    salt_ratio = Q / 2
    
    random_image = np.random.uniform(low=0.0, high=1.0, size=img.shape)
    salt_image = 255.0*(random_image > (1-salt_ratio))
    pepper_image = 255.0*(random_image > salt_ratio)
    
    noisy_img = np.minimum(pepper_image, np.maximum(img, salt_image))
    
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    noisy_img = np.uint8(np.round(noisy_img))
    return noisy_img

# Guassian White Noise
def gaussin_noise(img, Q): # Q = std
    noisy_img =  img + np.random.normal(loc=0.0, scale=Q, size=img.shape)
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    noisy_img = np.uint8(np.round(noisy_img))
    return noisy_img

# Setting A Random window to Zero
def cropping_fillzero(img, Q): # Q = percentage
    h, w = img.shape[:2]
    blacked_pixels = int(Q * (h*w))
    height = int(np.floor(np.sqrt(blacked_pixels)))
    width = int(np.ceil(np.sqrt(blacked_pixels)))

    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    
    tmp_img = img.copy()
    tmp_img[y:y+height, x:x+width] = 0
    
    return tmp_img

def median_filt(img, Q): # Q = window size
    return median_filter(img, Q)

# Low-Pass filter (Average filtering)
def smoothing_mean(img, Q): # Q = window size
    n_channels = 1
    x_tmp = img.copy()
    if len(x_tmp.shape) == 3 and x_tmp.shape[2] == 3:
        n_channels = 3
    elif len(img.shape) == 2:
        x_tmp = x_tmp[..., np.newaxis]

    mean_kernel = np.full([Q, Q, n_channels], (1.0 / (Q * Q)))
    x_tmp = convolve(x_tmp, mean_kernel, mode='reflect')
    return x_tmp if n_channels == 3 else x_tmp[:,:,0]

# Low-Pass filter (Gaussian)
def smoothing_gaussian(img, Q): # Q = std of filter
    blurred_img = gaussian_filter(img, Q, mode='reflect')
    return blurred_img

# High-pass filter (Sharpening)
def sharpening(img, Q): # Q = radius
    im = PIL.Image.fromarray(img)
    im = im.filter(PIL.ImageFilter.UnsharpMask(radius=Q, percent=150, threshold=3))
    return np.array(im)

# Painting Attack
def painting(img, Q): # number of lines
    sentences = ['All that glitters is not gold',
                 'To thine own self be true, and it must ',
                 'follow, as the night the day, ',
                 'thou canst not then be false to any man.',
                 'Hell is empty and all the devils are here.',
                 'Love all, trust a few, do wrong to none.',
                 'By the pricking of my thumbs, Something wicked',
                 'this way comes. Open, locks, Whoever knocks!',
                 'These violent delights have violent ends...',
                 'Good night, good night! parting is such sweet ',
                 'sorrow, That I shall say good night till it be morrow.',
                 'The lady doth protest too much, methinks.',
                 'Brevity is the soul of wit.',
                 'Glendower: I can call spirits from the vasty deep.' ,
                 'Hotspur: Why, so can I, or so can any man; ',
                 'But will they come when you do call for them?',
                 ]
    n_sentences = Q
    font = 'Natural script.otf'
    font_size = 40
    im = img.copy()
    im = PIL.Image.fromarray(im).convert('RGBA')
    
    txt = Image.new('RGBA', im.size, (255,255,255,0))
    # get a font
    fnt = ImageFont.truetype(font, font_size)

    d = ImageDraw.Draw(txt)
    for i in range(min(n_sentences, len(sentences))):
        d.text((10,10+i*30), sentences[i], font=fnt, fill=(255,255,255,255))

    out = Image.alpha_composite(im, txt)
    out = out.convert('L')
    return np.array(out)


# Gradient attack
def gradient_attack(img, Q): # Q = gaussian std
    im = PIL.Image.fromarray(img)
    im_blur = im.filter(PIL.ImageFilter.GaussianBlur(radius=Q))
    return np.array(im, dtype=np.uint8)-np.array(im_blur, dtype=np.uint8)

def gradient_attack_mean(img, Q):
    blured_im = img.copy()
    mean_kernel = np.full([Q, Q], (1.0 / (Q * Q)))
    blured_im = convolve(blured_im, mean_kernel, mode='reflect')
    edge_img = img - blured_im
    return edge_img
    
# Resize Attack
def resize_attack(img, Q): # Q = resizing ratio
    im = PIL.Image.fromarray(img)
    w, h = im.size
    new_w, new_h = int(w*Q), int(h*Q)
    resized_img = im.resize((new_w, new_h), Image.BILINEAR)
    new_img = resized_img.resize((w, h), Image.BILINEAR)
    return np.array(new_img)

def no_attack(x,Q):
    return x

def grid_crop(img, Q, orig_img=None): # Q ratios
    block_size = 8
    block_switch = np.random.uniform(0.0, high=1.0, size=[img.shape[0]//block_size, img.shape[0]//block_size])
    block_switch = block_switch < Q
    new_img = img.copy()
    for i in range(block_switch.shape[0]):
        for j in range(block_switch.shape[1]):
            if block_switch[i,j] == 0:
                continue
            new_img[block_size*i:block_size+block_size*i, block_size*j:block_size+block_size*j] = 0
    return new_img

attacks_list = [
                {'func':jpg, 'name':'JPEG', 'params': np.array([90, 70, 60, 50, 30]), 'active':True}, # JPEG
                {'func':smoothing_mean, 'name':'Average Filter', 'params': np.array([3, 5, 7, 9]), 'active':True}, # Smoothing mean
                {'func':smoothing_gaussian, 'name':'Gaussian Filter', 'params':np.arange(1, 3, 0.2), 'active':True}, # Smoothing gaussian
                {'func':sharpening, 'name':'Sharpening', 'params':np.arange(1, 50, 1), 'active':True},
                {'func':cropping_fillzero, 'name':'Cropping ZeroFill', 'params':np.arange(0.01, 0.4, 0.05), 'active':True}, # Cropping random patches and set to zero
                {'func':gaussin_noise, 'name':'Gaussian Noise', 'params':np.arange(1, 30, 1), 'active':True}, # Gaussian noise},
                {'func':salt_pepper_noise, 'name':'Salt & Pepper Noise', 'params':np.arange(0.01, 0.2, 0.01), 'active':True}, # Salt and pepper noise}
                {'func':median_filt, 'name': 'Median Filter', 'params':np.array([3, 5, 7, 9], dtype=np.int32),'active':True},
                {'func':painting, 'name':'Painting', 'params':np.array([1, 3, 6, 9, 12, 15, 16, 18], dtype=np.int32), 'active':True},
                {'func':rotate, 'name':'Rotation', 'params':np.arange(-10.0,11.0,1), 'active':True},
                {'func':gradient_attack, 'name':'Edges', 'params':np.arange(0.5,1.1,0.1), 'active':True},
                {'func':resize_attack, 'name':'Resizing', 'params':np.arange(0.25, 2.25, 0.25), 'active':True},
                {'func':grid_crop, 'name':'Grid Crop', 'params':np.arange(0.01, 0.55, 0.05), 'active':True},
                ] 

# %%
# Expriments
alpha_values = np.array(range(1, 5+1, 1)) / (5.0)
num_random_watermarks = 1

w_rows = int((patch_rows) / block_size)
w_cols = int((patch_cols) / block_size)
bits_per_patch = w_rows * w_cols
n_patches = int((img_rows * img_cols) / (patch_rows * patch_cols))
total_cap = n_patches * bits_per_patch

message_length = 1024
analysis_folder = os.path.join(analysis_folder, '{}-Bits'.format(message_length))
if os.path.exists(analysis_folder) == False:
    os.mkdir(analysis_folder)
assert total_cap % message_length == 0, 'Total Capacity must be dividable by message length'
n_redundancy = total_cap // message_length

# %%
psnr_means = []
psnr_stds = []
ssim_means = []

# Computing PSNRs
print('Computing PSNRs...')

for test_img in test_imgs_files:
    print('\tProcessing ', test_img, ' ...')
    im = plt.imread(os.path.join(test_folder, test_img))
#    if im == None:
#        print('\t[!] Error loading ', test_img)
#        continue
    if im.shape[-1] == 3:
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = im
    
    # Normalize image
    Im_normalized = (im_gray.copy() - mean_normalize if Is_mean_normalized else 0) / std_normalize
    
    num_batch = (img_rows * img_cols) // (patch_rows * patch_cols)
    psnr_values_per_alpha_mean = []
    psnr_values_per_alpha_std = []
    ssim_values_per_alpha_mean = []
    
    for alpha in alpha_values:    
        # Compute PSNRs
        tmp_psnr = []
        tmp_ssim = []
        for n in range(num_random_watermarks ):
            Im_32x32_patchs = vf.partitioning(Im_normalized, p_size=patch_rows)
            W = np.random.randint(low=0, high=2, size=(num_batch, w_rows, w_cols, 1)).astype(np.float32)
            # Apply embedding network
            Iw_batch = embedding_net.predict_on_batch([Im_32x32_patchs, W, np.array([alpha])])
            # reconstruct Iw
            Iw = vf.tiling(Iw_batch, rec_size=img_rows)
            Iw *= std_normalize
            Iw += mean_normalize if Is_mean_normalized else 0
            Iw[Iw > 255] = 255
            Iw[Iw < 0] = 0
            Iw = np.uint8(Iw.squeeze())
            # PSNR
            #psnr = 10*np.log10(255**2/np.mean((im_gray - Iw)**2))
            psnr = compare_psnr(im_gray, Iw, data_range=255)
            tmp_psnr.append(psnr)
            # SSIM
            tmp_ssim.append(compare_ssim(im_gray, Iw, win_size=9, data_range=255))
            
            # Save sample image
            if n == 0 and save_samples == True:
                cv2.imwrite(os.path.join(sampled_embeded_folder, '{}_[{}].png'.format(test_img[:-4], alpha)), Iw)
        
        psnr_values_per_alpha_mean.append(np.mean(tmp_psnr))
        psnr_values_per_alpha_std.append(np.std(tmp_psnr))
        ssim_values_per_alpha_mean.append(np.mean(tmp_ssim))
    
    psnr_means.append(psnr_values_per_alpha_mean)
    psnr_stds.append(psnr_values_per_alpha_std)
    ssim_means.append(ssim_values_per_alpha_mean)
 
# %% 
print('Computing BER....')

bers_per_attacks = []
for attack in attacks_list: # for all attacks
    sampled_folder_per_attack = os.path.join(sampled_attack_folder, attack['name'])
    if os.path.exists(sampled_folder_per_attack) == False:
        os.mkdir(sampled_folder_per_attack)

    if attack['active'] == False:
        continue
    
    print('\tPerforming {} attack...'.format(attack['name']))
    
    bers_per_attack_params = []
    for attack_params in tqdm(attack['params']): # for all attack params

        bers_per_attack_per_image = []
        bers_per_attack_per_image_robust = []
        for test_img in test_imgs_files:  # for all images
            im = plt.imread(os.path.join(test_folder, test_img))
            if im.shape[-1] == 3:
                im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            else:
                im_gray = im
            
            # Normalize image
            Im_normalized = (im_gray.copy() - mean_normalize if Is_mean_normalized else 0) / std_normalize
            
            bers_per_alpha = []
            
            for alpha in alpha_values:
                tmp_bers = []
                for n in range(num_random_watermarks):
                    Im_32x32_patchs = vf.partitioning(Im_normalized, p_size=patch_rows)

                    #W = np.random.randint(low=0, high=2, size=(n_patch, w_rows, w_cols, 1)).astype(np.float32)
                    W = np.random.randint(low=0, high=2, size=(message_length,1)).astype(np.float32)
                    W = np.reshape(W, [-1, w_rows, w_cols])

                    #Redundant embedding
                    N_mtx = (img_rows//patch_rows) # assuming img_rows == img_cols
                    W_robust = np.zeros([N_mtx, N_mtx, w_rows, w_cols], dtype=np.float32)
                    k = 0
                    n_repeats = 0
                    for d in range(N_mtx):
                        for i in range(N_mtx):
                            if k >= W.shape[0]:
                                break
                            W_robust[i, (i+d)%N_mtx, :, :] = W[k, :, :] ##### % 8 for 256x256
                            n_repeats += 1
                            if n_repeats >= n_redundancy:
                                n_repeats = 0
                                k += 1
                                
                    W_robust = np.reshape(W_robust, [-1, w_rows, w_cols, 1])

                    # Apply embedding network
                    Iw_batch = embedding_net.predict_on_batch([Im_32x32_patchs, W_robust, np.array([alpha])])

                    # reconstruct Iw
                    Iw = vf.tiling(Iw_batch, rec_size=img_rows)
                    Iw *= std_normalize
                    Iw += mean_normalize if Is_mean_normalized else 0
                    Iw[Iw > 255] = 255
                    Iw[Iw < 0] = 0
                    Iw = np.uint8(Iw.squeeze())
                    
                    # Apply Attack
                    Iw_attacked = attack['func'](Iw, attack_params)

                    Iw_tmp = Iw_attacked
                    
                    Iw_attacked = (Iw_attacked - mean_normalize if Is_mean_normalized else 0) / std_normalize
                    
                    
                    Iw_attacked_patchs = vf.partitioning(Iw_attacked, p_size=patch_rows)
        
                    # Feed to extractor
                    w_batch = extractor_net.predict_on_batch([Iw_attacked_patchs])
                    w_batch = w_batch > 0.5

                    # Majority voting
                    w_batch = np.reshape(w_batch, [N_mtx, N_mtx, w_rows, w_cols])
                    w_extracted = np.zeros_like(W)
                    
                    k = 0
                    n_repeats = 0
                    
                    for d in range(N_mtx):
                        for i in range(N_mtx):
                            if k >= w_extracted.shape[0]:
                                break
                            w_extracted[k, :, :] += w_batch[i, (i+d)%N_mtx, :, :] ####### % 8 for 256x256 ?
                            n_repeats += 1
                            if n_repeats >= n_redundancy:
                                n_repeats = 0
                                k += 1
                    
                    w_extracted = (w_extracted > n_redundancy//2)

        
                    # Compute BER
                    xor_w = (W != w_extracted)
                    ber = np.sum(xor_w) / (message_length)
                    tmp_bers.append(ber)
                    
                    # Same sample
                    if n == 0 and alpha == 1.0 and save_samples == True:
                        file_name = '{}_{}_{:.2f}_{:.3f}.png'.format(attack['name'], test_img[:-4], attack_params, ber)
                        Iw_tmp = np.uint8(Iw_tmp)
                        cv2.imwrite(os.path.join(sampled_folder_per_attack, file_name), Iw_tmp)
                
                bers_per_alpha.append(np.mean(tmp_bers))
            
            bers_per_attack_per_image.append(bers_per_alpha)
        bers_per_attack_params.append(bers_per_attack_per_image)
    bers_per_attacks.append(np.array(bers_per_attack_params))

bers_array = bers_per_attacks # [attack, attack_param, images, alpha]

# Save Results in a mat file
print('Saving Results...')
var_dict = {}


var_dict['psnr_means'] = psnr_means
var_dict['psnr_std'] = psnr_stds
var_dict['ssim_means'] = ssim_means

attack_idx = - 1
for idx, attack in tqdm(enumerate(attacks_list)):
    if attack['active'] == False:
        continue
    attack_idx += 1
    var_dict[ attack['name'] ] = bers_array[attack_idx]

var_dict['images'] = test_imgs_files
sio.savemat(os.path.join(analysis_folder, 'ber_analysis.mat'), mdict=var_dict)

# %% Plotting
# Plot PSNR
plt.figure(figsize=(8,4))
for i in range(len(psnr_means)):
    plt.errorbar(x=alpha_values, y=psnr_means[i], yerr=1*np.array(psnr_stds[i]), marker='^')

#plt.xticks(np.arange(len(alpha_values)), alpha_values)
plt.xlabel('Strength Factor(Alpha)')
plt.ylabel('PSNR')
plt.title('PSNR Per Alpha')
plt.legend(test_imgs_files)
plt.savefig(os.path.join(analysis_folder, 'PSNR.png'))

# Plot BERs per Attack
attack_idx = -1
for i, attack in enumerate(attacks_list):
    if attack['active'] == False:
        continue
    attack_idx += 1
    print('Plotting {}...'.format(attack['name']))
    tmp_ber_array = bers_array[attack_idx]
    mean_bers = np.mean(tmp_ber_array, axis=1)
    
    # 3D Plot
    fig = plt.figure(figsize=(8,4))
    ax = Axes3D(fig)
    X, Y = np.meshgrid(alpha_values, attack['params'])
    ax.plot_surface(X , Y, mean_bers, cmap=cm.coolwarm, antialiased=True)
    ax.set_xlabel('Strength Factor')
    ax.set_ylabel('Parameters')
    ax.set_zlabel('BER')
    ax.set_title(attack['name'])
    fig.savefig(os.path.join(analysis_folder, '{}_3D.png'.format(attack['name'])))
    
    # 2D Line plot
    selected_params = sorted(np.random.permutation(len(attack['params']))[:4])
    plt.figure(figsize=(8,4))
    plt.plot(mean_bers.T[:, selected_params])
    plt.xticks(np.arange(len(alpha_values)), alpha_values)
    plt.xlabel('Strength Factor(Alpha)')
    plt.ylabel('BER')
    plt.legend(attack['params'][selected_params])
    plt.title(attack['name'])
    plt.savefig(os.path.join(analysis_folder, '{}_2D_sampled.png'.format(attack['name'])))
    
#impulse responce for a flat image
Im = np.ones((256,256)) * 0.5
Im_32x32_patchs = vf.partitioning(Im, p_size=32)
W = np.zeros((64,4,4,1))
Iw_batch = embedding_net.predict_on_batch([Im_32x32_patchs, W, np.array([1.0])])
Iw_zeros = vf.tiling(Iw_batch ,256)
W[:,1,1,:] = 1
Iw_batch = embedding_net.predict_on_batch([Im_32x32_patchs, W, np.array([1.0])])
Iw_impulse = vf.tiling(Iw_batch ,256)
Impulse_responce = Iw_impulse-Iw_zeros

plt.figure(figsize=(8,4))
plt.imshow(Impulse_responce[0:32,0:32],cmap=cm.inferno)
plt.colorbar()
plt.savefig(os.path.join(analysis_folder, '{}.png'.format('Impulse_responce')))

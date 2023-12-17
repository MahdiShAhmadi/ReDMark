"""

*** MT-Net - multi-attack(salt & pepper(4%), gaussian noise(std=3), jpeg(70) and smoothing(3x3)), loss 1:1, cifar10 + resampled_pascal 
*** GT-Net - Single Attack, Gaussian noise, loss 3:1, cifar10+resampled_pascal, 100 epochs, lr=1e-4
*** JT-Net - Single Attack, JPEG(70), loss 3:1, cifar10+resampled_pascal, 100 epochs, lr=1e-4

"""

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm
import os
from include import loss_functions
from include.my_circular_layer import Conv2D_circular

keras = tf.keras
layers = keras.layers
K = keras.backend


def multiply_255(x):
    return x*255.0   

def divide_255(x):
    return x/255.0 

def scalar_output_shape(input_shape):
    return input_shape

def multiply_scalar(x, scalar):
    return x * tf.convert_to_tensor(scalar, tf.float32)

def dropout_blocks(x):
    noise = tf.random.uniform(shape=[1,4,4,1],maxval=1,dtype=tf.float32,seed=None)
    noise = noise > 0.25
    noise = tf.cast(noise, tf.float32)
#    noise = tf.convert_to_tensor(noise, dtype='float32')
    return x*noise

def salt_pepper_noise(x, salt_ratio):
    # Thanks to https://tyfkda.github.io/blog/2016/09/22/tensorflow-salt-pepper.html
    random_image = tf.random.uniform(shape=[1,4,4,64],minval=0.0, maxval=1,dtype=tf.float32,seed=None)
    
    salt_image = tf.cast(tf.greater_equal(random_image, 1.0 - salt_ratio), tf.float32) - 0.5
    pepper_image = tf.cast(tf.greater_equal(random_image, salt_ratio), tf.float32) - 0.5
    
    noised_image = tf.minimum(tf.maximum(x, salt_image), pepper_image)
    return noised_image

def random_switch_new(x, prob):
    sampled_x = np.random.choice(x, size=1, p=prob)
    return sampled_x[0]
    
def random_switch(x, prob):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    noise_A = tf.random.uniform(shape=[1],maxval=1,dtype=tf.float32,seed=None)
    noise_B = tf.random.uniform(shape=[1],maxval=1,dtype=tf.float32,seed=None)
    noise0 = noise_A >= 0.5
    noise0 = tf.cast(noise0, tf.float32)
    noise1 = noise_A < 0.5
    noise1 = tf.cast(noise1, tf.float32)
    noise2 = noise_B >= 0.5
    noise2 = tf.cast(noise2, tf.float32)
    noise3 = noise_B < 0.5
    noise3 = tf.cast(noise3, tf.float32)
    return noise2*(x0*noise0 + x1*noise1) + noise3*(x2*noise0 + x3*noise1)

def UniformNoise(x, val):
    noise = tf.random.uniform(shape=[32,4,4,64],minval=-val,maxval=val,dtype=tf.float32,seed=None)
    return x + noise
    

Q = 70
jpeg_noise = 0.55
q_mtx = sio.loadmat('./transforms/jpeg_qm.mat')['qm']
q_mtx = q_mtx.astype('float32')
if (Q < 50):
    S = 5000/Q
else:
    S = 200 - 2*Q
q_mtx = np.floor((S * q_mtx + 50.0) / 100.0)
q_mtx = np.reshape(q_mtx ,(64,1))*1.0
q_mtx = np.repeat(q_mtx[np.newaxis,...], 4, axis=0)
q_mtx = np.repeat(q_mtx[np.newaxis,...], 4, axis=0)
q_mtx = np.squeeze(q_mtx) 
q_mtx[q_mtx == 0] = 1


# input image dimensions
img_rows, img_cols = 32, 32
block_size = 8

use_circular = True 

conv2d_layer = layers.Conv2D if use_circular == False else Conv2D_circular

combine_cifar_pascal = True
selected_dataset = 'cifar' # cifar or pascal

# the data, split between train and test sets
print('Loading dataset...')
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = keras.datasets.cifar10.load_data()
x_train_cifar = x_train_cifar.reshape(x_train_cifar.shape[0], img_rows, img_cols, 3)
x_train_cifar = x_train_cifar[:,:,:,1]
x_train_cifar = x_train_cifar.reshape(x_train_cifar.shape[0], img_rows, img_cols, 1)
x_test_cifar = x_test_cifar.reshape(x_test_cifar.shape[0], img_rows, img_cols, 3)
x_test_cifar = x_test_cifar[:,:,:,1]
x_test_cifar = x_test_cifar.reshape(x_test_cifar.shape[0], img_rows, img_cols, 1)

x_train_pascal = sio.loadmat('./images/pascal/pascal_resampled.mat')['patches']
x_train_pascal = x_train_pascal[..., np.newaxis]

# Combine 
if combine_cifar_pascal == False:
    x_train = x_train_cifar if selected_dataset == 'cifar' else x_train_pascal
else:
    x_train = np.concatenate([x_train_cifar, x_train_pascal], axis=0)

x_train = x_train.astype('float32')
x_train = (x_train-128.0)/255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

# Watermark encoder

input_img = layers.Input(shape=(img_rows, img_cols, 1), name='input_img')

w_rows = int((img_rows) / block_size)
w_cols = int((img_cols) / block_size)
input_watermark = layers.Input(shape=(w_rows, w_cols, 1), name='input_watermark')

# Rearrange input 
rearranged_img = l1 = layers.Lambda(tf.nn.space_to_depth, arguments={'block_size':block_size}, name='rearrange_img')(input_img)
trainable_transform = False 
num_of_filters = block_size**2  
dct_layer = layers.Conv2D(num_of_filters, (1, 1), activation='linear', padding='same', use_bias=False, trainable=trainable_transform, name='dct')
idct_layer = layers.Conv2D(num_of_filters, (1, 1), activation='linear', padding='same', use_bias=False, trainable=trainable_transform, name='idct')
dct_layer_img = dct_layer(rearranged_img)

# Concatenating The Image's dct coefs and watermark
encoder_input = layers.Concatenate(axis=-1, name='encoder_input')([dct_layer_img, input_watermark])

# Encoder
encoder_model = layers.Conv2D(num_of_filters, (1, 1), dilation_rate=1, activation='elu', padding='same', name='enc_conv1')(encoder_input)
encoder_model = conv2d_layer(num_of_filters, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv2')(encoder_model)
encoder_model = conv2d_layer(num_of_filters, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv3')(encoder_model)
encoder_model = conv2d_layer(num_of_filters, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv4')(encoder_model)
encoder_model = conv2d_layer(num_of_filters, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv5')(encoder_model)
encoder_model = idct_layer(encoder_model)
encoder_model = layers.Add(name='residual_add')([encoder_model, l1])

x = layers.Lambda(tf.nn.depth_to_space, arguments={'block_size':block_size}, name='enc_output_depth2space')(encoder_model)

##################### Noise Attack  #############################
noise_std = 3.0
noise_attacked = layers.Activation(multiply_255)(encoder_model)
noise_attacked = layers.GaussianNoise(stddev=noise_std, name='guassian_noise_attack')(noise_attacked)
noise_attacked = layers.Activation(divide_255)(noise_attacked)

###################  Dropout block-wise Attack  ############################
dropout_attacked_blocks = layers.Lambda(dropout_blocks, name='blockwise_dropout_attack')(encoder_model)

###################  Salt and Pepper Attack ############################
salt_pepper_attacked = layers.Lambda(salt_pepper_noise, arguments={'salt_ratio':0.04}, name='salt_pepper_attack')(encoder_model)

#####################  Jpeg_attake   ############################
jpeg_attaked = dct_layer(encoder_model)
jpeg_attaked = layers.Lambda(lambda x: (x*255) / q_mtx, output_shape=scalar_output_shape, name='jpg1')(jpeg_attaked)
#jpeg_attaked = layers.Lambda(keras.backend.round)(jpeg_attaked)
jpeg_attaked = layers.Lambda(UniformNoise, arguments={'val':jpeg_noise})(jpeg_attaked)
jpeg_attaked = layers.Lambda(lambda x: (x/255) * q_mtx, output_shape=scalar_output_shape, name='jpg2')(jpeg_attaked)
jpeg_attaked = idct_layer(jpeg_attaked)

#####################    smoothing_attak  #######################
smoothing_layer = layers.Conv2D(1, (3, 3), dilation_rate=1, padding='same', name='smoothing_attak',use_bias=False, trainable=False)
mean_attacked = smoothing_layer(x)
mean_attacked = layers.Lambda(tf.nn.space_to_depth, arguments={'block_size':block_size}, name='mean_attack_space2depth')(mean_attacked)

#####################    sharpenning(edge)_attak  #######################
mean = smoothing_layer(x)
mean = layers.Lambda(multiply_scalar, arguments={'scalar':-1.0})(mean)
x2 = layers.Lambda(multiply_scalar, arguments={'scalar':1.0})(x)

sharpenning_attacked = layers.Add(name='sharpenning_subtract')([x2, mean])
sharpenning_attacked = layers.Lambda(tf.nn.space_to_depth, arguments={'block_size':block_size}, name='sharpening_attack_space2depth')(sharpenning_attacked)

#################   Random selection of attacks  #################
identity_layer = encoder_model

attack_list = [salt_pepper_attacked, noise_attacked, jpeg_attaked, mean_attacked]

attack_prob = None 
assert attack_prob is None or np.sum(attack_prob) == 1.0, 'Sum of probabilities for attacks must be 1.0'

attacked_Iw = layers.Lambda(random_switch, arguments={'prob':attack_prob}, name='Random_selection_of_attacks')(attack_list)
rounding_noise = layers.GaussianNoise(stddev=0.003, name='rounding_noise')(attacked_Iw)

# Watermark decoder
decoder_model = dct_layer(rounding_noise)
decoder_model = layers.Conv2D(num_of_filters, (1, 1), dilation_rate=1, activation='elu', padding='same', name='dec_conv1')(decoder_model)
decoder_model = conv2d_layer(num_of_filters, (2, 2), dilation_rate=1, activation='elu', padding='same', name='dec_conv2')(decoder_model)
decoder_model = conv2d_layer(num_of_filters, (2, 2), dilation_rate=1, activation='elu', padding='same', name='dec_conv3')(decoder_model)
decoder_model = conv2d_layer(num_of_filters, (2, 2), dilation_rate=1, activation='elu', padding='same', name='dec_conv4')(decoder_model)
decoder_model = layers.Conv2D(1, (1, 1), dilation_rate=1, activation='sigmoid', padding='same', name='dec_output_depth2space')(decoder_model)

# Whole model
model = keras.models.Model(inputs=[input_img, input_watermark], outputs=[x, decoder_model])
model.summary()

# Set weights
dct_mtx = sio.loadmat('./transforms/DCT_coef.mat')['DCT_coef']
dct_mtx = np.reshape(dct_mtx, [1,1,num_of_filters,num_of_filters])
model.get_layer('dct').set_weights(np.array([dct_mtx]))

idct_mtx = sio.loadmat('./transforms/IDCT_coef.mat')['IDCT_coef']
idct_mtx = np.reshape(idct_mtx, [1,1,num_of_filters,num_of_filters])
model.get_layer('idct').set_weights(np.array([idct_mtx]))

mean_attack = 1/9.0 * np.ones((3,3,1,1))  ################## <LOAD WEIGHTS> ##########################
#model.get_layer('smoothing_attak').set_weights([mean_attack])
#keras.utils.plot_model(model, to_file='model_general_dct.png')

# Define loss
ssim_win_size = 8
loss_object = loss_functions.SSIM_MSE_LOSS(ssim_relative_loss=1.0,mse_relative_loss=0.0,ssim_win_size=ssim_win_size)
ssimmse_loss = loss_object.ssimmse_loss

lr = 1e-4
enc_output_weight = 1.0
dec_output_weight = 1.0
model.compile(loss={'enc_output_depth2space':ssimmse_loss , 'dec_output_depth2space': 'binary_crossentropy'},#'mean_squared_error'},
                loss_weights={'enc_output_depth2space': enc_output_weight, 'dec_output_depth2space': dec_output_weight},
              optimizer=keras.optimizers.SGD(lr=lr, momentum=0.98))


# %% Training 
exp_id = 'new_model'

if os.path.exists('./logs/{}'.format(exp_id)) == False:
    os.mkdir('./logs/{}'.format(exp_id))
    os.mkdir('./logs/{}/Weights'.format(exp_id))
    
log_dir = './logs/{}'.format(exp_id)
tf_logger = tf.summary.create_file_writer(log_dir)


batch_size = 32
epochs = 100
offset = 0 # To be able to continue training
steps = 10000 #int(np.ceil(60000 / batch_size))

for e in range(epochs):
    print('Epochs {}...'.format(e+1))
    loss_w = []
    loss_I = []
    for step in tqdm(range(steps)):
        img_idx = np.random.randint(0, x_train.shape[0], batch_size)
        water_idx = np.random.randint(0, x_train.shape[0], batch_size)
    
        I = x_train[img_idx, :,:,:]
        
        W = np.random.randint(low=0, high=2, size=(batch_size, w_rows, w_cols,1)).astype(np.float32)

        encoder_output = I
        decoder_output = W
        
        model.train_on_batch(x=[I, W], y=[encoder_output, decoder_output])
        model_output = model.predict_on_batch([I, W])
        
        loss_I.append(((model_output[0] - encoder_output)**2).mean())
        loss_w.append(((model_output[1] - decoder_output)**2).mean())
    
        if step % 1000 == 0:
            print('\tStep {}...'.format(step+1))
            Iw_encoder = model_output[0]
            W_decoder = model_output[1][0,:,:,0]
            
            plt.subplot(221)
            plt.imshow(I[0,:,:,0], cmap='gray')
            plt.title('CoverImage[I]')
            
            plt.subplot(222)
            plt.imshow(W[0,:,:,0], cmap='gray')
            plt.title('Watermark[W]')
            
            plt.subplot(223)
            plt.imshow(Iw_encoder[0,:,:,0], cmap='gray')
            plt.title('Iw')
            
            plt.subplot(224)
            plt.imshow(W_decoder, cmap='gray')
            plt.title('Extracted W')
            
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            
    mean_error_w = np.mean(loss_w)
    mean_error_I = np.mean(loss_I)
    psnr = 10*np.log10(1**2/mean_error_I)
    print('\tI Error = {} And W Error = {}'.format(mean_error_I, mean_error_w))
    print('PSNR is: ' ,psnr)
    with tf_logger.as_default():
        tf.summary.scalar('W_MSE', mean_error_w, e+1)
        tf.summary.scalar('I_MSE', mean_error_I, e+1)
        tf.summary.scalar('PSNR', psnr, e+1)
    
    if (e+1) % 10 == 0:
        model.save_weights('./logs/{}/Weights/weights_{}.h5'.format(exp_id, e+1+offset))
    
            
model.save_weights('./logs/{}/Weights/weights_final.h5'.format(exp_id))

# Save info
with open('./logs/{}/exp_info.txt'.format(exp_id), 'w') as f:
    f.write('EXP ID : {}\n'.format(exp_id))
    f.write('Block size : {}\n'.format(block_size))
    f.write('LR : {}\n'.format(lr))
    f.write('Epochs : {}, Steps : {}, Offset : {}\n'.format(epochs, steps, offset))
    f.write('Batch size : {}\n'.format(batch_size))
    f.write('Relative loss : Iw={} - W={}\n'.format(enc_output_weight, dec_output_weight))
    f.write('Gaussian Noise STD : {}\n'.format(noise_std))
    f.write('Trainble Transform : {}\n'.format(trainable_transform))
    f.write('PSNR on last epoch : {}\n'.format(psnr))

import numpy as np

def partitioning(unpartitioned, p_size):
    up_size = unpartitioned.shape[0]
    parts = np.zeros((int((up_size/p_size)**2), p_size, p_size, 1))
    n=0
    for u in range(int(up_size/p_size)):
        for v in range(int(up_size/p_size)):
            parts[n,:,:,0] = unpartitioned[p_size*u:p_size*u+p_size , p_size*v:p_size*v+p_size]
            n = n + 1
    return parts

def tiling(partitions, rec_size):
    p_size = partitions.shape[1]
    reconstructed_img = np.zeros((rec_size,rec_size))
    n=0
    for u in range(int(rec_size/p_size)):
        for v in range(int(rec_size/p_size)):
            reconstructed_img[p_size*u:p_size*u+p_size , p_size*v:p_size*v+p_size] = partitions[n,:,:,0]
            n = n + 1
    return reconstructed_img

def W_tiling(partitions, rec_size, num_bitplane):
    p_size = partitions.shape[1]
    reconstructed_W = np.zeros((rec_size, rec_size, num_bitplane))
    n=0
    for u in range(int(rec_size/p_size)):
        for v in range(int(rec_size/p_size)):
            reconstructed_W[p_size*u:p_size*u+p_size , p_size*v:p_size*v+p_size, :] = partitions[n,:,:,:]
            n = n + 1
    return reconstructed_W
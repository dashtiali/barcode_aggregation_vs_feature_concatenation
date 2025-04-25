"""
Apply ULBP and get distance matrix
@author: Dashti
"""

import numpy as np
from scipy.spatial import distance

def compute_dist_matrix(image, geometry=3, rotation=0):
    lbp_idx = get_geometry_ulbp(geometry)

    if np.size(lbp_idx) == 0:
        return []
    
    dist_mat = ULBP_top_basis(image, lbp_idx[rotation])

    if dist_mat.size == 0:
        return np.array([])
    
    return dist_mat

# Function to transfer image to LBP domain
def lbp8_image(img):
    n, m = img.shape
    padded_img = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    lbp = np.zeros_like(img, dtype=int)
    
    for i in range(1, n+1):
        for j in range(1, m+1):
           bin_str = ''
           neighbors = [
               padded_img[i-1,j-1], padded_img[i-1,j], padded_img[i-1,j+1], padded_img[i,j+1],
               padded_img[i+1,j+1], padded_img[i+1,j], padded_img[i+1,j-1], padded_img[i,j-1]
           ]
           
           bin_str = ''.join(['1' if padded_img[i,j] > n else '0' for n in neighbors])
           lbp[i-1, j-1] = int(bin_str, 2)
    
    return lbp

# Function to create distance matrix
def ULBP_top_basis(img, lbp_idx):
    # Compute Local Binary Pattern (LBP) of the image
    lbp_img = lbp8_image(img)
    
    binary_idx = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48,
                56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131,
                135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227,
                231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]
    
    n, m = lbp_img.shape
    pts = []

    for i in range(n): 
        for j in range(m): 
            if lbp_img[i, j] == binary_idx[lbp_idx - 1]:
                pts.append([i, j])
    
    dist_mat = distance.squareform(distance.pdist(pts)) if pts else np.array([])
    return dist_mat

# Function to get all rotations of a geometry
def get_geometry_ulbp(geometry_no):
    geometry_mat = [
        [2, 3, 5, 8, 12, 17, 23, 30],
        [4, 6, 9, 13, 18, 24, 31, 37],
        [7, 10, 14, 19, 25, 32, 38, 43],
        [11, 15, 20, 26, 33, 39, 44, 48],
        [16, 21, 27, 34, 40, 45, 49, 52],
        [22, 28, 35, 41, 46, 50, 53, 55],
        [29, 36, 42, 47, 51, 54, 56, 57]
    ]
    return geometry_mat[geometry_no] if geometry_no in range(7) else None

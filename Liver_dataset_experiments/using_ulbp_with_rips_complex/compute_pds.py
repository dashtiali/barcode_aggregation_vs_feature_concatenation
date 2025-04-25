"""
Persistent Diagrams Computation
@author: Dashti
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

import numpy as np
import multiprocessing as mp
import glob
import time
from persistent_diagrams import compute_pds_from_point_cloud
from tqdm import tqdm
from lbp import compute_dist_matrix
from enums import FiltrationTypes

def get_barcods(image_path):
    ulbp_geometry = 3
    ulbp_rotation = 0

    image = np.load(image_path)

    if image.any():
        dist_matrix = compute_dist_matrix(image, ulbp_geometry, ulbp_rotation)
        if dist_matrix.size == 0:
            pds = [[], []]
        else:
            pds = compute_pds_from_point_cloud(dist_matrix, max_dim=1, filtration_type=FiltrationTypes.RIPS_COMPLEX, is_distance_matrix=True)
    else:
        pds = []
        print(f'{os.path.basename(image_path)} : Image is empty!\n')

    return [os.path.basename(image_path), pds]

if __name__ == '__main__':
    start_time = time.time()
    
    main_path = r'../../Datasets/liverTumours'
    classes = ['ICC', 'HCC']
    datasets_path = {name: os.path.join(main_path, name) for name in classes}

    output_folder = 'persistent_diagrams'
    os.makedirs(output_folder, exist_ok=True)

    for dataset_name, dataset_path in tqdm(datasets_path.items(), leave=False):
        args = glob.glob(dataset_path + '\\' + "*.npy", recursive=True)

        pool = mp.Pool()
        results = list(tqdm(pool.imap(get_barcods, args), leave=False, total=len(args), colour='green'))
        pool.close()

        out_data = {i[0]: i[1] for i in results}
        np.save(os.path.join(output_folder, f'{dataset_name}_pds_dict.npy'), out_data)

    elapsed_time = time.time() - start_time
    print(elapsed_time)
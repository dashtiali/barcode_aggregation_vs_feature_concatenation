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
import itertools
from PIL import Image

def get_barcods(image_path):
    ulbp_geometry = [3, 4]
    ulbp_rotation = range(8)
    
    # Load the image
    image = Image.open(image_path).convert('L')
    image = np.array(image)
    base_name = os.path.basename(image_path)
    all_pds = []
    
    if image.any():
        for g in ulbp_geometry:
            for r in ulbp_rotation:
                dist_matrix = compute_dist_matrix(image, g, r)
                pds = compute_pds_from_point_cloud(dist_matrix, max_dim=1, filtration_type=FiltrationTypes.RIPS_COMPLEX, is_distance_matrix=True)
                all_pds.append([f'{base_name}_g{g}_r{r}', pds])
    else:
        pds = []
        print(f'{base_name} : Image is empty!\n')

    return all_pds

if __name__ == '__main__':
    start_time = time.time()
    
    main_path = r'../../Datasets/BUSI'
    classes = ['benign', 'malignant']
    datasets_path = {name: os.path.join(main_path, name) for name in classes}

    output_folder = 'persistent_diagrams'
    os.makedirs(output_folder, exist_ok=True)

    for dataset_name, dataset_path in tqdm(datasets_path.items(), leave=False):
        args = glob.glob(dataset_path + '\\' + "*.jpg", recursive=True)

        pool = mp.Pool()
        results = list(tqdm(pool.imap(get_barcods, args), leave=False, total=len(args), colour='green'))
        pool.close()
        results = list(itertools.chain(*results))

        out_data = {i[0]: i[1] for i in results}
        np.save(os.path.join(output_folder, f'{dataset_name}_pds_dict.npy'), out_data)

    elapsed_time = time.time() - start_time
    print(elapsed_time)
"""
Persistent Barcodes Aggregation
@author: Dashti
"""

import os
import numpy as np
import time
from tqdm import tqdm
import re

def get_patient_ID(file_name):
    match = re.match(r'^(.*?)_slice+', file_name)
    if match:
        return match.group(1)
    else:
        return ''

def aggregate_pds(pds_dict):
    aggr_pds_dict = {}

    for item in pds_dict:
        ct_id = item[0]
        if ct_id not in aggr_pds_dict:
            aggr_pds_dict[ct_id] = [item[1][0], item[1][1]]
        else:
            aggr_pds_dict[ct_id][0] = np.vstack((aggr_pds_dict[ct_id][0], item[1][0]))
            aggr_pds_dict[ct_id][1] = np.vstack((aggr_pds_dict[ct_id][1], item[1][1]))
    
    return aggr_pds_dict

if __name__ == '__main__':
    start_time = time.time()
    
    main_path = r'..\persistent_diagrams'
    datasets = ['partial_nephrectomy', 'radical_nephrectomy']
    
    output_folder = 'aggregated_persistent_diagrams'
    os.makedirs(output_folder, exist_ok=True)

    for dataset_name in tqdm(datasets):
        pds_path = f'{main_path}\\{dataset_name}_pds_dict.npy'
        pds_dict = np.load(pds_path, allow_pickle=True)
        pds_dict = pds_dict.item()
        pds_dict = [[get_patient_ID(file_name), pds] for file_name, pds in pds_dict.items() if len(pds) != 0]
        aggr_pds_dict = aggregate_pds(pds_dict)

        np.save(os.path.join(output_folder, f'{dataset_name}_pds_dict.npy'), aggr_pds_dict)

    elapsed_time = time.time() - start_time
    print(elapsed_time)
"""
Barcode Vectorization
@author: Dashti Ali
"""

import numpy as np
import os
import multiprocessing as mp
import time
import vectorization as vec
from tqdm import tqdm
import re

def Compute_tda_feature(args):
    file_name = args[0][0]
    pds = args[0][1]
    feature_extractor = args[1]

    feature_vectors = []

    for pd in pds:
        if feature_extractor == vec.GetPersLandscapeFeature:
            feature = feature_extractor(pd, res=30, num=4)
        else:
            feature = feature_extractor(pd)
            
        feature_vectors.append(feature)

    return [file_name, feature_vectors]

def get_patient_ID(file_name):
    match = re.match(r'^(.*?)_slice+', file_name)
    if match:
        return match.group(1)
    else:
        return ''

def get_feature_names(feature_vector_name, length, prefix):
    feature_list = {
        'BettiCurve': [f'{prefix}_BettiCurve({i})' for i in range(length)],
        'EntropySummary': [f'{prefix}_EntropySummary({i})' for i in range(length)],
        'PersLandscape': [f'{prefix}_PersLandscape({i})' for i in range(length)],
        'PersStats': [f'{prefix}_' + s for s in ['Births_Mean', 'Births_STD', 'Births_Median', 'Births_IQR', 'Births_Range', 'Births_P10', 
                      'Births_P25', 'Births_P75', 'Births_P90', 'Deaths_Mean', 'Deaths_STD', 'Deaths_Median', 
                      'Deaths_IQR', 'Deaths_Range', 'Deaths_P10', 'Deaths_P25', 'Deaths_P75', 'Deaths_P90', 
                      'Midpoints_Mean', 'Midpoints_STD', 'Midpoints_Median', 'Midpoints_IQR', 'Midpoints_Range', 
                      'Midpoints_P10', 'Midpoints_P25', 'Midpoints_P75', 'Midpoints_P90', 'Lifespans_Mean', 
                      'Lifespans_STD', 'Lifespans_Median', 'Lifespans_IQR', 'Lifespans_Range', 'Lifespans_P10', 
                      'Lifespans_P25', 'Lifespans_P75', 'Lifespans_P90', 'Count', 'Entropy']],
        'PersTropicalCoordinates': [f'{prefix}_PersTropicalCoordinates({i})' for i in range(length)]
        }
    
    feature_name = feature_list[feature_vector_name] if feature_vector_name in feature_list else []
    
    return feature_name

if __name__ == '__main__':
    start_time = time.time()
    datasets = ['partial_nephrectomy', 'radical_nephrectomy']

    os.makedirs('extracted_features')

    with tqdm(datasets) as datasets_tqdm:
        for dataset_name in datasets_tqdm:
            datasets_tqdm.set_description(f'Dataset: {dataset_name}')
            
            pds_path = f'aggregated_persistent_diagrams\\{dataset_name}_pds_dict.npy'
            pds_dict = np.load(pds_path, allow_pickle=True)
            pds_dict = pds_dict.item()
            pds_dict = [[ct_file_name, pds] for ct_file_name, pds in pds_dict.items() if len(pds) != 0]

            feature_list = {
                'BettiCurve': vec.GetBettiCurveFeature,
                'EntropySummary': vec.GetEntropySummary,
                'PersStats': vec.GetPersStats,
                'PersTropicalCoordinates': vec.GetPersTropicalCoordinatesFeature,
                'PersLandscape': vec.GetPersLandscapeFeature
                }

            main_feat_dir = f'extracted_features\\{dataset_name}'
            os.makedirs(main_feat_dir, exist_ok=True)
            
            with tqdm(feature_list, colour='green') as feature_list_tqdm:
                for feature_name in feature_list_tqdm:
                    # print(f'======Extracting feature: {feature_name}======\n')
                    feature_list_tqdm.set_description(f'Extracting feature: {feature_name}')

                    feature_extractor = feature_list[feature_name]
                    args = [[p, feature_extractor] for p in pds_dict]
                    
                    pool = mp.Pool()
                    results = pool.map(Compute_tda_feature, args)
                    pool.close()

                    # Feature matrix for PH_0
                    feature_matrix_dim_0 = np.column_stack((np.array([result[0] for result in results]),
                                                            np.array([result[1][0] for result in results])))
                    
                    dim_0_headers = ['Patient_ID',] + get_feature_names(feature_name, len(results[0][1][0]), 'dim(0)')

                    # Feature matrix for PH_1
                    feature_matrix_dim_1 = np.column_stack((np.array([result[0] for result in results]),
                                                            np.array([result[1][1] for result in results])))
                    
                    dim_1_headers = ['Patient_ID'] + get_feature_names(feature_name, len(results[0][1][1]), 'dim(1)')

                    # Sort the matrices
                    feature_matrix_dim_0 = feature_matrix_dim_0[feature_matrix_dim_0[:, 0].argsort()]
                    feature_matrix_dim_1 = feature_matrix_dim_1[feature_matrix_dim_1[:, 0].argsort()]
                    
                    # Save as CSV
                    np.savetxt(f'{main_feat_dir}\\feature_matrix_dim_0_{feature_name}.csv', feature_matrix_dim_0, delimiter=',', fmt='%s', header=','.join(dim_0_headers), comments='')
                    np.savetxt(f'{main_feat_dir}\\feature_matrix_dim_1_{feature_name}.csv', feature_matrix_dim_1, delimiter=',', fmt='%s', header=','.join(dim_1_headers), comments='')
            
    elapsed_time = time.time() - start_time
    print(elapsed_time)
"""
Features Concatenation
@author: Dashti
"""

import os
import time
from tqdm import tqdm
import pandas as pd

def concatenate_features(df):
    # Drop 'File_name' column
    df = df.drop(columns=['File_name'])

    # Number of items per group (assumes equal items in each group)
    num_items = df.groupby('Patient_ID').size().iloc[0]

    # Group by 'Patient_ID' and concatenate each group as a row vector
    df_grouped = df.groupby('Patient_ID').apply(lambda x: x.drop(columns=['Patient_ID']).values.flatten()).reset_index()
    df_grouped.columns = ['Patient_ID', 'Concatenated_Features']

    # Create new column names with suffixes
    feature_names = df.columns[1:]  # Excluding 'Patient_ID'
    new_column_names = [f"{feature}_{suffix}" for suffix in range(1, num_items + 1) for feature in feature_names]

    # Expand concatenated features into separate columns with the new column names
    concatenated_features = pd.DataFrame(df_grouped['Concatenated_Features'].tolist(), columns=new_column_names)
    df_final = pd.concat([df_grouped[['Patient_ID']], concatenated_features], axis=1)

    # Show the final data frame
    return df_final

if __name__ == '__main__':
    start_time = time.time()
    
    main_path = 'extracted_features'
    datasets = ['ICC', 'HCC']

    output_folder = 'concatenated_features'
    os.makedirs(output_folder, exist_ok=True)

    for dataset in datasets:
        if not os.path.isdir(f'{output_folder}\\{dataset}'):
            os.mkdir(f'{output_folder}\\{dataset}')

    feature_list = ['BettiCurve','EntropySummary','PersStats','PersLandscape','PersTropicalCoordinates']
    ph_dims = [0, 1]

    for dataset_name in tqdm(datasets):
        for feature_name in feature_list:
            for dim in ph_dims:
                feature_path = f'{main_path}\\{dataset_name}\\feature_matrix_dim_{dim}_{feature_name}.csv'
                feature_df = pd.read_csv(feature_path)
                concatenated_feature_df = concatenate_features(feature_df)
                concatenated_feature_path = f'{output_folder}\\{dataset_name}\\feature_matrix_dim_{dim}_{feature_name}.csv'
                concatenated_feature_df.to_csv(concatenated_feature_path, index=False)

    elapsed_time = time.time() - start_time
    print(elapsed_time)
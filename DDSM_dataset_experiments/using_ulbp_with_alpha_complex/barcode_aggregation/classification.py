"""
Classification
@author: Dashti
"""

import pandas as pd
from pycaret.classification import *
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import re
import os
import shutup
from tqdm import tqdm
import random

def aggregate_features(featur_dict):
    feature_agg = []

    for i, feature_name in enumerate(featur_dict):
        for j, dim in enumerate(featur_dict[feature_name]):
            feature_df = pd.read_csv(featur_dict[feature_name][dim])

            # Initialize the aggregated feature data frame with the first two columns
            if i == j == 0:
                feature_agg = pd.DataFrame(feature_df[['Patient_ID']])

            # Merge DataFrames on 'File_name'
            feature_agg = pd.merge(feature_agg, feature_df, on='Patient_ID')
            
    return feature_agg

class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, min_features=1):
        self.min_features = min_features  # Minimum number of features to select
        self.selected_features = []

    def fit(self, X, y):
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y)

        # Select features with non-zero coefficients
        non_zero_features = X.columns[lasso.coef_ != 0].tolist()

        # If fewer than `min_features` are selected, select the top-ranked features
        if len(non_zero_features) < self.min_features:
            # Rank all features by absolute coefficient values
            top_features = np.argsort(-np.abs(lasso.coef_))[:self.min_features]
            self.selected_features = X.columns[top_features].tolist()
        else:
            self.selected_features = non_zero_features

        return self

    def transform(self, X):
        return X[self.selected_features]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features

def save_auc_plot(pycaret_model, file_name):
    plot_model(pycaret_model, plot='auc', save=True, plot_kwargs = {'classes' : get_config('pipeline').steps[0][1].transformer.classes_})

    if os.path.exists('AUC.png'):
        os.rename('AUC.png', file_name)

def save_conf_matrix_plot(pycaret_model, file_name):
    plot_model(pycaret_model, plot='confusion_matrix', save=True, plot_kwargs = {'classes' : get_config('pipeline').steps[0][1].transformer.classes_})

    if os.path.exists('Confusion Matrix.png'):
        os.rename('Confusion Matrix.png', file_name)

def add_space_between_words(string):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', string)

def format_feature_name(feature_name):
    return add_space_between_words(feature_name).replace('Pers', 'Persistent').replace('Stats', 'Statistics')

if __name__ == '__main__':
    shutup.please()

    def seed_everything(seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    rand_seed = 123
    seed_everything(rand_seed)

    main_feat_dir = 'extracted_features'
    output_dir = 'classification_results'
    plots_dir = f'{output_dir}\\plots'
    saved_models_dir = f'{output_dir}\\saved_models'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)

    if not os.path.isdir(saved_models_dir):
        os.mkdir(saved_models_dir)

    feature_sets = [
                        ['BettiCurve'],
                        ['EntropySummary'],
                        ['PersStats'],
                        ['PersLandscape'],
                        ['PersTropicalCoordinates']
                    ]

    classes = {'Normal': 0, 'Abnormal': 1}
    ph_dims = [0, 1]

    metric_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1']

    folds_result_columns = ['Feature Selection Algorithm',
                            'Feature',
                            'Model',
                            'Accuracy ± Std.',
                            'AUC ± Std.',
                            'Recall ± Std.',
                            'Prec. ± Std.',
                            'F1 ± Std.']

    df_fold_results = pd.DataFrame(columns=folds_result_columns)

    test_result_columns = ['Feature Selection Algorithm', 'Feature', 'Model', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1']
    df_test_results = pd.DataFrame(columns=test_result_columns)

    for feature_list in tqdm(feature_sets):
        feature_name = ' - '.join([format_feature_name(f) for f in feature_list])
        print(f'\nProcessing Feature list {feature_name} ....')
        feature_selection_results = f'Feature: {feature_name}\n'

        combined_data = []
        for c in classes.keys():
            feature_dict = {feature_name:
                                {d: os.path.join(main_feat_dir, c, f'feature_matrix_dim_{d}_{feature_name}.csv') for d in ph_dims}
                            for feature_name in feature_list}

            features = aggregate_features(feature_dict)
            features['Class'] = classes[c]
            combined_data.append(features)

        combined_data = pd.concat(combined_data, ignore_index=True)
        # Remove columns with all zero values
        combined_data = combined_data.loc[:, (combined_data != 0).any(axis=0)]
        
        test_size=0.2
        integer_portion = int(combined_data.shape[0] * test_size)
        test_size = integer_portion/combined_data.shape[0]

        # Perform the split
        train_data, test_data = train_test_split(
                                            combined_data, 
                                            test_size=test_size, 
                                            stratify=combined_data['Class'], 
                                            random_state=rand_seed
                                        )
        
        feature_selectors = {'Lasso': LassoFeatureSelector()}

        for algorithms_name, sel_algorithm in feature_selectors.items():

            # Setup PyCaret
            exp1 = setup(data=train_data,
                         test_data=test_data,
                         target='Class',
                         ignore_features=['Patient_ID'],
                         fold=5,
                         train_size=0.8,
                         normalize = True,
                         preprocess=True,
                         normalize_method='zscore',
                         feature_selection=False,
                         custom_pipeline=sel_algorithm,
                         session_id=rand_seed,
                         verbose=False,
                         log_data=False)

            feature_selection_results = f'Feature: {feature_name}\n'
            feature_selection_results += f'Feature selection algorithm: {algorithms_name}\n'
            feature_selection_results += f'Number of selected features: {exp1.get_config("X_train_transformed").shape[1]}\n'
            feature_selection_results += f'Selected features list: [{", ".join(exp1.get_config("X_train_transformed").columns)}]\n\n'

            np.save(f'{output_dir}\\{feature_name}_selected_feature.npy', exp1.get_config("X_train_transformed").columns)

            with open(f"{output_dir}\\feature_selection_results.txt", mode="a") as f:
                f.write(feature_selection_results)

            best_models = compare_models(n_select=3, verbose=False)
            dt_results = pull()
            dt_results.to_csv(f'{output_dir}\\compare_models_{"_".join(feature_list)}.csv', header=True)

            if not isinstance(best_models, list):
                best_models = [best_models]

            tuned_best_models = []
            models_metrics = []

            for model in best_models:
                print(f'Tuning model: {type(model).__name__} ...')
                tuned_model = tune_model(model, optimize = 'Recall', search_library='optuna', verbose=False)
                tuned_best_models.append(tuned_model)
                dt_results = pull()
                models_metrics.append(dt_results)
                save_model(model, f'{saved_models_dir}\\{feature_name}_tuned_{type(model).__name__}_model',)

            model_names = [add_space_between_words(type(model).__name__) for model in tuned_best_models]

            data = []

            for model_name, metrics in zip(model_names, models_metrics):
                formatted_metrics = [algorithms_name, feature_name, model_name]
                mean_values = metrics.loc['Mean', metric_columns]
                std_values = metrics.loc['Std', metric_columns]
                formatted_metrics.extend([f"{mean:.4f} ± {std:.4f}" for mean, std in zip(mean_values, std_values)])
                data.append(formatted_metrics)

            df = pd.DataFrame(data, columns=folds_result_columns)
            df.to_csv(f'{output_dir}\\tuned_models_metrics_{"_".join(feature_list)}_{algorithms_name}.csv', header=True, encoding="cp1252")
            df_fold_results = pd.concat([df_fold_results, df], ignore_index=True)

            # Make predictions on test data
            #######################################

            df = pd.DataFrame(columns=test_result_columns)

            for model_name, model in zip(model_names, tuned_best_models):
                try:
                    save_auc_plot(model, f'{plots_dir}\\{"_".join(feature_list)}_{algorithms_name}_tuned_{model_name}_model_test_data_auc_plot.png')
                    save_conf_matrix_plot(model, f'{plots_dir}\\{"_".join(feature_list)}_{algorithms_name}_tuned_{model_name}_model_test_data_confusion_matrix.png')
                except:
                    print(f'Cannot create all the plots for {model_name} !\n')

                predictions = predict_model(model, verbose=False)
                results = pull()
                results['Feature Selection Algorithm'] = algorithms_name
                results['Feature'] = feature_name
                results['Model'] = model_name

                df = pd.concat([df, results[test_result_columns]], ignore_index=True)


            df.to_csv(f'{output_dir}\\tuned_best_models_test_data_metrics_{"_".join(feature_list)}_{algorithms_name}.csv', header=True)
            df_test_results = pd.concat([df_test_results, df], ignore_index=True)

            #######################################

            print(f'Creating ensemble stack models ...')
            stack_model = stack_models(estimator_list = tuned_best_models.copy(), meta_model = tuned_best_models[0], optimize = 'Recall', verbose=False)

            print(f'Tuning ensemble model ...')
            tuned_stack_model = tune_model(stack_model)
            tuned_stack_model_results = pull()

            formatted_metrics = [algorithms_name, feature_name, 'Ensemble']
            mean_values = tuned_stack_model_results.loc['Mean', metric_columns]
            std_values = tuned_stack_model_results.loc['Std', metric_columns]
            formatted_metrics.extend([f"{mean:.4f} ± {std:.2f}" for mean, std in zip(mean_values, std_values)])

            df = pd.DataFrame([formatted_metrics], columns=folds_result_columns)
            df.to_csv(f'{output_dir}\\tuned_ensemble_model_avg_5_fold_metrics_{"_".join(feature_list)}_{algorithms_name}.csv', header=True, encoding="cp1252")
            df_fold_results = pd.concat([df_fold_results, df], ignore_index=True)

            # Make predictions on test data
            #######################################

            try:
                save_auc_plot(tuned_stack_model, f'{plots_dir}\\{"_".join(feature_list)}_{algorithms_name}_tuned_ensemble_model_test_data_auc_plot.png')
                save_conf_matrix_plot(tuned_stack_model, f'{plots_dir}\\{"_".join(feature_list)}_{algorithms_name}_tuned_ensemble_model_test_data_confusion_matrix.png')
            except:
                print(f'Cannot create all the plots for ensemble model!\n')

            predictions = predict_model(tuned_stack_model)
            tuned_stack_model_results = pull()

            tuned_stack_model_results['Feature Selection Algorithm'] = algorithms_name
            tuned_stack_model_results['Feature'] = feature_name
            tuned_stack_model_results['Model'] = 'Ensemble'

            df = tuned_stack_model_results[test_result_columns]
            df.to_csv(f'{output_dir}\\tuned_ensemble_model_test_data_metrics_{"_".join(feature_list)}_{algorithms_name}.csv', header=True)


            df_test_results = pd.concat([df_test_results, df], ignore_index=True)
            save_model(tuned_stack_model, f'{saved_models_dir}\\{feature_name}_tuned_ensemble_model',)

            #######################################

    df_fold_results.to_csv(f'{output_dir}\\all_fold_results.csv', header=True, encoding="cp1252")
    df_test_results.to_csv(f'{output_dir}\\all_test_results.csv', header=True, encoding="cp1252")

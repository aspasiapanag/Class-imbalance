import argparse
import pandas as pd
import numpy as np
import biogeme.database as db
from sklearn.model_selection import StratifiedShuffleSplit
from utils import      hyperparameter_tuning_RF, create_model_RF,\
                       SMOTENC_preprocessing, NBU_preprocessing, get_overlapping_samples,\
                       simulation_MNL_model, define_utility_functions, train_MNL_model,\
                       get_samples_with_missing_values, merge_samples_with_and_without_missing_values,\
                       set_availability, get_samples_without_missing_values, get_confusion_matrix, append_metrics, \
                       append_metrics_separation_scheme_binary, append_metrics_separation_scheme_multiclass, store_results

from load_data import load_data_RF, load_data_MNL

import warnings
warnings.filterwarnings('ignore')



def get_args_parser():
    parser = argparse.ArgumentParser(add_help = False)

    
    parser.add_argument('--path_data', type = str, default = './data/',\
                        help = 'Please specify the path to the data files.')
  
    
    parser.add_argument('--filename_train_RF', type = str, default = 'data_train_RF.csv',\
                        help = 'Please provide the name of the training data file for the RF model.')
    
    parser.add_argument('--filename_test_RF', type = str, default = 'data_test_RF.csv',\
                        help = 'Please provide the name of the test data file for the RF model.')
    
    parser.add_argument('--filename_val_RF', type = str, default = 'data_val_RF.csv',\
                        help = 'Please provide the name of the validation data file for the RF model.')
    
    parser.add_argument('--filename_MNL', type = str, default = 'data_MNL.csv',\
                        help = 'Please provide the name of the data file for the MNL model.')

    parser.add_argument('--technique', type = str, default = 'Baseline',\
                        choices = ['Baseline','SMOTENC','NBU','SeparationScheme'],\
                        help = 'Please choose a technique for the data preprocessing. In case of choosing Baseline\
                        no preprocessing technique is implemented.') 
    
    parser.add_argument('--model', type = str, default = 'RandomForestClassifier',\
                         choices = ['RandomForestClassifier', 'MNL'], help = 'Please specify the model.') 
    
    parser.add_argument('--classification_type', type = str, default = 'Multiclass',\
                         choices = ['Binary', 'Multiclass'], help = 'Please specify the type of classification.')

    parser.add_argument('--rounds', type = int, default = 5, help = 'Please specify the number of times to run the model during \
                        its training phase.')

    
    parser.add_argument('--random_state', type = int, default = 42, help = 'Please specify the parameter to control the algorithmic randomization.')

    parser.add_argument('--smotenc_neighbors', type = int, default = 3 , help = 'Please specify the number of the nearest neighbors\
                                                                            to be considered when creating synthetic data from the\
                                                                            minority class.')
     
    parser.add_argument('--percentage_smotenc', type = int, default = 0.3 , help = 'Please indicate the percentage of minority samples to be generated relative \
                                                                                    to the number of majority samples.')   

    parser.add_argument('--nbu_neighbors', type = int, default = 3 , help = 'Please specify the number of the nearest neighbors\
                                                                            to be considered when eliminating data from the majority class.')

    parser.add_argument('--separation_scheme_neighbors', type = int, default = 3 ,\
                         help = 'Please specify the number of the nearest neighbors\
                                 to be considered when defining the overlapping region.')

    parser.add_argument('--n_estimators', default = [100, 250], help = 'Please specify the \
                         the number of tree estimators to be used in hyperparameter tuning.') 
    
    parser.add_argument('--max_depth', default = [None, 20],help = 'Please specify the maximum depth of\
                        tree estimators to be used in hyperparameter tuning.') 

    parser.add_argument('--min_samples_leaf', default = [1, 5], help = 'Please specify the minimum number of \
                        samples required at each leaf node to be used in hyperparameter tuning.') 

    parser.add_argument('--max_samples', default = [None, 0.5], help = 'Please specify the number of \
                        samples utilized for constructing each tree estimator to be used in hyperparameter tuning.')

    parser.add_argument('--max_features', default = ['sqrt', None], help = 'Please specify the \
                        number of features to be considered when looking for the best split to be used in hyperparameter tuning.')

    parser.add_argument('--pretrained_argument', type = int, default = np.nan, help = 'Please specify whether the \
                         the pretrained model is used or not.') 
    
    return parser 




def main_imbalance(args):

    if args.model == 'RandomForestClassifier':

        data_train, data_test, data_val = load_data_RF(args.path_data, args.filename_train_RF, args.filename_test_RF,\
                                        args.filename_val_RF, args.classification_type)
    else:
        data_MNL = load_data_MNL(args.path_data, args.filename_MNL, args.classification_type, args.technique)


    if args.model == 'RandomForestClassifier':
    
        if args.technique == 'SeparationScheme': 
            
            data_train, data_test, data_val, overlapping_region_indices = get_overlapping_samples (data_train, data_test, data_val, args.separation_scheme_neighbors)

            # For simplicity, hyperparameter tuning is performed using the entire dataset instead of individually tuning each model for the overlapping
            # and non-overlapping regions.
            estimators, max_depth, min_samples_leaf, max_samples, max_features = hyperparameter_tuning_RF(args.n_estimators,\
                                                                            args.max_depth, args. min_samples_leaf,\
                                                                            args.max_features, args.max_samples,\
                                                                            data_train, data_val)
            
            for i in range(args.rounds): # run multiple times to account for the model's stochasticity

                # samples that belong to the overlapping region get the label 2, while the rest get the label 1
                data_first_stage = data_train.copy()
                data_first_stage.loc[list(overlapping_region_indices),'overlap'] = 2 
                data_first_stage.loc[data_first_stage.overlap != 2, 'overlap'] = 1

                model_first_stage = create_model_RF (estimators, max_depth, min_samples_leaf, max_samples, max_features)
                model_first_stage.fit(data_first_stage.loc[:, [x for x in data_first_stage.columns if x not in list(('khvm','overlap'))]], data_first_stage['overlap'])

                # in the first stage the model predicts whether a sample belongs to the overlapping or the non-overlapping region
                y_pred_first_stage = model_first_stage.predict(data_test.loc[:, (data_test.columns != 'khvm')])

                confusion_matrix_first_stage = get_confusion_matrix(data_test,y_pred_first_stage)

                # data predicted to belong to the overlapping region pass to the second stage
                y_pred_first_stage = pd.DataFrame(y_pred_first_stage, columns=['overlap'])
                ind = y_pred_first_stage.loc[y_pred_first_stage.overlap == 2].index
                data_second_stage = data_test.loc[ind,:]

                # train a model in the overlapping region
                data_train_overlapping_region = data_train.loc[list(overlapping_region_indices),:]
                model_overlapping_region = create_model_RF (estimators, max_depth, min_samples_leaf, max_samples, max_features)
                model_overlapping_region.fit(data_train_overlapping_region.loc[:, (data_train_overlapping_region.columns != 'khvm')], data_train_overlapping_region.khvm )

                # in the second stage the model predicts the class of all samples within the overlapping region
                y_pred_overlapping_region = model_overlapping_region.predict(data_second_stage.loc[:, (data_second_stage.columns != 'khvm')])
                
                confusion_matrix_overlapping_region = get_confusion_matrix(data_second_stage, y_pred_overlapping_region)
                
                if args.classification_type == 'Binary': 
                    recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, total_accuracy,\
                    balanced_accuracy =  append_metrics_separation_scheme_binary(confusion_matrix_first_stage, confusion_matrix_overlapping_region, i)

                else:

                    data_non_overlapping_region = data_train.copy().drop(list(overlapping_region_indices))

                    # train a model in the non-overlapping region
                    model_non_overlapping_region = create_model_RF (estimators, max_depth, min_samples_leaf, max_samples, max_features)
                    model_non_overlapping_region.fit(data_non_overlapping_region.loc[:, (data_non_overlapping_region.columns != 'khvm')], data_non_overlapping_region.khvm)

                    data_third_stage = data_test.loc[y_pred_first_stage.loc[(y_pred_first_stage.overlap == 1)].index,:]

                    # in the third stage the model predicts the class of the samples belonging to the non-overlapping region
                    y_pred_non_overlapping_region = model_non_overlapping_region.predict(data_third_stage.loc[:, (data_third_stage.columns != 'khvm')])

                    confusion_matrix_non_overlapping_region = get_confusion_matrix(data_third_stage, y_pred_non_overlapping_region)

                    recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, recall_bike,\
                    precision_bike, f1_score_bike, total_accuracy,balanced_accuracy = \
                    append_metrics_separation_scheme_multiclass(confusion_matrix_first_stage, confusion_matrix_overlapping_region, confusion_matrix_non_overlapping_region, i)

            
            # save the models for each stage
            store_results(args.classification_type, args.technique, args.model)
            
       
        else: 

            if args.technique == 'NBU':
    
                data_train = NBU_preprocessing (data_train, args.nbu_neighbors, args.model)
                data_val = pd.get_dummies(data = data_val, columns = ['sted_o','sted_d','hhgestinkg','leeftijd','gemgr'], dtype = int, drop_first = False)
                data_test = pd.get_dummies(data = data_test, columns = ['sted_o','sted_d','hhgestinkg','leeftijd','gemgr'], dtype = int, drop_first = False)
        
            elif args.technique == 'SMOTENC': 
        
                data_train = SMOTENC_preprocessing(data_train, args.percentage_smotenc, args.random_state, args.smotenc_neighbors, args.model)
                
        
            estimators, max_depth, min_samples_leaf, max_samples, max_features = hyperparameter_tuning_RF(args.n_estimators,\
                                                                args.max_depth, args. min_samples_leaf,\
                                                                args.max_features, args.max_samples,\
                                                                data_train, data_val)
        
    
            model = create_model_RF(estimators, max_depth, min_samples_leaf, max_samples, max_features)

            for i in range(args.rounds):
            
                model.fit(data_train.loc[:, (data_train.columns != 'khvm')], data_train.khvm)
                y_pred = model.predict(data_test.loc[:, (data_test.columns != 'khvm')])

                confusion_matrix = get_confusion_matrix(data_test, y_pred)

                if args.classification_type == 'Binary':
                    recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, total_accuracy,\
                    balanced_accuracy = append_metrics(confusion_matrix, args.classification_type, data_test.khvm, y_pred, i)

                else:

                    recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, recall_bike,\
                    precision_bike, f1_score_bike, total_accuracy,balanced_accuracy = append_metrics(confusion_matrix, args.classification_type, data_test.khvm, y_pred, i)

            store_results(args.classification_type, args.technique, args.model)
        
    elif args.model == 'MNL':

        if args.technique == 'SeparationScheme': # the combination of the Separation scheme technique with the MNL model is not implemented in this project
            print('MNL model cannot run in combination with the SeparationScheme technique')
        
        else:
            if args.technique == 'Baseline':  
                data_MNL.fillna(99999, inplace = True) # biogeme considers variables with this value as missing

            if args.technique == 'NBU': # conduct one-hot encoding so that features can be utilized in the Euclidean distance calculations
                dummies = ['leeftijd','herkomst', 'sted_o','sted_d', 'kmotiefv','season','vertprov','period_o','gemgr','hhgestinkg']
                for dummy in dummies:
                    data_MNL = pd.get_dummies(data = data_MNL, columns = [dummy], dtype = int, drop_first = False)
                    

            data_MNL = data_MNL.reset_index() 
            data_MNL.drop(columns = 'index', inplace = True) #

            data_MNL_x  = data_MNL.loc[:, data_MNL.columns != 'khvm'] 
            data_MNL_y = pd.DataFrame(data = data_MNL.khvm, columns = ['khvm']) 

            # adopt a stratified splitting approach to ensure the preservation of the target class ratio across all sets
            sss = StratifiedShuffleSplit(n_splits=3, test_size=0.33, random_state=0) 

            for i, (train_index, test_index) in enumerate(sss.split(data_MNL_x, data_MNL_y)):  # perform cross validation

                data_x_train, data_y_train = data_MNL_x.loc[train_index,:], data_MNL_y.loc[train_index] 
                data_x_test, data_y_test = data_MNL_x.loc[test_index,:], data_MNL_y.loc[test_index]     
                    
                if args.technique == 'Baseline': 
                    data_train_final = data_x_train.join(data_y_train)
                    data_test_final = data_x_test.join(data_y_test)
                        
                    database_train = db.Database('database_train', data_train_final) 
                    

                else: 
                    data_train = data_x_train.join(data_y_train) 
                    data_test_final = data_x_test.join(data_y_test) 
                    data_test_final.fillna(99999, inplace = True)


                    # when generating minority samples or eliminating majority samples, only samples without missing values are considered, 
                    # as those with missing values cannot be utilized in the distance metric calculations.
                    if args.technique == 'SMOTENC':
                        data_train_without_missing_values = SMOTENC_preprocessing(data_train, args.percentage_smotenc, args.random_state, args.smotenc_neighbors, args.model)
                    elif args.technique == 'NBU':
                        
                        data_train_without_missing_values_x, data_train_without_missing_values_y = get_samples_without_missing_values(data_x_train, data_y_train)
                        data_train_without_missing_values = data_train_without_missing_values_x.join(data_train_without_missing_values_y)
                        data_train_without_missing_values = NBU_preprocessing (data_train_without_missing_values, args.nbu_neighbors, args.model)
                        
                    data_x_train_without_missing_values = data_train_without_missing_values.loc[:, data_train_without_missing_values.columns != 'khvm']
                    data_y_train_without_missing_values = data_train_without_missing_values.khvm

                    data_x_train_missing_values, data_y_train_missing_values = get_samples_with_missing_values(data_train.loc[:, data_train.columns != 'khvm'], data_train.khvm)

                    data_x_train_final, data_y_train_final = merge_samples_with_and_without_missing_values(data_x_train_missing_values, data_y_train_missing_values,\
                                                data_x_train_without_missing_values, data_y_train_without_missing_values)

                    data_x_train_final = set_availability(data_x_train_final)
                    data_x_train_final.fillna(99999, inplace = True) 
                    data_x_train_final = data_x_train_final.reset_index()
                    data_x_train_final.drop(columns = 'index', inplace = True)

                    
                    data_train_final = data_x_train_final.join(data_y_train_final) 
                    data_train_final= data_train_final[['car_dur','car_cost','pt_dur','transit_cost','khvm', 'av_car',\
                                                           'bike_dur','av_pt','av_bike']]
                    database_train = db.Database('data_train_final', data_train_final)

                V, av = define_utility_functions(args.classification_type)

                results = train_MNL_model(V, av, i, database_train, args.classification_type)

                simulation_MNL_model(V, av, results, args.classification_type, data_test_final, i)
                
            store_results(args.classification_type, args.technique, args.model)

       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main_imbalance(args)
                
                 
    
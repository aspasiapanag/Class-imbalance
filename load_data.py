import pandas as pd
from utils import get_class_label

def load_data_RF(path_data, filename_train_RF, filename_test_RF, filename_val_RF, classification_type):

    '''
    Get the data utilized in the Random Forest model.

    Args:
    path_data (str): The path of the data files.
    filename_train_RF (str): The name of the training dataset.
    filename_test_RF (str): The name of the test dataset.
    filename_val_RF (str): The name of the validation dataset.
    classification_type (str): The  type of classification (binary or multiclass).

    Returns:
    tuple: A tuple containing the following elements:
           - pd.DataFrame: The training dataset utilized in the RF model.
           - pd.DataFrame: The test dataset utilized in the RF model.
           - pd.DataFrame: The validation dataset utilized in the RF model.

    '''

    data_train = pd.read_csv(path_data + filename_train_RF, index_col=0)
    data_test = pd.read_csv(path_data + filename_test_RF, index_col=0)
    data_val = pd.read_csv(path_data + filename_val_RF, index_col=0)

    if classification_type == 'Binary': 

        data_train = data_train.loc[(data_train.khvm == get_class_label('car')) | (data_train.khvm == get_class_label('transit')),:]
        data_test = data_test.loc[(data_test.khvm == get_class_label('car')) | (data_test.khvm == get_class_label('transit')),:]
        data_val = data_val.loc[(data_val.khvm == get_class_label('car')) | (data_val.khvm == get_class_label('transit')),:]

        # remove unrelated features for the binary classification
        data_train.drop(columns = ['bike_dur'], inplace = True)
        data_val.drop(columns = ['bike_dur'], inplace = True)
        data_test.drop(columns = ['bike_dur'], inplace = True)

    else: 
        data_train = data_train.loc[(data_train.khvm == get_class_label('car')) | (data_train.khvm == get_class_label('transit')) | (data_train.khvm == get_class_label('bike')),:]
        data_test = data_test.loc[(data_test.khvm == get_class_label('car')) | (data_test.khvm == get_class_label('transit')) | (data_test.khvm == get_class_label('bike')),:]
        data_val = data_val.loc[(data_val.khvm == get_class_label('car')) | (data_val.khvm == get_class_label('transit')) | (data_val.khvm == get_class_label('bike')),:]
        
    return (data_train, data_test, data_val)
    

def load_data_MNL(path_data, filename_MNL, classification_type, technique):

    '''
    Get the data utilized in the MNL model.

    Args:
    path_data (str): The path of the dataset.
    filename_MNL (str): The name of the dataset.
    classification_type (str): The type of classification (binary or multiclass).
    technique (str): The chosen techique (Baseline/ SMOTENC / NBU / Separation scheme).

    Returns:
    pd.Dataframe: The dataset utilized in the MNL model.

    '''

    data_MNL = pd.read_csv(path_data + filename_MNL, index_col=0)
  
    if technique == "Baseline":
        data_MNL = data_MNL[['car_dur','car_cost','pt_dur','transit_cost','khvm', 'av_car',\
                                                           'bike_dur','av_pt','av_bike']]
    
    if classification_type == 'Binary':
        data_MNL.drop(columns = ['bike_dur', 'av_bike'])
        data_MNL = data_MNL.loc[(data_MNL.khvm == 1) | (data_MNL.khvm == 3),:]
    else:
        data_MNL = data_MNL.loc[(data_MNL.khvm == 1) | (data_MNL.khvm == 3) | (data_MNL.khvm == 4),:]

    return (data_MNL)

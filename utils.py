import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.neighbors import NearestNeighbors
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta 
import biogeme.database as db
from biogeme.expressions import Variable
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pandas as pd
import statistics
import numpy as np
import json
from datetime import datetime


# create lists to store evaluation results
ASC_car = []
ASC_transit = []
B_COST_car = []
B_COST_transit = []
B_TIME_car = [] 
B_TIME_transit = []
B_TIME_bike = []
VOT_car = []
VOT_transit = []
Final_log_likelihood = []

precision_car = []
recall_car = []
f1_score_car = []

precision_bike = []
recall_bike = []
f1_score_bike = []

precision_transit = []
recall_transit = []
f1_score_transit = []

total_accuracy = []
balanced_accuracy = []


def get_class_label(target_class):

    '''
    Get the label of each class in the dataset.
    
    Args:
    target class (str): The class label.

    Returns:
	int: The number corresponding to each class label.

    '''
    
    if target_class == 'car':
        return 1
    elif target_class == 'transit':
        return 3
    elif target_class == 'bike':
        return 4



def hyperparameter_tuning_RF (n_estimators, max_depth, min_samples_leaf,\
                            max_features, max_samples, data_train, data_val):
    
    '''
    Hyperparameter tuning of the Random Forest model.

    Args:
    n_estimators (int): The number of tree estimators.
    max_depth (int); The maximum depth of tree estimators.
    min_samples_leaf (int or float): The minimum number of samples required at each leaf node.
    max_samples (int or float): The number of samples utilized for constructing each estimator.
    max_features (int or float): The number of features to be considered when looking for the best split.
    data_train (pd.DataFrame): The training dataset.
    data_val (pd.DataFrame): The validation dataset.

    Returns:
    tuple: A tuple containing the following elements.
           - int: The optimized number of tree estimators.
           - int: The optimized depth of tree estimators.
           - int or float: The optimized number of samples at each leaf node.
           - int or float : The optimized number of samples utilized for constructing each estimator. 
           - int or float: The optimized number of features to be considered for the best split.

    '''
    accuracy_max = 0
    
    for estimators in n_estimators:
        for depth in max_depth:
            for min_samples in min_samples_leaf:
                for n_feat in max_features:
                    for n_samples in max_samples:

                        rf = RandomForestClassifier(n_estimators = estimators, max_depth = depth,\
                                            min_samples_leaf = min_samples, max_samples = n_samples, \
                                            max_features = n_feat)
                        
                        rf.fit(data_train.loc[:, data_train.columns != 'khvm'], data_train.khvm)
                        y_pred = rf.predict(data_val.loc[:, data_val.columns != 'khvm'])
                        accuracy = accuracy_score(data_val.khvm, y_pred)
                    
                        if accuracy > accuracy_max:
                            accuracy_max = accuracy
                            estimators_tuned = estimators
                            max_depth_tuned = depth
                            min_samples_leaf_tuned = min_samples
                            max_samples_tuned = n_samples
                            max_features_tuned = n_feat

    print(f'Hyperparameter tuning is completed. Best hyperparameter values: tree_estimators = {estimators_tuned}, max_depth = {max_depth_tuned},\
          min_samples_leaf = {min_samples_leaf_tuned}, max_samples = {max_samples_tuned}, max_features = {max_features_tuned}. \
          Maximum accuracy achieved : {round(accuracy_max,4)}.')
          
    

    return (estimators_tuned, max_depth_tuned, min_samples_leaf_tuned, max_samples_tuned, max_features_tuned)


             
def create_model_RF (estimators, max_depth, min_samples_leaf, max_samples, max_features):

    '''
    Create a Random forest model.

    Args:
    estimators (int):  The number of tree estimators.
    max_depth (int): The maximum depth of tree estimators.
    min_samples_leaf (int or float): The number of samples required to be at each leaf node.
    max_samples(int or float): The number of samples used to construct each tree estimator.
    max_features (int or float): The number of features to be considered when looking for the best split.
    
    Returns:
    RandomForestClassifier: The initialized RandomForestClassifier model.
    '''
    model = RandomForestClassifier(n_estimators = estimators, max_depth = max_depth,\
                                            min_samples_leaf = min_samples_leaf, max_samples = max_samples, \
                                            max_features = max_features)
    return model


    
def SMOTENC_preprocessing(data_train, percentage, random_state, smotenc_neighbors, model):

    '''
        Create synthetic samples from the minority class.

        Args:
        data_train (pd.DataFrame): The training dataset.
        percentage (float): The percentage of minority samples to be created relative to the majority samples.
        random_state (int) : The seed used by the random number generator.
        smotenc_neighbors (int): The number of nearest neighbors to be considered when creating the synthetic samples.
        model (str): The chosen model (RF/MNL).

        Returns:
        pd.DataFrame: The training dataset after the implementation of the SMOTENC technique.
        
    '''
    
    categ_features = ['hhpers', 'hhsam', 'gemgr', 'geslacht', 'leeftijd', 'betwerk',
    'opleiding', 'hhgestinkg', 'oprijbewijsau', 'hhauto', 'hhefiets',
    'ovstkaart', 'feestdag','weekdag', 'sted_o', 'sted_d','herkomst',
    'kmotiefv', 'period_o','season', 'vertprov','peak_hour']

    # The one-hot encoded data needs to be reverted to its original state to prevent scenarios
    # such as new samples having zero values across all categories.
    if model == 'RandomForestClassifier':
        data_train.loc[data_train.herkomst_2 == 1,'herkomst'] = 2
        data_train.loc[data_train.herkomst_3 == 1,'herkomst'] = 3
        data_train.loc[((data_train.herkomst_2 != 1) & (data_train.herkomst_3 != 1)),'herkomst'] = 1

        data_train.loc[data_train.kmotiefv_2 == 1,'kmotiefv'] = 2
        data_train.loc[data_train.kmotiefv_3 == 1,'kmotiefv'] = 3
        data_train.loc[((data_train.kmotiefv_2 != 1) & (data_train.kmotiefv_3 != 1)),'kmotiefv'] = 1

        data_train.loc[data_train.period_o_2 == 1,'period_o'] = 2
        data_train.loc[data_train.period_o_3 == 1,'period_o'] = 3
        data_train.loc[((data_train.period_o_2 != 1) & (data_train.period_o_3 != 1)),'period_o'] = 1

        data_train.loc[data_train.season_2 == 1,'season'] = 2
        data_train.loc[data_train.season_3 == 1,'season'] = 3
        data_train.loc[data_train.season_4 == 1,'season'] = 4
        data_train.loc[((data_train.season_2 != 1) & (data_train.season_3 != 1) & (data_train.season_4 != 1)),'season'] = 1

        data_train.loc[data_train.vertprov_2 == 1,'vertprov'] = 2
        data_train.loc[data_train.vertprov_3 == 1,'vertprov'] = 3
        data_train.loc[data_train.vertprov_4 == 1,'vertprov'] = 4
        data_train.loc[data_train.vertprov_5 == 1,'vertprov'] = 5
        data_train.loc[data_train.vertprov_6 == 1,'vertprov'] = 6
        data_train.loc[data_train.vertprov_7 == 1,'vertprov'] = 7
        data_train.loc[data_train.vertprov_8 == 1,'vertprov'] = 8
        data_train.loc[data_train.vertprov_9 == 1,'vertprov'] = 9
        data_train.loc[data_train.vertprov_10 == 1,'vertprov'] = 10
        data_train.loc[data_train.vertprov_11 == 1,'vertprov'] = 11
        data_train.loc[data_train.vertprov_12 == 1,'vertprov'] = 12

        data_train.loc[((data_train.vertprov_2 != 1) & (data_train.vertprov_3 != 1) & (data_train.vertprov_4 != 1) &\
                        (data_train.vertprov_5 != 1) & (data_train.vertprov_6 != 1) & (data_train.vertprov_7 != 1) &
                        (data_train.vertprov_8 != 1) & (data_train.vertprov_9 != 1) & (data_train.vertprov_10 != 1) &
                        (data_train.vertprov_11 != 1) & (data_train.vertprov_12 != 1)),'vertprov'] = 1

        data_train.drop(columns = [ 'herkomst_2',
            'herkomst_3', 'kmotiefv_2', 'kmotiefv_3', 'period_o_2', 'period_o_3',
            'season_2', 'season_3', 'season_4', 'vertprov_2', 'vertprov_3',
            'vertprov_4', 'vertprov_5', 'vertprov_6', 'vertprov_7', 'vertprov_8',
            'vertprov_9', 'vertprov_10', 'vertprov_11', 'vertprov_12'], inplace = True)
    
 
    if model == 'RandomForestClassifier':

        num_samples = round(len(data_train.khvm.loc[data_train.khvm == get_class_label('car')]) * percentage)

        sm = SMOTENC(sampling_strategy = {get_class_label('transit'): num_samples}, random_state = random_state, \
            categorical_features = categ_features, k_neighbors = smotenc_neighbors)
        
      
        X_res, y_res = sm.fit_resample(data_train.loc[:, (data_train.columns != 'khvm')], data_train.khvm)

        data_train = X_res.join(y_res)

        data_train = pd.get_dummies(data = data_train, columns = ['herkomst','kmotiefv','period_o','season','vertprov'], drop_first = True)
        data_train.columns = [col.split('.')[0] for col in data_train.columns]

     
    else:
        
        data_x_train_without_missing_values, data_y_train_without_missing_values = get_samples_without_missing_values(data_train.loc[:, data_train.columns != 'khvm'], data_train.khvm)

        num_samples_required = round(data_train.khvm.value_counts()[1] * percentage) 
        missing_minority_samples = num_samples_required - (data_train.khvm.value_counts()[3] - data_y_train_without_missing_values.value_counts()[3])
        num_samples = missing_minority_samples

        sm = SMOTENC(sampling_strategy = {get_class_label('transit'): num_samples}, random_state = random_state, \
            categorical_features = categ_features, k_neighbors = smotenc_neighbors)

        X_res, y_res = sm.fit_resample(data_x_train_without_missing_values, data_y_train_without_missing_values)
        data_train = X_res.join(y_res)

        data_train = pd.get_dummies(data = data_train, columns = ['herkomst','kmotiefv','period_o','season','vertprov'], drop_first = True)

    return (data_train)


def NBU_preprocessing (data_train, nbu_neighbors, model): 

    '''
    Eliminate samples from classes other than the minority class.

    Args: 
    data_train (pd.DataFrame): The training dataset.
    nbu_neighbors (int): The number of nearest neighbors to be considered for each sample not belonging to the minority class.

    Returns:
    pd.DataFrame: The training dataset after the implementation of the NBU technique.

    '''

    train_x = data_train.reset_index() 
    train_x.drop(columns = 'index', inplace = True)

    if model == 'RandomForestClassifier':
        # perform one-hot encoding so that features can be used in euclidean distance calculations
        train_x = pd.get_dummies(data = train_x, columns = ['sted_o','sted_d','hhgestinkg','leeftijd','gemgr'], dtype = int, drop_first = False)
        

    neigh = NearestNeighbors(n_neighbors = nbu_neighbors) 
    neigh.fit(train_x.loc[:, train_x.columns != 'khvm'])  

    neighbors = neigh.kneighbors(train_x.loc[:, train_x.columns != 'khvm'], return_distance = False)

    indices_remove = [] 

   
    for index, array in enumerate(neighbors): 

        s = False
        if (train_x.khvm.iloc[index] == get_class_label('car')) |(train_x.khvm.iloc[index] == get_class_label('bike')):
           
            for i in array: 
                if train_x.khvm.iloc[index] == get_class_label('transit'): 
                    s = True
            if s ==  True: 
                indices_remove.append(index) 
                
    train_x = train_x.drop(indices_remove) 

    return train_x


def get_overlapping_samples (data_train, data_test, data_val, separation_scheme_neighbors):

    '''
    Return the indices of overlapping samples, which belong to different classes but exhibit similar or identical feature values. 
    Overlapping samples encompass all minority samples and their nearest neighbors belonging to classes other than the minority class.
    
    Args:
    data_train (pd.DataFrame): The training dataset.
    data_test (pd.DataFrame): The test dataset.
    data_val (pd.DataFrame): The validation dataset.
    separation_scheme_neighbors: The number of nearest neighbors to be considered when defining the overlapping region.

    Returns: 
    tuple: A tuple containing the following elements:
           -pd.DataFrame: The training dataset with reset indices.
           -pd.DataFrame: The test dataset with reset indices.
           -set: A set containing the indices of the samples belonging to the overlapping region.

    '''

    #perform one-hot encoding so that features can be utilized in the euclidean distance calculations
    data_train = pd.get_dummies(data = data_train, columns = ['sted_o','sted_d','hhgestinkg','leeftijd','gemgr'], dtype = int, drop_first = False)
    data_test =  pd.get_dummies(data = data_test, columns = ['sted_o','sted_d','hhgestinkg','leeftijd','gemgr'], dtype = int, drop_first = False)
    data_val =  pd.get_dummies(data = data_val, columns = ['sted_o','sted_d','hhgestinkg','leeftijd','gemgr'], dtype = int, drop_first = False)

   
    data_train = data_train.reset_index()
    data_train.drop(columns = 'index', inplace = True)

    data_test = data_test.reset_index()
    data_test.drop(columns = 'index', inplace = True)

    # define the overlapping region
    neigh = NearestNeighbors(n_neighbors = separation_scheme_neighbors)
    neigh.fit(data_train.loc[:, data_train.columns != 'khvm'])

    neighbors = neigh.kneighbors(data_train.loc[:, data_train.columns != 'khvm'], return_distance = False)
    
    # get the indices of the samples belonging to the overlapping region

    overlap_positive_index_train = set() # positive refers to the minority class
    overlap_negative_index_train = set() # negative refers to the majority class

    for index, array in enumerate(neighbors):
        if data_train['khvm'][index] == get_class_label('transit'): 
            overlap_positive_index_train.add(index)  
            for i in array: 
                if data_train['khvm'][i] != get_class_label('transit'): 
                    overlap_negative_index_train.add(i) 

  
    overlap_region_indices = overlap_positive_index_train.union(overlap_negative_index_train)

    
    return (data_train, data_test, data_val, overlap_region_indices)



def get_samples_without_missing_values(train_x, train_y):
        
        '''
        Get the training samples that do not contain any missing values.

        Args:
        train_x (pd.DataFrame): The training dataset (without the target variable).
        train_y (pd.Series): A series containing the target variable.

        Returns:
        tuple: A tuple containing the following elements:
               - pd.DataFrame: The training dataset (without the target variable) excluding samples with NaN values.
               - pd.Series: A series containing the target variable of samples without NaN values.

        '''
        data_x_train_without_missing_values = train_x[~train_x.isnull().any(axis=1)] 
        data_x_train_without_missing_values.drop(columns = ['av_car','av_pt'], inplace = True) 
        indices_without_missing_values = data_x_train_without_missing_values.index 
        data_y_train_without_missing_values = train_y.loc[indices_without_missing_values]

        data_x_train_without_missing_values = data_x_train_without_missing_values.reset_index() 
        data_x_train_without_missing_values.drop(columns = 'index', inplace = True) 

        data_y_train_without_missing_values = data_y_train_without_missing_values.reset_index() 
        data_y_train_without_missing_values.drop(columns = 'index', inplace = True)
        
        return (data_x_train_without_missing_values, data_y_train_without_missing_values)


def get_samples_with_missing_values(train_x, train_y):
        
        '''
        Get the training samples that contain missing values.

        Args:
        train_x (pd.DataFrame): The training dataset (without the target variable).
        train_y (pd.Series): A series containing the target variable.

        Returns:
        tuple: A tuple containing the following elements:
               - pd.DataFrame: The training dataset (without the target variable) including only samples with NaN values.
               - pd.Series: A series containing the target variable of samples with NaN values.
        '''
        indices_without_missing_values = train_x[~train_x.isnull().any(axis=1)].index
        
        data_x_train_missing_values = train_x.drop(indices_without_missing_values)
        data_x_train_missing_values.drop(columns = ['av_car','av_pt'], inplace = True)
        data_y_train_missing_values = train_y.drop(indices_without_missing_values)

        return  data_x_train_missing_values, data_y_train_missing_values


def merge_samples_with_and_without_missing_values(data_x_train_missing_values, data_y_train_missing_values,\
                                    data_x_train_without_missing_values, data_y_train_without_missing_values):
        '''
        Merge the training samples with and without missing values.

        Args:
        data_x_train_missing_values (pd.DataFrame):The training dataset (including only explanatory features) including only samples with NaN values.
        data_y_train_missing_values (pd.Series): A series containing the target variable of samples with NaN values.
        data_x_train_without_missing_values (pd.DataFrame): The training dataset (without the target variable) excluding samples with NaN values.
        data_y_train_without_missing_values (pd.Series): A series containing the target variable of samples without NaN values.

          training set samples with missing values, training set samples without missing values

        Returns:
        tuple: A tuple containing the following elements:
              - pd.DataFrame: The training dataset (including only explanatory features)
              - pd.Series: A series containing the target variable.

        '''
        
        subsets_x = [data_x_train_missing_values, data_x_train_without_missing_values]
        subsets_y = [data_y_train_missing_values, data_y_train_without_missing_values] 
        data_x_train_final = pd.concat(subsets_x)
        data_y_train_final = pd.concat(subsets_y)
        data_y_train_final = data_y_train_final.reset_index()
        data_y_train_final.drop(columns = 'index', inplace = True)

        return  data_x_train_final,data_y_train_final

def set_availability(data_MNL_x):

    '''
    Determine the availability of each mode.

    Args:
    data_MNL_x (pd.DataFrame): The dataset utilized in the MNL model.

    Returns:
    pd.DataFrame: The dataset utilized in the MNL model including the availability columns.

    '''
    data_MNL_x.loc[pd.isna(data_MNL_x.car_dur),'av_car'] = 0
    data_MNL_x.loc[data_MNL_x.oprijbewijsau == 0,'av_car'] = 0
    data_MNL_x.loc[data_MNL_x.av_car != 0, 'av_car'] = 1

    data_MNL_x.loc[pd.isna(data_MNL_x.pt_dur),'av_pt'] = 0
    data_MNL_x.loc[data_MNL_x.av_pt != 0, 'av_pt'] = 1

    return data_MNL_x


def define_utility_functions(classification_type):

    '''
    Define variables, alternative specific constants, beta coefficients and utiity functions.

    Args:
    classification type(str): The type of classification (binary/multiclass).

    Returns:
    tuple: A tuple containing the following elements:
          -dictionary: A dictionary including the utility functions.
          -dictionary: A dictionary including modes' availanbility.

    '''

    hhpers = Variable('hhpers')
    hhsam = Variable('hhsam')
    gemgr = Variable('gemgr')
    geslacht = Variable('geslacht')
    leeftijd = Variable('leeftijd')
    herkomst = Variable('herkomst')
    betwerk = Variable('betwerk')
    opleiding = Variable('opleiding')
    hhgestinkg = Variable('hhgestinkg')
    oprijbewijsau = Variable('oprijbewijsau')
    hhauto = Variable('hhauto')
    hhefiets = Variable('hhefiets')
    ovstkaart = Variable('ovstkaart')
    feestdag = Variable('feestdag')
    kmotiefv = Variable('kmotiefv')
    bike_dur = Variable('bike_dur')
    car_dur = Variable('car_dur')
    pt_dur = Variable('pt_dur')
    car_changes = Variable('car_changes')
    pt_changes = Variable('pt_changes')
    season = Variable('season')
    vertprov = Variable('vertprov')
    actduur = Variable('actduur')
    av_car = Variable('av_car')
    av_pt = Variable('av_pt')
    av_bike = Variable('av_bike')
    peak_hour = Variable('peak_hour')
    period_o = Variable('period_o')
    weekday = Variable('weekday')
    sted_o = Variable('sted_o')
    sted_d = Variable('sted_d')
    transit_cost = Variable('transit_cost')
    car_cost = Variable('car_cost')
    khvm = Variable('khvm')
    leeftijd_1 = Variable('leeftijd_1')
    leeftijd_2 = Variable('leeftijd_2')
    leeftijd_3 = Variable('leeftijd_3')
    herkomst_1 = Variable('herkomst_1')
    herkomst_2 = Variable('herkomst_2')
    herkomst_3 = Variable('herkomst_3')
    sted_o_1 = Variable('sted_o_1')
    sted_o_2 = Variable('sted_o_2')
    sted_o_3 = Variable('sted_o_3')
    sted_o_4 = Variable('sted_o_4')
    sted_o_5 = Variable('sted_o_5')
    sted_d_1 = Variable('sted_d_1')
    sted_d_2 = Variable('sted_d_2')
    sted_d_3 = Variable('sted_d_3')
    sted_d_4 = Variable('sted_d_4')
    sted_d_5 = Variable('sted_d_5')
    kmotiefv_1 = Variable('kmotiefv_1')
    kmotiefv_2 = Variable('kmotiefv_2')
    kmotiefv_3 = Variable('kmotiefv_3')
    season_1 = Variable('season_1')
    season_2 = Variable('season_2')
    season_3 = Variable('season_3')
    season_4 = Variable('season_4')
    vertprov_1 = Variable('vertprov_1')
    vertprov_2 = Variable('vertprov_2')
    vertprov_3 = Variable('vertprov_3')
    vertprov_4 = Variable('vertprov_4')
    vertprov_5 = Variable('vertprov_5')
    vertprov_6 = Variable('vertprov_6')
    vertprov_7 = Variable('vertprov_7')
    vertprov_8 = Variable('vertprov_8')
    vertprov_9 = Variable('vertprov_9')
    vertprov_10 = Variable('vertprov_10')
    vertprov_11 = Variable('vertprov_11')
    vertprov_12 = Variable('vertprov_12')
    period_o_1 = Variable('period_o_1')
    period_o_2 = Variable('period_o_2')
    period_o_3 = Variable('period_o_3')
    gemgr_1 = Variable('gemgr_1')
    gemgr_2 = Variable('gemgr_2')
    gemgr_3 = Variable('gemgr_3')
    hhgestinkg_1 = Variable('hhgestinkg_1')
    hhgestinkg_2 = Variable('hhgestinkg_2')
    
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)

    if classification_type == 'Binary':
        ASC_TRANSIT = Beta ('ASC_TRANSIT', 0, None, None, 1)
    else:
        ASC_TRANSIT = Beta ('ASC_TRANSIT', 0, None, None, 0)
        ASC_BIKE = Beta('ASC_BIKE', 0, None, None, 1)
        B_TIME_BIKE = Beta('B_TIME_BIKE', 0, None, None, 0)
        V_BIKE = (ASC_BIKE + (B_TIME_BIKE * bike_dur))

    B_TIME_CAR = Beta('B_TIME_CAR', 0, None, None, 0)
    B_COST_CAR = Beta('B_COST_CAR', 0 , None, None, 0)

    B_TIME_TRANSIT = Beta('B_TIME_TRANSIT', 0, None, None, 0)
    B_COST_TRANSIT = Beta('B_COST_TRANSIT', 0 , None, None, 0)

    V_CAR = (ASC_CAR +  (B_TIME_CAR  * car_dur) + (B_COST_CAR * car_cost))
    V_TRANSIT =(ASC_TRANSIT + (B_TIME_TRANSIT *  pt_dur) + (B_COST_TRANSIT * transit_cost))

    if classification_type == 'Binary':
        V = {1: V_CAR, 3: V_TRANSIT}
        av = {1: av_car, 3: av_pt} 
    else:
        V = {1: V_CAR, 3: V_TRANSIT, 4: V_BIKE}
        av = {1: av_car, 3: av_pt, 4: av_bike}

    return V, av



def train_MNL_model(V, av, i, database_train, classification_type):

    '''
    Train the MNL model.

    Args:
    V (dict): The dictionary containing the utility functions.
    av (dict): The dictionary containing the modes' availability.
    i (int): The iteration number.
    database_train (pd.DataFrame): The data utilized in the training of the MNL model.
    classification_type (str): The type of the classification (binary/multiclass).
    
    Returns:
    biogeme.bioResults: An object containing the estimation results.

    '''

    global ASC_car, B_COST_car, B_COST_transit,\
           B_TIME_car, B_TIME_transit, B_TIME_bike, VOT_car, VOT_transit, Final_log_likelihood

    khvm = Variable('khvm')
    logprob2 = models.loglogit(V, av, khvm)
    the_biogeme = bio.BIOGEME(database_train, logprob2)

    # set reporting levels
    the_biogeme.generate_pickle = False
    the_biogeme.generate_html = False
    the_biogeme.saveIterations = False

    if classification_type == 'Binary':
        the_biogeme.modelName ='MNL binary_' + str(i) # name the model
    else:
        the_biogeme.modelName ='MNL multiclass_' + str(i) 

    results = the_biogeme.estimate()
    betas = results.getEstimatedParameters()
    ASC_car.append(betas.loc['ASC_CAR']['Value'])
    if classification_type != 'Binary':
        ASC_transit.append(betas.loc['ASC_TRANSIT']['Value'])
    B_COST_car.append(betas.loc['B_COST_CAR']['Value'])
    B_COST_transit.append(betas.loc['B_COST_TRANSIT']['Value'])
    B_TIME_car.append(betas.loc['B_TIME_CAR']['Value'])
    B_TIME_transit.append(betas.loc['B_TIME_TRANSIT']['Value'])

    if classification_type != 'Binary':
        B_TIME_bike.append(betas.loc['B_TIME_BIKE']['Value'])

    VOT_car.append(60 * (betas.loc['B_TIME_CAR']['Value'] / (betas.loc['B_COST_CAR']['Value'])))
    VOT_transit.append(60 * (betas.loc['B_TIME_TRANSIT']['Value'] / betas.loc['B_COST_TRANSIT']['Value']))
    Final_log_likelihood.append(round(results.getGeneralStatistics()['Final log likelihood'][0],2))

    return results



def simulation_MNL_model(V, av, results, classification_type, data_test, i):

    '''
    Make predictions using the MNL model and append the evaluation metrics to their lists.

    Args:
    V (dict): The dictionary containing the utiity functions.
    av (dict): The dictionary containing modes' availability.
    results(biogeme.bioResults): An object containing the estimation results.
    classification type (str): The type of classification (binary/multiclass) 
    data_test (pd.DataFrame): The dataset used to test the MNL model.
    i (int): The iteration number.

    Returns:
    None

    '''


    global recall_car, precision_car, f1_score_car, recall_transit, precision_transit, f1_score_transit,recall_bike,\
           precision_bike, f1_score_bike, total_accuracy, balanced_accuracy

    database_test = db.Database('data_test_final', data_test)
                        
    prob_car = models.logit(V, av, 1)
    prob_transit = models.logit(V,av,3)

    if classification_type != 'Binary':
        prob_bike = models.logit(V,av,4)
        simulate ={'Prob. Car':  prob_car ,
                    'Prob. Transit': prob_transit,
                    'Prob. Bike': prob_bike}
    else:
        simulate ={'Prob. Car':  prob_car ,
                    'Prob. Transit': prob_transit}

    biogeme = bio.BIOGEME(database_test, simulate) 
    betaValues = results.getBetaValues ()
    simulatedValues = biogeme.simulate(betaValues)
    
    prob_max = simulatedValues.idxmax(axis=1)

    if classification_type == "Binary":
        prob_max = prob_max.replace({'Prob. Car': 1, 'Prob. Transit': 3})

        confusion_matrix_MNL = get_confusion_matrix (data_test, prob_max)

        recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, total_accuracy,\
        balanced_accuracy = append_metrics(confusion_matrix_MNL, classification_type, data_test.khvm, prob_max, i)
    else:
        prob_max = prob_max.replace({'Prob. Car': 1, 'Prob. Transit': 3, 'Prob. Bike': 4})

        confusion_matrix_MNL = get_confusion_matrix (data_test, prob_max)

        recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, recall_bike,\
                precision_bike, f1_score_bike, total_accuracy,balanced_accuracy = append_metrics(confusion_matrix_MNL, classification_type, data_test.khvm, prob_max, i)


def get_balanced_accuracy(test_y, pred_y):

    '''
    Calculate the balanced accuracy metric.
    
    Args: 
    test_y (pd.Series): A series containing the target variable from the test set.
    pred_y (pd.Series): A series containing the the predicted target variable.

    Returns:
    float: The balanced accuracy metric.

    '''
    balanced_accuracy =  round(balanced_accuracy_score(test_y, pred_y),2) * 100

    return balanced_accuracy 


def get_total_accuracy(test_y, pred_y):

    '''
    Args: 
    test_y (pd.Series): A series containing the target variable from the test set.
    pred_y (pd.Series): A series containing the predicted target variable.

    Returns:
    float: The total accuracy metric.
    '''

    total_accuracy = round(accuracy_score(test_y, pred_y),2)*100

    return total_accuracy

def get_confusion_matrix (test_y, pred_y):
    
    '''
    Compute the confusion matrix.

    Args:
    test_y (pd.Series): A series containing the target variable from the test set.
    pred_y (pd.Series): A series containing the predicted target variable.


    Returns:
    pd.DataFrame: The confusion matrix.

    '''

    data = {'y_Actual':  test_y.khvm,
                'y_Predicted': pred_y
                }
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    
    return confusion_matrix

def append_metrics(confusion_matrix, classification_type, test_y, y_pred, index): 
    '''
    Append the evaluation metrics to their lists.

    Args: 
    confusion matrix (pd.DataFrame): The confusion matrix.
    classification type (str): The type of classification (binary/multiclass).
    test_y (pd.Series): A series containing the target variable from the test set.
    pred_y (pd.Series): A series containing the predicted target variable.
    index (int): The iteration number.

    Returns: 
    typle: A tuple containing the following elements:
           list: A list containing the recall of the transit class for all iterations.
           list: A list containing the precision of the transit class for all iterations.
           list: A list containing the f1_score of the transit class for all iterations.
           list: A list containing the recall of the car class for all iterations.
           list: A list containing the precision of the car class for all iterations.
           list: A list containing the f1_score of the car class for all iterations.
           list: A list containing the recall of the bike class for all iterations.
           list: A list containing the precision of the bike class for all iterations.
           list: A list containing the f1_score of the bike class for all iterations.
           list: A list containing the balanced accuracy for all iterations.
           list: A list containing the total accuracy for all iterations.

    '''

    global precision_car, recall_car, f1_score_car, precision_transit, recall_transit, f1_score_transit,\
           precision_bike, recall_bike, f1_score_bike, balanced_accuracy, total_accuracy
    
    if classification_type == 'Binary':
        precision_car.append(round(confusion_matrix.loc[1,1]/(confusion_matrix.loc[1,1]+confusion_matrix.loc[3,1])*100,))
        recall_car.append(round(confusion_matrix.loc[1,1]/(confusion_matrix.loc[1,1]+confusion_matrix.loc[1,3])*100,))
        f1_score_car.append(round(2 * (precision_car[index] * recall_car[index])/(precision_car[index] + recall_car[index]),2))

        precision_transit.append(round(confusion_matrix.loc[3,3]/(confusion_matrix.loc[3,3]+confusion_matrix.loc[1,3])*100))
        recall_transit.append(round(confusion_matrix.loc[3,3]/(confusion_matrix.loc[3,3]+confusion_matrix.loc[3,1])*100))
        f1_score_transit.append(round(2 * (precision_transit[index] * recall_transit[index])/(precision_transit[index] + recall_transit[index])))

        balanced_accuracy.append(get_balanced_accuracy(test_y, y_pred))
        total_accuracy.append(get_total_accuracy(test_y, y_pred))

        return recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, total_accuracy, balanced_accuracy

    else:

        precision_car.append(round(confusion_matrix.loc[1,1]/(confusion_matrix.loc[1,1]+confusion_matrix.loc[3,1]+confusion_matrix.loc[4,1])*100,))
        recall_car.append(round(confusion_matrix.loc[1,1]/(confusion_matrix.loc[1,1]+confusion_matrix.loc[1,3]+confusion_matrix.loc[1,4])*100,))
        f1_score_car.append(round(2 * (precision_car[index] * recall_car[index])/(precision_car[index] + recall_car[index]),2))

        precision_transit.append(round(confusion_matrix.loc[3,3]/(confusion_matrix.loc[3,3]+confusion_matrix.loc[1,3]+confusion_matrix.loc[4,3])*100))
        recall_transit.append(round(confusion_matrix.loc[3,3]/(confusion_matrix.loc[3,3]+confusion_matrix.loc[3,1] + confusion_matrix.loc[3,4] )*100))
        f1_score_transit.append(round(2 * (precision_transit[index] * recall_transit[index])/(precision_transit[index] + recall_transit[index])))

        precision_bike.append(round(confusion_matrix.loc[4,4]/(confusion_matrix.loc[4,4]+confusion_matrix.loc[1,4]+confusion_matrix.loc[3,4])*100,))
        recall_bike.append(round(confusion_matrix.loc[4,4]/(confusion_matrix.loc[4,4]+confusion_matrix.loc[4,3]+confusion_matrix.loc[4,1])*100,))
        f1_score_bike.append(round(2 * (precision_bike[index] * recall_bike[index])/(precision_bike[index] + recall_bike[index]),2))

        balanced_accuracy.append(get_balanced_accuracy(test_y,y_pred))
        total_accuracy.append(get_total_accuracy(test_y, y_pred))
        
        return recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, recall_bike,\
               precision_bike, f1_score_transit, total_accuracy,balanced_accuracy

def append_metrics_separation_scheme_binary(confusion_matrix_first_stage, confusion_matrix_overlapping_region,index):

    global precision_car, recall_car, f1_score_car, precision_transit, recall_transit, f1_score_transit,\
           precision_bike, recall_bike, f1_score_bike, balanced_accuracy, total_accuracy
    
    '''
    Append the evaluation metrics to their lists.

    Args:
    confusion_matrix_first_stage (pd.DataFrame): The confusion matrix of the classification of the first stage.
    confusion_matrix_overlapping_region (pd.DataFrame): The confusion matrix of the classification within the overlapping region.
    index (int): The iteration number.
   
    Returns: 
    typle: A tuple containing the following elements:
           list: A list containing the recall of the transit class for all iterations.
           list: A list containing the precision of the transit class for all iterations.
           list: A list containing the f1_score of the transit class for all iterations.
           list: A list containing the recall of the car class for all iterations.
           list: A list containing the precision of the car class for all iterations.
           list: A list containing the f1_score of the car class for all iterations.
           list: A list containing the total accuracy for all iterations.
           list: A list containing the balanced accuracy for all iterations.

    '''
  
    # transit class
    recall_transit.append(round(confusion_matrix_overlapping_region.loc[3][3]/\
                    (confusion_matrix_overlapping_region.loc[3][3]+confusion_matrix_first_stage.loc[3][1]\
                    +confusion_matrix_overlapping_region.loc[3][1]),2)*100)   
    
    precision_transit.append(round(confusion_matrix_overlapping_region.loc[3,3]/(confusion_matrix_overlapping_region.loc[3,3] + \
                        confusion_matrix_overlapping_region.loc[1,3]),2)*100)
    
    f1_score_transit.append(round(2*(recall_transit[index] * precision_transit[index])/(recall_transit[index] + precision_transit[index]),2))

    # car class

    recall_car.append(round((confusion_matrix_first_stage.loc[1][1] + confusion_matrix_overlapping_region.loc[1][1])/\
                        (confusion_matrix_first_stage.loc[1][1] + confusion_matrix_overlapping_region.loc[1][1] + \
                        confusion_matrix_overlapping_region.loc[1][3]),2) *100 )
    
    precision_car.append(round((confusion_matrix_first_stage.loc[1][1] + confusion_matrix_overlapping_region.loc[1][1])/\
                        (confusion_matrix_first_stage.loc[1][1] + confusion_matrix_overlapping_region.loc[1][1] + \
                        confusion_matrix_first_stage.loc[3][1] + confusion_matrix_overlapping_region.loc[3][1]),2)*100)
    
    f1_score_car.append(round(2*(recall_car[index] * precision_car[index])/(recall_car[index] + precision_car[index]),2))

    # total metrics

    total_accuracy.append(round((confusion_matrix_overlapping_region.loc[3][3] + confusion_matrix_first_stage.loc[1][1] +\
                        confusion_matrix_overlapping_region.loc[1][1])/\
                        (confusion_matrix_overlapping_region.loc[3][3] +  confusion_matrix_first_stage.loc[3][1]+\
                        confusion_matrix_overlapping_region.loc[3][1]+confusion_matrix_first_stage.loc[1][1] + \
                        confusion_matrix_overlapping_region.loc[1][1] + \
                        confusion_matrix_overlapping_region.loc[1][3]),2))

    balanced_accuracy.append(1/2*(recall_transit[index] + recall_car[index]))

    return recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, total_accuracy,\
           balanced_accuracy



def append_metrics_separation_scheme_multiclass(confusion_matrix_first_stage, confusion_matrix_overlapping_region, confusion_matrix_non_overlapping_region, index):

    global precision_car, recall_car, f1_score_car, precision_transit, recall_transit, f1_score_transit,\
           precision_bike, recall_bike, f1_score_bike, balanced_accuracy, total_accuracy
    
    '''
    Append the evaluation metrics to their lists.

    Args:
    confusion_matrix_first_stage (pd.DataFrame): The confusion matrix of classification of the first stage.
    confusion_matrix_overlapping_region (pd.DataFrame): The confusion matrix of the classification within the overlapping region.
    confusion_matrix_non_overlapping_region (pd.DataFrame): The confusion matrix of the classification within the non-overlapping region.
    index (int): The iteration number.
   
    Returns: 
    typle: A tuple containing the following elements:
           list: A list containing the recall of the transit class for all iterations.
           list: A list containing the precision of the transit class for all iterations.
           list: A list containing the f1_score of the transit class for all iterations.
           list: A list containing the recall of the car class for all iterations.
           list: A list containing the precision of the car class for all iterations.
           list: A list containing the f1_score of the car class for all iterations.
           list: A list containing the recall of the bike class for all iterations.
           list: A list containing the precision of the bike class for all iterations.
           list: A list containing the f1_score of the bike class for all iterations.
           list: A list containing the total accuracy for all iterations.
           list: A list containing the balanced accuracy for all iterations.

    '''
    
    # transit class
    recall_transit.append(round(confusion_matrix_overlapping_region.loc[3][3]/(confusion_matrix_overlapping_region.loc[3][3]+ \
                    confusion_matrix_first_stage.loc[3][1] + confusion_matrix_overlapping_region.loc[3][1] + \
                    confusion_matrix_overlapping_region.loc[3][4]),2)*100)
    
    precision_transit.append(round(confusion_matrix_overlapping_region.loc[3][3]/(confusion_matrix_overlapping_region.loc[3][3]+\
                    confusion_matrix_overlapping_region.loc[1][3]+ confusion_matrix_overlapping_region.loc[4][3]),2)*100)
    
    f1_score_transit.append(round(2*(recall_transit[index] * precision_transit[index])/(recall_transit[index] + precision_transit[index]),2))
    
    # car class 
    recall_car.append(round((confusion_matrix_non_overlapping_region.loc[1][1] + confusion_matrix_overlapping_region.loc[1][1])/\
                     (confusion_matrix_non_overlapping_region.loc[1][1] + confusion_matrix_overlapping_region.loc[1][1] +\
                     confusion_matrix_non_overlapping_region.loc[1][4] + confusion_matrix_overlapping_region.loc[1][3]+\
                     confusion_matrix_overlapping_region.loc[1][4]),2)*100)
    
    precision_car.append(round((confusion_matrix_non_overlapping_region.loc[1][1] + confusion_matrix_overlapping_region.loc[1][1])/\
                 ((confusion_matrix_non_overlapping_region.loc[1][1] + confusion_matrix_overlapping_region.loc[1][1]) + \
                  confusion_matrix_non_overlapping_region.loc[4][1]+\
                  confusion_matrix_non_overlapping_region.loc[3][1] + confusion_matrix_overlapping_region.loc[3][1] + \
                  confusion_matrix_overlapping_region.loc[4][1]),2)*100)
    
    f1_score_car.append(round(2*(recall_car[index] * precision_car[index])/(recall_car[index] + precision_car[index])))

    # bike class

    recall_bike.append(round((confusion_matrix_non_overlapping_region.loc[4][4] + confusion_matrix_overlapping_region.loc[4][4])/\
                      (confusion_matrix_non_overlapping_region.loc[4][4] + confusion_matrix_overlapping_region.loc[4][4] + \
                      confusion_matrix_non_overlapping_region.loc[4][1]+confusion_matrix_overlapping_region.loc[4][3]+\
                      confusion_matrix_overlapping_region.loc[4][1]),2)*100)
    
    precision_bike.append(round((confusion_matrix_non_overlapping_region.loc[4][4] + confusion_matrix_overlapping_region.loc[4][4])/\
                 (confusion_matrix_non_overlapping_region.loc[1][4] + confusion_matrix_non_overlapping_region.loc[4][4] + \
                 confusion_matrix_non_overlapping_region.loc[3][4] + confusion_matrix_overlapping_region.loc[1][4] + \
                 confusion_matrix_overlapping_region.loc[3][4] + confusion_matrix_overlapping_region.loc[4][4]),2)*100)
                     

    f1_score_bike.append(round(2*(recall_bike[index] * precision_bike[index])/(recall_bike[index] + precision_bike[index])))


    # total metrics
    total_accuracy.append(round((confusion_matrix_overlapping_region.loc[3][3] + confusion_matrix_non_overlapping_region.loc[1][1] + \
                                 confusion_matrix_overlapping_region.loc[1][1] + confusion_matrix_non_overlapping_region.loc[4][4] + \
                                 confusion_matrix_overlapping_region.loc[4][4])/\
                                 (confusion_matrix_overlapping_region.loc[3][3] +confusion_matrix_overlapping_region.loc[3][1] + \
                                 confusion_matrix_overlapping_region.loc[3][4] + confusion_matrix_first_stage.loc[3][1] + 
                                 confusion_matrix_non_overlapping_region.loc[1][1] + confusion_matrix_overlapping_region.loc[1][1] + \
                                 confusion_matrix_non_overlapping_region.loc[1][4] + confusion_matrix_overlapping_region.loc[1][3]+\
                                 confusion_matrix_overlapping_region.loc[1][4] + confusion_matrix_non_overlapping_region.loc[4][4] + \
                                 confusion_matrix_overlapping_region.loc[4][4] + confusion_matrix_non_overlapping_region.loc[4][1] + \
                                 confusion_matrix_overlapping_region.loc[4][3] + confusion_matrix_overlapping_region.loc[4][1]),2))
    
    balanced_accuracy.append(1/3*(recall_transit[index] + recall_car[index] + recall_bike[index]))

    return recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, recall_bike,\
           precision_bike, f1_score_bike, total_accuracy,balanced_accuracy

                
def store_results(classification_type, technique, model):

    global recall_transit, precision_transit, f1_score_transit, recall_car, precision_car, f1_score_car, recall_bike,\
           precision_bike, f1_score_bike, total_accuracy,balanced_accuracy, ASC_car, ASC_transit, B_COST_car, B_COST_transit,\
           B_TIME_car, B_TIME_transit, B_TIME_bike, VOT_car, VOT_transit, Final_log_likelihood
    
    '''
    Store the evaluation metrics.
    
    Args:
    classification_type (str): The type of the classification (binary/classification).
    technique (str): The chosen technique (Baseline/SMOTENC/NBU/Separation scheme).
    model (str): The chosen model (RF/MNL).

    Returns:
    None
    '''
    
    results ={
                model + " " + technique +" " + "transit class" : {

                        "mean recall": round(np.mean(recall_transit),2),
                        "sd recall" :round(statistics.stdev(recall_transit),2),

                        "mean precision" : round(np.mean(precision_transit),2),
                        "sd precision" : round(statistics.stdev(precision_transit),2),

                        "mean f1_score" : round(np.mean(f1_score_transit),2),
                        "sd f1_score": round(statistics.stdev(f1_score_transit),2)
                        },

                model + " " + technique+ " " + "car class": {

                        "mean recall": round(np.mean(recall_car),2),
                        "sd recall" :round(statistics.stdev(recall_car),2),

                        "mean precision" : round(np.mean(precision_car),2),
                        "sd precision" : round(statistics.stdev(precision_car),2),

                        "mean f1_score" : round(np.mean(f1_score_car),2),
                        "sd f1_score": round(statistics.stdev(f1_score_car),2)
                        },

                model + " " + technique + " " + "total_metrics": {

                        "mean total accuracy": round(np.mean(total_accuracy),2),
                        "sd total accuracy": round(statistics.stdev(total_accuracy),2),

                        "mean balanced accuracy": round(np.mean(balanced_accuracy),2),
                        "sd balanced accuracy": round(statistics.stdev(balanced_accuracy),2)

                        }
            }
        
    if classification_type != 'Binary':

        results.update({model + " " + technique + " " + "bike class": {

                    "mean recall": round(np.mean(recall_bike),2),
                    "sd recall": round(statistics.stdev(recall_bike),2),

                    "mean precision": round(np.mean(precision_bike),2),
                    "sd precision": round(statistics.stdev(precision_bike),2),

                    "mean f1_score": round(np.mean(f1_score_bike),2),
                    "sd f1_score": round(statistics.stdev(f1_score_bike),2),
            }})
        
        if model == 'MNL':

            results.update({model + " " + technique + " " + 'B_TIME_BIKE' : {

                    "mean B_TIME_bike": round(np.mean(B_TIME_bike),4),
                    "sd B_TIME_bike": round(statistics.stdev(B_TIME_bike),4)

                    }})
            
    if model == 'MNL':

        results.update({model + " " + technique : {

                    "mean ASC_car": round(np.mean(ASC_car),4),
                    "sd ASC_car": round(statistics.stdev(ASC_car),4), 

                    "mean B_TIME_car": round(np.mean(B_TIME_car),4),
                    "sd B_TIME_car": round(statistics.stdev(B_TIME_car),4),

                    "mean B_TIME_transit": round(np.mean(B_TIME_transit),4),
                    "sd B_TIME_transit": round(statistics.stdev(B_TIME_transit),4),

                    "mean B_COST_car": round(np.mean(B_COST_car),4),
                    "sd B_COST_car": round(statistics.stdev(B_COST_car),4),

                    "mean B_COST_transit": round(np.mean(B_COST_transit),4),
                    "sd B_COST_transit": round(statistics.stdev(B_COST_transit),4),

                    "mean VOT_car": round(np.mean(VOT_car),4),
                    "sd VOT_car": round(statistics.stdev(VOT_car),4),

                    "mean VOT_transit": round(np.mean(VOT_transit),4),
                    "sd VOT_transit": round(statistics.stdev(VOT_transit),4),

                    "mean Final_log_likelihood": round(np.mean(Final_log_likelihood),4),
                    "sd Final_log_likelihood": round(statistics.stdev(Final_log_likelihood),4),

            }})
        
        
        if classification_type != 'Binary':

            results.update({model + " " + technique + " " +'ASC_transit': {

                    "mean ASC_transit": round(np.mean(ASC_transit),4),
                    "sd ASC_transit": round(statistics.stdev(ASC_transit),4),

                    }})
        

    current  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    current = current.replace('/','_').replace(',','_').replace(':','_')
    output_file = open("output_" + model + "_" + technique + " " + str(current) + ".json", "w") 

    json.dump(results, output_file, indent = 6) 

    output_file.close() 


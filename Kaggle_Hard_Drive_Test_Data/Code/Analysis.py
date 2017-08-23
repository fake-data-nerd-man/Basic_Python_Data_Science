# -*- coding: utf-8 -*-
"""
Hard-Drive Test Data
"""

import pandas as pd
#import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

#from sklearn import tree
##############################################################################
# Read CSV (comma-separated) file into DataFrame
def Data_Reading():
    df = pd.read_csv( "C:/Users/Kaustav Saha/Desktop/Python_Machine_Learning/Kaggle_Hard_Drive_Test_Data/Data/harddrive.csv" )
    return(df)

#############################################################################

# Data Cleaning
def Data_Cleaning(data):
    
    # Basic Understanding of the Data
    # Data Distribution of the Failure State
    failure_df = data.loc[:,['failure']]
    #print(failure_df.describe())
    
    # Frequency Table of the Failure State
    failure_data_distribution = pd.value_counts(failure_df.failure).reset_index()
    failure_data_distribution.columns = ['failure_states', 'frequency']
    print(failure_data_distribution)
    
    # df = df.drop('column_name', 1)
                           # where 1 is the axis number (0 for rows and 1 for columns.)
                           
    # To delete the column without having to reassign df you can do:
    # df.drop('column_name', axis=1, inplace=True)

    # Drop un-necessary Fields
    #print(data.columns.values)
    data.drop('date', axis=1, inplace=True)
    data.drop('serial_number', axis=1, inplace=True)
    #print(data.columns.values)
    #print(data.dtypes)
     
    # Make the model attribute / parameter categorical
    data["model"] = data["model"].astype('category')
    #print(data.dtypes)
    
    # Changing the colnames to more readable format
    data.columns = [      "model", 
                          "capacity", "failure", 
                          "1_normalized", "1_raw",
                          "2_normalized", "2_raw",
                          "3_normalized", "3_raw",
                          "4_normalized", "4_raw",
                          "5_normalized", "5_raw",
                          "6_normalized", "6_raw",
                          "7_normalized", "7_raw",
                          "8_normalized", "8_raw",
                          "9_normalized", "9_raw",
                          "10_normalized", "10_raw",
                          "11_normalized", "11_raw",
                          "12_normalized", "12_raw",
                          "13_normalized", "13_raw",
                          "14_normalized", "14_raw",
                          "15_normalized", "15_raw",
                          "16_normalized", "16_raw",
                          "17_normalized", "17_raw",
                          "18_normalized", "18_raw",
                          "19_normalized", "19_raw",
                          "20_normalized", "20_raw",
                          "21_normalized", "21_raw",
                          "22_normalized", "22_raw",
                          "23_normalized", "23_raw",
                          "24_normalized", "24_raw",
                          "25_normalized", "25_raw",
                          "26_normalized", "26_raw",
                          "27_normalized", "27_raw",
                          "28_normalized", "28_raw",
                          "29_normalized", "29_raw",
                          "30_normalized", "30_raw",
                          "31_normalized", "31_raw",
                          "32_normalized", "32_raw",
                          "33_normalized", "33_raw",
                          "34_normalized", "34_raw",
                          "35_normalized", "35_raw",
                          "36_normalized", "36_raw",
                          "37_normalized", "37_raw",
                          "38_normalized", "38_raw",
                          "39_normalized", "39_raw",
                          "40_normalized", "40_raw",
                          "41_normalized", "41_raw",
                          "42_normalized", "42_raw",
                          "43_normalized", "43_raw",
                          "44_normalized", "44_raw",
                          "45_normalized", "45_raw"  ]
    
    #print(data.columns.values)
    return(data)
#############################################################################

## Count the NA's Analysis
def NA_Count_Analysis(data):
    # print(data.isnull().sum(axis=1).tolist())
    print(data.isnull().sum(axis=1))
    
##############################################################################

def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum()/len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_percentages = mis_val_table.rename(
        columns = {0 : 'Missing_Values', 1 : 'Percentage_of_Missing_Values'})
        return mis_val_table_percentages 

##############################################################################

## Variable Elimination based on percentage of NA in the data
def variable_elimination(df,missing_factor): 
        df = df[(df.Percentage_of_Missing_Values < missing_factor)]
        return df
    
#####################################################

## Remove columns with any missing values ####################################
def missing_value_elimination(data): 
    df = data.dropna(axis=1,how='any')
    return(df)
 
##############################################################################

'''
Definition: DataFrame.dropna(self, axis=0, how='any', thresh=None, subset=None)
Docstring:
Return object with labels on given axis omitted where alternately any
or all of the data are missing

Parameters
----------
axis : {0, 1} --> 0:row, 1:column
how : {'any', 'all'}
    any : if any NA values are present, drop that label
    all : if all values are NA, drop that label
thresh : int, default None
    int value : require that many non-NA values
subset : array-like
    Labels along other axis to consider, e.g. if you are dropping rows
    these would be a list of columns to include

Returns
-------
dropped : DataFrame

df = df.dropna(axis=1,how='any')

'''

#############################################################################
def predictor_target_formation(data): 
    predictors = data[['1_normalized','1_raw','3_normalized','3_raw']]
    targets = data['failure']
    return(predictors,targets)

##############################################################################
## Data Splitting into training and testing sets
def data_split(predictors, targets, test_size): 
    predictor_training_data, predictor_testing_data, target_training_data, target_testing_data  = \
                                   train_test_split(predictors, targets, test_size = 0.3)
    return(predictor_training_data, predictor_testing_data, target_training_data, target_testing_data)

############### Models ######################################################
def model_building(predictor_training_data,target_training_data): 
    
    #Build Model on training data
    model = DecisionTreeClassifier()
    model = model.fit(predictor_training_data,target_training_data)

    return(model)

##############################################################################

def prediction(model,predictor_testing_data): 
    
    # Prediction
    predictions = model.predict(predictor_testing_data)
    return(predictions)

#############################################################################

def accuracy(target_testing_data,predictions): 
    
    # Accuracy
    print("-----------------------------------------------")
    print("Confusion Matrix")
    print(confusion_matrix(target_testing_data,predictions))
    print("-----------------------------------------------")
    print("Accuracy =" , (accuracy_score(target_testing_data, predictions))*100, "%")

#########################################################

def main():
    
    raw_data = Data_Reading()
    
    # Basic Statistics of all the columns
    # print(raw_data.describe()) 
    
    # shape --> Return a tuple representing the dimensionality of the DataFrame.
    # print(raw_data.shape) # (3179295, 95) --> (rows,col)
    
    cleaned_data = Data_Cleaning(raw_data)
    # print(cleaned_data.shape) # (3179295, 93) --> (rows,col)
    #########################################################################
    
    NA_Count_Analysis(cleaned_data)
    
    # Percentage of missing values -- Ascertaing the missing factor from here
    mis_val_table_percentages = missing_values_table(cleaned_data)
    print(mis_val_table_percentages)
    
    # Setting the missing factor to be 50% which means 50% of missing data is allowed
    colnames_after_variable_elimination = variable_elimination(mis_val_table_percentages,50)
    print(colnames_after_variable_elimination.shape) # (3179295, 93) --> (rows,col)
    
    # Extract out the column names and subset the initial dataframe based on it.
    refined_colnames = colnames_after_variable_elimination.index.tolist()
    refined_data = cleaned_data[refined_colnames]
    print(" Refined Data Shape = ", refined_data.shape) # (3179295, 51) --> (rows,col)
    
    # Dropping columns with NAs
    print("Number of columns before dropping missing value columns = ", len(refined_data.columns))
    print("--------------------------------------------------------------------")
    data = missing_value_elimination(refined_data)
    print("Number of columns after dropping missing value columns = ", len(data.columns))
    
    #####################################################################
    # Working Data
    print("--------------------------------------------------------------------")
    print("Working Data Shape = ", data.shape) # (3179295, 29) --> (rows,col)
    
    ###########################################################################
    
    ## Data Splitting
    predictors, targets = predictor_target_formation(data)
    
    predictor_training_data, predictor_testing_data,                        \
                             target_training_data, target_testing_data =    \
                                            data_split(predictors, targets, 0.3)
    
    ## Model Building
    model = model_building(predictor_training_data,target_training_data)
    
    ## Predictions
    predictions = prediction(model, predictor_testing_data)
    
    ## Accuracy
    accuracy(target_testing_data,predictions)

###############################################################################
    

main()
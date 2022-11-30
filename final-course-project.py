import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


raw_dataset = pd.read_csv('raw_datasets\churn_raw_data.csv') # imports csv into a data frame

churn_df = raw_dataset # the working dataframe 

## ----- Detect Duplicates -----
# Checks if 1st col equal to 2nd col in data set
# print(churn_df.iloc[:, 0].equals(churn_df.iloc[:, 1])) # returns True


## ----- Treat Duplicates -----
# Remove 1st column; it is a duplicate of the 2nd column
churn_df.drop(columns=churn_df.columns[0], axis=1, inplace=True)
# print(churn_df.head())

churn_df = churn_df.round({'MonthlyCharge': 2, 'Tenure': 2, 'Outage_sec_perweek': 1})


## ----- Detect Missing Values -----
# print(churn_df.isna().sum()) # find the number in NaNs in column


## ----- Treat Missing Values -----
# Replace the NaNs in the Age column with the mean
# ['Children', 'Age', 'Income', 'Techie', 'Phone', 'TechSupport', 'Tenure', 'Bandwidth_GB_Year']
replace_nan_list = ['Children', 'Age', 'Income', 'Tenure', 'Bandwidth_GB_Year']

for var in replace_nan_list:
    var_dataframe = churn_df[var]
    var_mean = var_dataframe.mean()
    var_dataframe.fillna(var_mean, inplace=True) # replaces NaN vals with mean vals

## print(churn_df.isna().sum()) # finds the number in NaNs in column


## ----- Detect Outliers -----
# churn_df['Zscore_Population'] = stats.zscore(churn_df.iloc[::, 0])
# print(churn_df['Zscore_Population'])
# Plot histogram to check for outliers with Seaborn
sns.set()
col_header_list = ['Lat', 'Lng', 'Population', 'Children', 'Age', 'Income', 'Outage_sec_perweek',
                   'Email', 'Contacts', 'Yearly_equip_failure', 'Tenure', 'MonthlyCharge',
                   'Bandwidth_GB_Year', 'item1', 'item2', 'item3','item4', 'item5', 'item6',
                   'item7', 'item8']

# col_header_list = ['Lat', 'Lng', 'Population', 'Children', 'Age', 'Income', 'Outage_sec_perweek',
#                    'Email']
has_outliers_arr = []

for header in col_header_list:
    z_scores = stats.zscore(churn_df[header])
    # print(header, type(z_scores))
    if z_scores.min() < -3.9 or z_scores.max() > 3.9:
        has_outliers_arr.append(( header, True ))
    else:
        has_outliers_arr.append(( header,False ))

    _ = plt.hist(z_scores)
    _ = plt.xlabel('Z-Score')
    _ = plt.ylabel(f'{header} Freq')
    # plt.show()
print(has_outliers_arr)

## ----- Treat Outliers -----



## ----- Other data quality issues -----
# limit decimal numbers to 100th point aka 2 decimal nums after point
#   - MonthlyCharge, Tenure, Outage_sec_perweek, 
# churn_df = churn_df.round({'MonthlyCharge': 2, 'Tenure': 2, 'Outage_sec_perweek': 1})
# print(churn_df[['MonthlyCharge', 'Tenure', 'Outage_sec_perweek']])

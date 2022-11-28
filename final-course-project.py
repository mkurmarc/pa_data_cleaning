import pandas as pd
from sklearn.decomposition import PCA
import numpy as np


raw_dataset = pd.read_csv('raw_datasets\churn_raw_data.csv')


# check each columns response and determine what cleaning needs to be done

# limit decimal numbers to 100th point aka 2 decimal nums after point
#   - MonthlyCharge, Tenure, Outage_sec_perweek, 
churn_df = raw_dataset
churn_df = churn_df.round({'MonthlyCharge': 2, 'Tenure': 2, 'Outage_sec_perweek': 1})
# print(churn_df[['MonthlyCharge', 'Tenure', 'Outage_sec_perweek']])

# remove 1st column; it is a duplicate of the 2nd column
churn_df.drop(columns=churn_df.columns[0], axis=1, inplace=True)
# print(churn_df)

churn_df['Children'].fillna(0, inplace=True) # removing the NAs in column - in favor of 0
# print(churn_df['Children'])
# print(churn_df['Children'].isna().sum())
print(churn_df.isna().sum()) # find the number in NaNs in column

# churn_df.isnull().sum(axis=1) # Find the number of NaNs in each row

# Replace the NaNs in the column with the median
# churn_df['Age'].fillna(churn_df['Age'].median(), inplace=True)

# Replace the NaNs in the column with the mean
# churn_df['Age'].fillna(churn_df['Age'].mean(), inplace=True)

# print(churn_df.dtypes) # Shows the data type of each column. Object means the column has multiple valuet ypes


import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


raw_dataset = pd.read_csv('raw_datasets\churn_raw_data.csv')

churn_df = raw_dataset

# ----------------------------- Duplicates -------------------------------
# Checks if 1st col equal to 2nd col in data set
# print(churn_df.iloc[:, 0].equals(churn_df.iloc[:, 1])) # returns True

# remove 1st column; it is a duplicate of the 2nd column
churn_df.drop(columns=churn_df.columns[0], axis=1, inplace=True)
# print(churn_df.head())

# -------------------------- Missing Values ------------------------------
# print(churn_df.isna().sum()) # find the number in NaNs in column
# Replace NaNs in column to 0
churn_df['Children'].fillna(0, inplace=True) 
# print(churn_df['Children'].isna().sum()) # Check how many NaNs in column

# Replace the NaNs in the Age column with the mean
# churn_df['Age'].fillna(churn_df['Age'].mean(), inplace=True)
# print(churn_df['Children'].isna().sum()) # Check how many NaNs in column

# print(churn_df.isna().sum()) # finds the number in NaNs in column

# ----------------------------- Outliers ---------------------------------
# do for all columns of numeric value
churn_df['Zscore_Population']=stats.zscore(churn_df.iloc[::, 0])
# print(churn_df[['Zscore_Population', 'Population']].head())

# Plot histogram to check for outliers
churn_df['Zscore_Population'].plot(kind='bar')
# churn_df[['Zscore_Population', 'Population']].plot()


# ---------------------- Other data quality issues -----------------------

# limit decimal numbers to 100th point aka 2 decimal nums after point
#   - MonthlyCharge, Tenure, Outage_sec_perweek, 
churn_df = churn_df.round({'MonthlyCharge': 2, 'Tenure': 2, 'Outage_sec_perweek': 1})
# print(churn_df[['MonthlyCharge', 'Tenure', 'Outage_sec_perweek']])


# print(churn_df['Age'].min())
# print(churn_df['Age'].max())
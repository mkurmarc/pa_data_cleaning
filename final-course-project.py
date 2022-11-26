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


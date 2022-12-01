import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


raw_dataset = pd.read_csv('raw_datasets\churn_raw_data.csv') # imports csv into a data frame

churn_df = raw_dataset # the working dataframe 

## ----- Detect Duplicates -----
# Check if Customer_id, Interactionares not unique columns; all vals are unique
# print(churn_df['Customer_id'].duplicated().value_counts()) 
# print(churn_df['Interaction'].duplicated().value_counts()) 

# Checks if 1st col equal to 2nd col in data set
# print(churn_df.iloc[:, 0].equals(churn_df.iloc[:, 1])) # returns True


## ----- Treat Duplicates -----
# Remove 1st column; it is a duplicate of the 2nd column
churn_df.drop(columns=churn_df.columns[0], axis=1, inplace=True)
# print(churn_df.head())


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

print(churn_df.isna().sum()) # finds the number in NaNs in column


## ----- Detect Outliers -----
# churn_df['Zscore_Population'] = stats.zscore(churn_df.iloc[::, 0])
# print(churn_df['Zscore_Population'].head())

test_df = churn_df

# test_df['Zscore_Population'] = stats.zscore(test_df['Children'])

test_df['Zscore_Children'] = stats.zscore(test_df['Children'])

test_outlier_data = test_df.query('Zscore_Children > 3.9 | Zscore_Children < -3.9')
print(test_outlier_data[['Children', 'Zscore_Children']])
print(test_outlier_data['Children'].describe())

# Plot histogram to check for outliers with Seaborn
sns.set() # now seaborn is mounted

# col_header_list = ['Lat', 'Lng', 'Population', 'Children', 'Age', 'Income', 'Outage_sec_perweek',
#                    'Email', 'Contacts', 'Yearly_equip_failure', 'Tenure', 'MonthlyCharge',
#                    'Bandwidth_GB_Year', 'item1', 'item2', 'item3','item4', 'item5', 'item6',
#                    'item7', 'item8']

# col_header_list = ['Lat', 'Lng', 'Population', 'Children', 'Age', 'Income', 'Outage_sec_perweek',
#                    'Email']
# has_outliers = []

# for header in col_header_list:
#     outliers = []
#     z_scores = stats.zscore(churn_df[header])
    
#     for score in z_scores:
#         if score < -3.9 or score > 3.9: 
#             outliers.append(score)
    
#     if len(outliers) > 0:
#         has_outliers.append((header, outliers))

    # _ = plt.hist(z_scores)
    # _ = plt.xlabel('Z-Score')
    # _ = plt.ylabel(f'{header} Freq')
    # plt.savefig(f'figures/{header}_hist.png')
    # plt.clf()

# print(has_outliers) # shows a list of tuples of variable names and their outliers

# for var, z_val in has_outliers:
#     print(var)
#     print(len(z_val))


## ----- Other data quality issues -----
# limit decimal numbers to 100th point aka 2 decimal nums after point
#   - MonthlyCharge, Tenure, Outage_sec_perweek, 
churn_df = churn_df.round({'MonthlyCharge': 2, 'Tenure': 2, 'Outage_sec_perweek': 1})
# print(churn_df[['MonthlyCharge', 'Tenure', 'Outage_sec_perweek']])

# Change Churn series data from yes/no to 1/0
churn_df['Churn'].replace({'Yes': 1, 'No': 0}, inplace=True)


## ----- PCA -----
test_pca = churn_df[['Children', 'Age', 'Income', 'Churn', 'Outage_sec_perweek','Email', 'Contacts', 
                     'Yearly_equip_failure', 'Tenure', 'MonthlyCharge']]

# Step 3: Normalize your data and applying PCA
test_pca_normalized = (test_pca - test_pca.mean()) / test_pca.std()

pca = PCA(n_components=test_pca.shape[1]) # how many principle components applied, defines shape
pca.fit(test_pca_normalized) # applies PCA func to normalized data set

# transforming it back to a data frame
test_pca2 = pd.DataFrame(pca.transform(test_pca_normalized),
                         columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
# print(test_pca2)

loadings = pd.DataFrame(pca.components_.T,
           columns=['PC1', 'PC2', 'PC3','PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'],
           index=test_pca_normalized.columns)
           
# print(loadings) # E1

## Selecting PCs
cov_matrix = np.dot(test_pca_normalized.T, test_pca_normalized / test_pca.shape[0])
eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_]

plt.plot(eigenvalues)
plt.xlabel('number of components')
plt.ylabel('eigenvalues')
plt.axhline(y=1, color="red")
# plt.show() 


## Extract dataframe 
churn_df.to_csv('cleaned_dataset.csv')
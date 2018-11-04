#%%
# =============================================================================
# Importing Modules
# =============================================================================

import pandas as pd
import numpy as np 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import time

#%% 
# =============================================================================
# Constants
# =============================================================================
MIN_AGE = 18  # Airbnb users must be at least 18 years old
MAX_AGE = 100 # Airbnb users not likely to be over 100 years old
start_time = time.time()

#%%
# =============================================================================
# Data Wrangling 
# =============================================================================

# Read in train and test data
df_train = pd.read_csv('train_users_2.csv')
df_test = pd.read_csv('test_users.csv')
id_test = df_test.id
len_train = df_train.shape[0] # Extract number of train rows

# Concatenate train and test data into one dataframe: df
df = pd.concat((df_train.drop(['country_destination'], axis=1), df_test), axis=0, ignore_index=True)
# Remove id and date_first_booking
df = df.drop(['id', 'date_first_booking'], axis=1)
# Fill NaNs with -1
df = df.fillna[-1]

# Convert strings to datetime format
df.timestamp_first_active = pd.to_datetime(df.timestamp_first_active, format='%Y%m%d%H%M%S')
df.date_account_created = pd.to_datetime(df.date_account_created) # Converting strings to DateTimes

df['month_account_created'] = df['date_account_created'].dt.month
df['month_1-2'] = ((df.month_account_created>=1) & (df.month_account_created<3)).map({True:1,False:0})
df['month_3-5'] = ((df.month_account_created>=3) & (df.month_account_created<6)).map({True:1,False:0})
df['month_6-8'] = ((df.month_account_created>=6) & (df.month_account_created<9)).map({True:1,False:0})
df['month_9-12'] =((df.month_account_created>=9) & (df.month_account_created<=12)).map({True:1,False:0})
df['month_unknown'] = (df.month_account_created.isnull()).map({True:1,False:0})

# Month Columns
month_columns = ['month_1-2', 'month_3-5', 'month_6-8', 'month_9-12', 'month_unknown']

# Cleaning Age Data
df.loc[df.age < MIN_AGE, 'age'] = np.nan
df.loc[df.age > MAX_AGE, 'age'] = np.nan

# Age brackets to categorize age data, factorization
# Buckets were determined based on 5 quintiles, determined using pd.Series.quantile()
df['age_18-26'] = ((df.age>=18) & (df.age<27)).map({True:1,False:0})
df['age_27-30'] = ((df.age>=27) & (df.age<31)).map({True:1,False:0})
df['age_31-35'] = ((df.age>=31) & (df.age<36)).map({True:1,False:0})
df['age_36-44'] = ((df.age>=36) & (df.age<45)).map({True:1,False:0})
df['age_45-100'] = ((df.age>=45) & (df.age<=100)).map({True:1, False:0})
df['age_unknown'] = (df.age.isnull()).map({True:1,False:0})

# Age columns
age_columns = ['age_18-26', 'age_27-30', 'age_31-35', 'age_36-44', 'age_45-100', 'age_unknown']

# Encode non-numeric columns
non_numeric = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 
               'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 
               'first_browser']
onehot = pd.get_dummies(df.loc[:,non_numeric], columns=non_numeric, prefix=non_numeric)

labels = df_train.country_destination.values
le = LabelEncoder()
y = le.fit_transform(labels)

X = pd.concat([onehot.loc[:len_train-1], df.loc[:len_train-1, month_columns], df.loc[:len_train-1,age_columns]], axis=1)

# Test set
X_test = pd.concat([onehot.loc[len_train:], df.loc[len_train:, month_columns], df.loc[len_train:,age_columns]], axis=1)

#%%
# =============================================================================
# Machine Learning
# =============================================================================

# Instantiate empty list to hold time elapsed
times = []

# Instantiate empty list to hold scores
scores = []

# Instantiate a Gradient Boosting Classifier: clf
clf = GradientBoostingClassifier(max_depth=3)

# Fit the Classifier to the Training Data
clf.fit(X, y)

# Predict the label probabilities for each row
y_pred = clf.predict_proba(X_test)

#%% Generating Submission
ids = [] # list of ids
cts = [] # list of countries

for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
cts += list(le.inverse_transform(np.argsort(y_pred[i])[::-1][:5]))

submission = pd.DataFrame()
submission['id'] = ids
submission['country'] = cts
submission.to_csv('submission.csv', index=False)

# Record Elapsed Time
elapsed_time = time.time() - start_time
times.append(elapsed_time)

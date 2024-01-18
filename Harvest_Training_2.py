# %%
import lightgbm as lgb
import json
import pickle
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 10)


# %%
df_train = pd.read_csv('train.csv', header=0)
df_test = pd.read_csv('test.csv', header=0)

# %%
df_train['train_flag'] = 1
df_test['train_flag'] = 0
df_test['Crop_Damage'] = 0
print(df_train.shape, df_test.shape)

df_data = pd.concat((df_train, df_test))
print(df_data.shape)
print(df_data)

# %%
feature_cols = df_train.columns.tolist()
feature_cols.remove('ID')
feature_cols.remove('Crop_Damage')
feature_cols.remove('train_flag')
label_col = 'Crop_Damage'
print(feature_cols)

# %%
df_data['ID_value'] = df_data['ID'].apply(lambda x: x.strip('F')).astype('int')

# %%
df_data = df_data.sort_values(['ID_value'])

# %%
df_data = df_data.reset_index(drop=True)

# %%
df_data['Soil_Type_Damage'] = df_data.sort_values(['ID_value']).groupby(['Soil_Type'])['Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values

df_data['Estimated_Insects_Count_Damage'] = df_data.sort_values(['ID_value']).groupby(['Estimated_Insects_Count'])['Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values


df_data['Crop_Type_Damage'] = df_data.sort_values(['ID_value']).groupby(['Crop_Type'])['Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values


df_data['Pesticide_Use_Category_Damage'] = df_data.sort_values(['ID_value']).groupby(['Pesticide_Use_Category'])['Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values


df_data['Season_Damage'] = df_data.sort_values(['ID_value']).groupby(['Season'])['Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values


df_data['Soil_Type_Damage_lag2'] = df_data.sort_values(['ID_value']).groupby(['Soil_Type'])['Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

df_data['Estimated_Insects_Count_Damage_lag2'] = df_data.sort_values(['ID_value']).groupby(['Estimated_Insects_Count'])['Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

df_data['Crop_Type_Damage_lag2'] = df_data.sort_values(['ID_value']).groupby(['Crop_Type'])['Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

df_data['Pesticide_Use_Category_Damage_lag2'] = df_data.sort_values(['ID_value']).groupby(['Pesticide_Use_Category'])['Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

df_data['Season_Damage_lag2'] = df_data.sort_values(['ID_value']).groupby(['Season'])['Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

# %%
df_data.loc[df_data['train_flag'] == 0, 'Crop_Damage'] = -999

# %%
df_data['Crop_Damage_lag1'] = df_data['Crop_Damage'].shift(fill_value=-999)
df_data['Estimated_Insects_Count_lag1'] = df_data['Estimated_Insects_Count'].shift(fill_value=-999)
df_data['Crop_Type_lag1'] = df_data['Crop_Type'].shift(fill_value=-999)
df_data['Soil_Type_lag1'] = df_data['Soil_Type'].shift(fill_value=-999)
df_data['Pesticide_Use_Category_lag1'] = df_data['Pesticide_Use_Category'].shift(fill_value=-999)
df_data['Number_Doses_Week_lag1'] = df_data['Number_Doses_Week'].shift(fill_value=-999)
df_data['Number_Weeks_Used_lag1'] = df_data['Number_Weeks_Used'].shift(fill_value=-999)
df_data['Number_Weeks_Quit_lag1'] = df_data['Number_Weeks_Quit'].shift(fill_value=-999)
df_data['Season_lag1'] = df_data['Season'].shift(fill_value=-999)

df_data['Crop_Damage_lag2'] = df_data['Crop_Damage'].shift(periods=2,fill_value=-999)
df_data['Estimated_Insects_Count_lag2'] = df_data['Estimated_Insects_Count'].shift(periods=2,fill_value=-999)
df_data['Crop_Type_lag2'] = df_data['Crop_Type'].shift(fill_value=-999)
df_data['Soil_Type_lag2'] = df_data['Soil_Type'].shift(fill_value=-999)
df_data['Pesticide_Use_Category_lag2'] = df_data['Pesticide_Use_Category'].shift(periods=2,fill_value=-999)
df_data['Number_Doses_Week_lag2'] = df_data['Number_Doses_Week'].shift(periods=2,fill_value=-999)
df_data['Number_Weeks_Used_lag2'] = df_data['Number_Weeks_Used'].shift(periods=2,fill_value=-999)
df_data['Number_Weeks_Quit_lag2'] = df_data['Number_Weeks_Quit'].shift(periods=2,fill_value=-999)
df_data['Season_lag2'] = df_data['Season'].shift(periods=2,fill_value=-999)


# %%
df_data

# %%
df_train, df_test = df_data[df_data.train_flag == 1], df_data[df_data.train_flag == 0]

# %%
df_train.drop(['train_flag'], inplace=True, axis=1)
df_test.drop(['train_flag'], inplace=True, axis=1)
df_test.drop([label_col], inplace=True, axis=1)

# %%
print(df_train.shape, df_test.shape)

# %%
del df_data

# %%
missing_impute = -999

# %%
df_train['Number_Weeks_Used'] = df_train['Number_Weeks_Used'].apply(lambda x: missing_impute if pd.isna(x) else x)
df_test['Number_Weeks_Used'] = df_test['Number_Weeks_Used'].apply(lambda x: missing_impute if pd.isna(x) else x)

df_train['Number_Weeks_Used_lag1'] = df_train['Number_Weeks_Used_lag1'].apply(lambda x: missing_impute if pd.isna(x) else x)
df_test['Number_Weeks_Used_lag1'] = df_test['Number_Weeks_Used_lag1'].apply(lambda x: missing_impute if pd.isna(x) else x)

df_train['Number_Weeks_Used_lag2'] = df_train['Number_Weeks_Used_lag2'].apply(lambda x: missing_impute if pd.isna(x) else x)
df_test['Number_Weeks_Used_lag2'] = df_test['Number_Weeks_Used_lag2'].apply(lambda x: missing_impute if pd.isna(x) else x)

# %%
df_train, df_eval = train_test_split(df_train, test_size=0.40, random_state=42, shuffle=True, stratify=df_train[label_col])

# %%
feature_cols = df_train.columns.tolist()
feature_cols.remove('ID')
feature_cols.remove('Crop_Damage')
feature_cols.remove('ID_value')
label_col = 'Crop_Damage'
print(feature_cols)

# %%
feature_cols

# %%
df_train.shape

# %%
df_train.head()

# %%
df_train['Season_lag2'].value_counts()
df_train['Season_lag1'].value_counts()
df_train['Season'].value_counts()
df_train['Season_Damage'].value_counts()
df_train['Season_Damage_lag2'].value_counts()
df_train['Soil_Type'].value_counts()
df_train['Soil_Type_lag1'].value_counts()
df_train['Soil_Type_lag2'].value_counts()
df_train['Soil_Type_Damage'].value_counts()
df_train['Soil_Type_Damage_lag2'].value_counts()
df_train['Crop_Type'].value_counts()
df_train['Crop_Type_lag1'].value_counts()
df_train['Crop_Type_lag2'].value_counts()
df_train['Crop_Type_Damage'].value_counts()
df_train['Crop_Type_Damage_lag2'].value_counts()
df_train['Estimated_Insects_Count'].value_counts()
df_train['Estimated_Insects_Count_lag1'].value_counts()
df_train['Estimated_Insects_Count_lag2'].value_counts()
df_train['Estimated_Insects_Count_Damage'].value_counts()
df_train['Estimated_Insects_Count_Damage_lag2'].value_counts()
df_train['Number_Doses_Week'].value_counts()
df_train['Number_Doses_Week_lag1'].value_counts()
df_train['Number_Doses_Week_lag2'].value_counts()
df_train['Number_Weeks_Quit'].value_counts()
df_train['Number_Weeks_Quit_lag1'].value_counts()
df_train['Number_Weeks_Quit_lag2'].value_counts()
df_train['Number_Weeks_Used'].value_counts()
df_train['Number_Weeks_Used_lag1'].value_counts()
df_train['Number_Weeks_Used_lag2'].value_counts()
df_train['Pesticide_Use_Category'].value_counts()
df_train['Pesticide_Use_Category_lag1'].value_counts()
df_train['Pesticide_Use_Category_lag2'].value_counts()
df_train['Pesticide_Use_Category_Damage'].value_counts()
df_train['Pesticide_Use_Category_Damage_lag2'].value_counts()
df_train['Crop_Damage_lag1'].value_counts()
df_train['Crop_Damage_lag2'].value_counts()




# %%
cat_cols = ['Crop_Type', 'Soil_Type', 'Pesticide_Use_Category', 'Season', 'Crop_Type_lag1', 'Soil_Type_lag1', 'Pesticide_Use_Category_lag1', 'Season_lag1']

# %%
params = {}
params['learning_rate'] = 0.04
params['max_depth'] = 18
params['n_estimators'] = 3000
params['objective'] = 'multiclass'
params['boosting_type'] = 'gbdt'
params['subsample'] = 0.7
params['random_state'] = 42
params['colsample_bytree']=0.7
params['min_data_in_leaf'] = 55
params['reg_alpha'] = 1.7
params['reg_lambda'] = 1.11
params['class_weight']: {0: 0.44, 1: 0.4, 2: 0.37}

# %%
df_eval.columns

# %%
clf = lgb.LGBMClassifier(**params)
    
clf.fit(df_train[feature_cols], df_train[label_col], early_stopping_rounds=100, eval_set=[(df_train[feature_cols], df_train[label_col]), (df_eval[feature_cols], df_eval[label_col])], eval_metric='multi_error', verbose=True, categorical_feature=cat_cols)

eval_score = accuracy_score(df_eval[label_col], clf.predict(df_eval[feature_cols]))

print('Eval ACC: {}'.format(eval_score))

# %%
best_iter = clf.best_iteration_
params['n_estimators'] = best_iter
print(params)

# %%
df_train = pd.concat((df_train, df_eval))
df_train['Crop_Damage'].value_counts()

# %%
clf = lgb.LGBMClassifier(**params)

clf.fit(df_train[feature_cols], df_train[label_col], eval_metric='multi_error', verbose=False, categorical_feature=cat_cols)

# eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))
eval_score_acc = accuracy_score(df_train[label_col], clf.predict(df_train[feature_cols]))

print('ACC: {}'.format(eval_score_acc))

# %%
preds = clf.predict(df_test[feature_cols])
preds

# %%
Counter(df_train['Crop_Damage'])

# %%
Counter(preds)

# %%
plt.rcParams['figure.figsize'] = (12, 6)
lgb.plot_importance(clf)
plt.show()

# %%
new=[[1,230,32,56,20.0,2,1,0,0.2,0.1,1.2,0.3,0.66,1.3,0.2,0.0,0.4,0.1,0,345,1,0,3,234,450,35.0,1,0,504,1,0,3,34,54,59.0,1]]

# %%
Ynew=clf.predict(new)

# %%
Ynew

# %%
"""
import pickle
file=open('harvestOutcome_model.pkl','wb')
pickle.dump(clf,file)
"""

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

lg_pred=clf.predict(df_train[feature_cols])
print(accuracy_score(df_train[label_col],lg_pred))
print(confusion_matrix(df_train[label_col],lg_pred))
print(classification_report(df_train[label_col],lg_pred))
#eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))
#print(eval_score_auc)

# %%
import joblib
joblib.dump(clf, "harvest_model.pkl")

# %%
data=[[1,230,32,56,20.0,2,1,0,0.2,0.1,1.2,0.3,0.66,1.3,0.2,0.0,0.4,0.1,0,345,1,0,3,234,450,35.0,1,0,504,1,0,3,34,54,59.0,1]]

# %%
model=joblib.load("harvest_model.pkl")
print(model.predict(data))
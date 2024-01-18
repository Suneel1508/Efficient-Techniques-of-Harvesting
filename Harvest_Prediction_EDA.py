# %%
"""
## Aim
#### To determine the outcome of the harvest season, i.e. whether the crop would be healthy (alive), damaged by pesticides, or damaged by other reasons.
"""

# %%
"""
## Data Description
#### We have two datasets given to train and test.
+ ID: UniqueID 
+ Estimated_Insects_Count: Estimated insects count per square meter 
+ Crop_Type: Category of Crop(0,1) 
+ Soil_Type: Category of Soil (0,1) 
+ Pesticide_Use_Category: Type of pesticides uses (1- Never, 2-Previously Used, 3-Currently Using) 
+ Number_Doses_Week: Number of doses per week 
+ Number_Weeks_Used: Number of weeks used 
+ Number_Weeks_Quit: Number of weeks quit 
+ Season: Season Category (1,2,3) 
+ Crop_Damage: Crop Damage Category (0=alive, 1=Damage due to other causes, 2=Damage due to Pesticides)
"""

# %%
"""
## Data Preprocessing
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# %%
df_tn=pd.read_csv("train.csv")
df_tn["source"]="train"
df_tst=pd.read_csv("test.csv")
df_tst["source"]="test"
df=df_tn
df.head()

# %%
df.tail()

# %%
"""
##### To determine to count of unique values in the train dataset and pront the unique values if there are less than 5.
"""

# %%
for i in df.columns:
    a=df[i].unique()
    len(a)
    print("Column Name:",i)
    print("Count of Unique Values:",len(a))
    if len(a)<5:
        print("Unique Values:",a)
    print()

# %%
df.nunique()

# %%
df.columns

# %%
df.isnull().sum()

# %%
"""
##### Observation 
+ Number_Weeks_Used has 9000  missing data.
"""

# %%
df.describe()

# %%
#Replacing null values with mean of the particular coloumn
df['Number_Weeks_Used'].fillna(df['Number_Weeks_Used'].mean(),inplace=True)

# %%
df.isnull().sum()

# %%
"""
##### Observation 
+ Null values in Number_Weeks_Used has been replaced.
"""

# %%
df.describe()

# %%
df.dtypes

# %%
df.dtypes.value_counts()

# %%
sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')

# %%
"""
##### Observation:
+ Estimated_Insects_count,Pesticide_use_category and Number_weeks_used are positively correlated with Crop damage.

+ Number_weeks_used  is positively correlated with Estimated_Insects_count and Pesticide_use_category. 

+ Number_weeks_Quit is highly negatively correlated with Pesticide_use_category and Number_weeks_used.
"""

# %%
"""
## Univariate Analysis
"""

# %%
"""
#### Crop Damage
"""

# %%
plt.figure(figsize=(12,5))
sns.catplot(x='Crop_Damage', data=df,kind='count',hue='Crop_Type')
plt.xlabel("Crop_Damage", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Crop_Damaged Grouped Count", fontsize=20)
plt.xticks(rotation=45)
plt.show()

# %%
"""
##### Observation:
+ Crop damage due to pesticides are less in comparison to damage due to other causes.

+ Crop type 0 has higher chance of survival compared to crop type 1. 
"""

# %%
plt.figure(figsize=(30,20))
sns.countplot(x='Estimated_Insects_Count', data=df,hue='Crop_Type')
plt.xlabel("Estimated_Insects_Count", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Insects Count w.r.t Crop Type", fontsize=20)
plt.xticks(rotation=45)
plt.show()

# %%
"""
##### Observation:
+ Crop 0 contains the most number of insects.
"""

# %%
#plt.figure(figsize=(12,5))
sns.countplot(x='Soil_Type', data=df,hue='Crop_Type')
plt.xlabel("Soil_Type", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Soil Type w.r.t Crop Type", fontsize=20)
plt.xticks(rotation=45)
plt.show()

# %%
"""
##### Observation:
+ Crop Type 0 is showing good results in both the soil types, like the count of the crop cultivated is high.
+ Crop Type 1 is more suitable with soil type 0 rather than soil type 1. 
"""

# %%
#plt.figure(figsize=(12,5))
sns.countplot(x='Pesticide_Use_Category', data=df,hue='Crop_Type')
plt.xlabel("Pesticide_Use_Category", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Pesticide_Use_Category w.r.t Crop Type", fontsize=20)
plt.xticks(rotation=45)
plt.show()

# %%
"""
##### Observation:
+ Usage of Type 1 pesticide is negligible compared to type 2 and type 3.
+ Type 2 pesticide is mostly is used on crop type 0, maybe because its more prone to insects.
+ Type 3 pesticide is equally used on both the type of crops.
"""

# %%
#plt.figure(figsize=(12,5))
sns.countplot(x='Number_Doses_Week', data=df,hue='Crop_Type')
plt.xlabel("Number_Doses_Week", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Number_Doses_Week w.r.t Crop Type", fontsize=20)
plt.xticks(rotation=45)
plt.show()

# %%
"""
### Observation:
+ Majority of crops require 20 weeks of pesticide usage
+ Crop_type 0 count is high when compared with crop type 1 in the usage of pesticides.
"""

# %%
plt.figure(figsize=(30,20))
sns.countplot(x='Number_Weeks_Used', data=df,hue='Crop_Type')
plt.xlabel("Number_Weeks_Used", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Number_Weeks_Used w.r.t Crop Type", fontsize=20)
plt.xticks(rotation=45)
plt.show()

# %%
#plt.figure(figsize=(30,20))
sns.countplot(x='Number_Weeks_Quit', data=df,hue='Crop_Type')
plt.xlabel("Number_Weeks_Quit", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Number_Weeks_Quit w.r.t Crop Type", fontsize=20)
plt.xticks(rotation=45)
plt.show()

# %%
sns.countplot(x='Season', data=df,hue='Crop_Type')
plt.xlabel("Season", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Season w.r.t Crop Type", fontsize=20)
plt.xticks(rotation=45)
plt.show()

# %%
"""
### Observation:
+ Type 2 season shows the highest production of both the type of crops.
+ Crop_type 0 shows major differences in production w.r.t season than crop_type 1
"""

# %%
"""
## Bivariate Analysis
"""

# %%
fig, [ax1,ax2,ax3] = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
ax1=sns.countplot(x="Crop_Damage" ,hue="Pesticide_Use_Category",data=df[df["Crop_Damage"]==0],ax=ax1)
ax1.set_title("Crop Damage vs Insect Count for Crop Type")
ax2=sns.countplot(x="Crop_Damage" ,hue="Pesticide_Use_Category",data=df[df["Crop_Damage"]==1],ax=ax2)
ax2.set_title("Crop Damage vs Number Week Used")
ax3=sns.countplot(x="Crop_Damage" ,hue="Pesticide_Use_Category",data=df[df["Crop_Damage"]==2],ax=ax3)
ax3.set_title("Crop Damage vs Pesticide Use Category ")

# %%
"""
##### Observation:
+ Type 2 pesticide is much safer to use as compared to Type 3 pesticide.
+ Type 3 pesticide shows most pesticide related damage to crops.
"""

# %%
plt.figure(figsize=(12,5))
g= sns.FacetGrid(df, col='Crop_Damage',size=5)
g = g.map(sns.distplot, "Number_Weeks_Used")
plt.show()

# %%
"""
##### Observation:
+ From Graph 1 we can conclude that till 20-25 weeks damage due to pesticide is negligible.
+ From Graph 3 we can see that after 20 weeks damage due to use of pesticide increrases significantly.
"""

# %%
sns.barplot(x="Crop_Damage" ,y="Estimated_Insects_Count",hue="Crop_Type",data=df)

# %%
"""
##### Observation:
+ Clearly observed that Most insect attacks are done on crop type 0.
"""

# %%
plt.figure(figsize=(12,5))
sns.catplot(x='Crop_Type',y='Number_Weeks_Used', data=df, palette="hls",kind='bar',col='Crop_Damage')
plt.xticks(rotation=45)
plt.show()

# %%
"""
##### Observation:
+ Crop Type 0 is more vulnerable to pesticide related and other damages as compared to Type1
+ Avg. duration of pesticide related damage is lower for Crop type 1.
"""

# %%
"""
## Outlier Analysis
"""

# %%
df.dtypes

# %%
"""
##### We need to search for outliers in:
+ Estimated_Insects_Count
+ Number_Doses_Week
+ Number_Weeks_Used
+ Number_Weeks_Quit
##### as the unique data in rest of the columns is less than 5 and the data cannot go out of bounds. 
"""

# %%
df.plot(kind="box",subplots=True,layout=(5,5),figsize=(15,15))

# %%
#Function to remove outliers in a particular column
def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    return df_final

# %%
"""
#### Estimated_Insects_Count
"""

# %%
sns.boxplot(x='Estimated_Insects_Count',data=df,orient='V',showfliers=True,\
            meanline=True,showmeans=True)

# %%
app_income = df['Estimated_Insects_Count']
q1,q2,q3 = app_income.quantile([0.25,0.5,0.75])
print('q1 =',q1,'q2 =',q2,'q3 =',q3)
iqr=q3-q1
print('iqr =',iqr)

# %%
print("Upper limit = ",(q3+1.5*iqr))
print("Lower limit = ",(q1-1.5*iqr))
print("Upper Outliers = ",(app_income>(q3+1.5*iqr)).sum())
print("Lower Outliers = ",(app_income<(q1-1.5*iqr)).sum())

# %%
#count before removing the outliers
df['Estimated_Insects_Count'].count()

# %%
outlier_removed_EIC=remove_outlier_IQR(df.Estimated_Insects_Count)

# %%
#count after removing the outliers
outlier_removed_EIC.count()

# %%
"""
#### Number_Doses_Week
"""

# %%
sns.boxplot(x='Number_Doses_Week',data=df,orient='V',showfliers=True,\
            meanline=True,showmeans=True)

# %%
app_income = df['Number_Doses_Week']
q1,q2,q3 = app_income.quantile([0.25,0.5,0.75])
print('q1 =',q1,'q2 =',q2,'q3 =',q3)
iqr=q3-q1
print('iqr =',iqr)

# %%
print("Upper limit = ",(q3+1.5*iqr))
print("Lower limit = ",(q1-1.5*iqr))
print("Upper Outliers = ",(app_income>(q3+1.5*iqr)).sum())
print("Lower Outliers = ",(app_income<(q1-1.5*iqr)).sum())

# %%
#count before removing the outliers
df['Number_Doses_Week'].count()

# %%
outlier_removed_NDW=remove_outlier_IQR(df.Number_Doses_Week)

# %%
#count after removing the outliers
outlier_removed_NDW.count()

# %%
"""
#### Number_Weeks_Used
"""

# %%
sns.boxplot(x='Number_Weeks_Used',data=df,orient='V',showfliers=True,\
            meanline=True,showmeans=True)

# %%
app_income = df['Number_Weeks_Used']
q1,q2,q3 = app_income.quantile([0.25,0.5,0.75])
print('q1 =',q1,'q2 =',q2,'q3 =',q3)
iqr=q3-q1
print('iqr =',iqr)

# %%
print("Upper limit = ",(q3+1.5*iqr))
print("Lower limit = ",(q1-1.5*iqr))
print("Upper Outliers = ",(app_income>(q3+1.5*iqr)).sum())
print("Lower Outliers = ",(app_income<(q1-1.5*iqr)).sum())

# %%
#count before removing the outliers
df['Number_Weeks_Used'].count()

# %%
outlier_removed_NWU=remove_outlier_IQR(df.Number_Weeks_Used)

# %%
#count after removing the outliers
outlier_removed_NWU.count()

# %%
"""
#### Number_Weeks_Quit
"""

# %%
sns.boxplot(x='Number_Weeks_Quit',data=df,orient='V',showfliers=True,\
            meanline=True,showmeans=True)

# %%
app_income = df['Number_Weeks_Quit']
q1,q2,q3 = app_income.quantile([0.25,0.5,0.75])
print('q1 =',q1,'q2 =',q2,'q3 =',q3)
iqr=q3-q1
print('iqr =',iqr)

# %%
print("Upper limit = ",(q3+1.5*iqr))
print("Lower limit = ",(q1-1.5*iqr))
print("Upper Outliers = ",(app_income>(q3+1.5*iqr)).sum())
print("Lower Outliers = ",(app_income<(q1-1.5*iqr)).sum())

# %%
#count before removing the outliers
df['Number_Weeks_Quit'].count()

# %%
outlier_removed_NWQ=remove_outlier_IQR(df.Number_Weeks_Quit)

# %%
#count after removing the outliers
outlier_removed_NWQ.count()

# %%
"""
## Storing cleaned data in new dataframe
"""

# %%
df1=pd.concat([df['ID'],outlier_removed_EIC,df['Crop_Type'],df['Soil_Type'],df['Pesticide_Use_Category'],outlier_removed_NDW,outlier_removed_NWU,outlier_removed_NWQ,df['Season'],df['Crop_Damage'],df['source']],axis=1)
df1

# %%
df1.plot(kind="box",subplots=True,layout=(5,5),figsize=(15,15))

# %%
"""
##### Observation:
+ we can see that outliers have been removed in Insect_Count, Number_Doses_Weeks,Number_Weeks_Quit
"""

# %%
df1.info()

# %%
df1.describe()

# %%
df1.isnull().sum()

# %%
"""
#### Handling the null values
"""

# %%
#Replacing null values with mean of the particular coloumn

df1['Estimated_Insects_Count'].fillna(df1['Estimated_Insects_Count'].mean(),inplace=True)
df1['Number_Doses_Week'].fillna(df1['Number_Doses_Week'].mean(),inplace=True)
df1['Number_Weeks_Used'].fillna(df1['Number_Weeks_Used'].mean(),inplace=True)
df1['Number_Weeks_Quit'].fillna(df1['Number_Weeks_Quit'].mean(),inplace=True)

# %%
df1.isnull().sum()

# %%
"""
## Skew Analysis
"""

# %%
df1.skew()

# %%
df1.hist(figsize=(15,15), layout=(4,4), bins=20)

# %%
"""
##### Observation
+ We can see that all the data is normally distributed.
"""

# %%
df1.info()

# %%
sns.heatmap(df1.corr(),annot=True)

# %%
df1.to_csv('harvest_cleaned_dataset.csv',index=False)
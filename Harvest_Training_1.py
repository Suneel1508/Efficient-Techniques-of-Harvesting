# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# %%
df=pd.read_csv("harvest_cleaned_dataset.csv")
df.head()

# %%
df['Crop_Damage'].value_counts()

# %%
df_damage2 = df[df['Crop_Damage'] == 2]

# %%
df_damage2

# %%
df_damage2.to_csv('harvest_damage_2.csv',index=False)

# %%
#Importing libraries
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

# %%
df = df.drop(columns=['ID','source'])

# %%
df_xc=df.drop(columns=['Crop_Damage'])
yc=df[["Crop_Damage"]]
print(df_xc.head())
print(yc.head())

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xc = sc.fit_transform(df_xc)
df_xc=pd.DataFrame(xc,columns=df_xc.columns)

# %%
#defining a function to find accuracy score, crossvalidation score for the given dataset
def max_acc_score(names,model_c,df_xc,yc):
    accuracy_scr_max = 0
    for r_state in range(42,100):
        train_xc,test_xc,train_yc,test_yc = train_test_split(df_xc,yc,random_state = r_state,test_size = 0.33,stratify = yc)
        model_c.fit(train_xc,train_yc)
        accuracy_scr = accuracy_score(test_yc,model_c.predict(test_xc))
        if accuracy_scr> accuracy_scr_max:
            accuracy_scr_max=accuracy_scr
            final_state = r_state
            final_model = model_c
            mean_acc = cross_val_score(final_model,df_xc,yc,cv=5,scoring="accuracy").mean()
            std_dev = cross_val_score(final_model,df_xc,yc,cv=5,scoring="accuracy").std()
            cross_val = cross_val_score(final_model,df_xc,yc,cv=5,scoring="accuracy")
    print('\033[1m',"Results for model : ",names,'\n','\033[0m'
          "max accuracy score is" , accuracy_scr_max ,'\n',
          "Mean accuracy score is : ",mean_acc,'\n',
          "Std deviation score is : ",std_dev,'\n',
          "Cross validation scores are :  " ,cross_val)
    print(" "*100)

# %%
#Now by using multiple Algorithms we are calculating the best Algo which suit best for our data set
accuracy_scr_max = []
accuracy=[]
std_dev=[]
mean_acc=[]
cross_val=[]
models=[]
models.append(('Random Forest', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('Gaussian NB',GaussianNB()))


for names,model_c in models:
    max_acc_score(names,model_c,df_xc,yc)

# %%
kNN=KNeighborsClassifier()
parameters={"n_neighbors":range(2,30)}
clf = GridSearchCV(kNN, parameters, cv=5,scoring="accuracy")
clf.fit(df_xc,yc)
clf.best_params_

# %%
#Again running KNeighborsClassifier with n_neighbor = 20
kNN=KNeighborsClassifier(n_neighbors=20)
max_acc_score("KNeighbors Classifier",kNN,df_xc,yc)

# %%
xc_train,xc_test,yc_train,yc_test=train_test_split(df_xc, yc,random_state = 80,test_size=0.20,stratify=yc)
kNN.fit(xc_train,yc_train)
yc_pred=kNN.predict(xc_test)

# %%
yc_pred

# %%
nnn = pd.DataFrame(yc_pred)

# %%
nnn.describe()

# %%
new = [[188,1,0,1,0,0,0,1]]

# %%
Ynew=kNN.predict(new)

# %%
Ynew

# %%
import joblib
joblib.dump(kNN, "harvest_outcome_model.pkl")

# %%
data = [[188,1,0,1,0,0,0,1]]

# %%
model=joblib.load("harvest_outcome_model.pkl")
print(model.predict(data))

# %%


# %%
data = [[342,1,0,1,0,0,0,2]]

# %%
model=joblib.load("harvest_outcome_model.pkl")
print(model.predict(data))

# %%

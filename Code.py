import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport # checking data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#open data
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

##imputation data
new_train_df = train_df.copy()
new_train_df['Age'].fillna(train_df['Age'].mean(), inplace = True)

##delete column there are missing value more than 50%
# new_train_df.isnull().mean()
new_train_df = new_train_df.drop(columns=['Cabin'])

#Create new column for group data
labels = ['Childhood', 'teens', 'Mature', 'Elderly']
bins = [0., 12., 22., 60., 100.]
new_train_df['Age_Category'] = pd.cut(new_train_df['Age'], labels=labels, bins=bins, include_lowest=False)

#One Hot Encoding string column
new_train_df = pd.get_dummies(new_train_df, columns=['Sex', 'Age_Category','Embarked'],dtype=float)

#Feture Engineering
features = ['Pclass', 'SibSp','Parch','Fare','Sex_female','Sex_male'
            ,'Age_Category_Childhood','Age_Category_teens','Age_Category_Mature','Age_Category_Elderly',
           'Embarked_C','Embarked_Q','Embarked_S']
X = new_train_df[features]
y = new_train_df.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Fit model & Prediction
DecisionModel = DecisionTreeClassifier(random_state = 1)
XgboostModel = XGBClassifier(random_state = 1)
RandomModel = RandomForestClassifier(random_state = 1)

DictModel = {'DecisionTreeClassifierr': DecisionModel,'RandomForestClassifier':RandomModel, 'XGBClassifie':XgboostModel}

#Fit model & Prediction
for Key, Model in DictModel.items():
    Model.fit(X_train, y_train)
    predictions1 = Model.predict(X_test)
    print('Model accuracy score with',Key,':', accuracy_score(y_test, predictions1))

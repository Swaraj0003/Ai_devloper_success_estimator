
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('dev.csv')






x=df.drop(['distractions','task_success'],axis=1)

y=df['task_success']



x['coffee_intake_mg'] = x['coffee_intake_mg'].apply(lambda interval: interval.mid)
x_round=x[x.select_dtypes(include='float').columns] = x.select_dtypes(include='float').round().astype(int)











x[x.select_dtypes(include='float').columns] = x.select_dtypes(include='float').round().astype(int)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=42)



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10]
}

model = RandomForestClassifier()
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(x, y)



import joblib

joblib.dump(grid.best_estimator_, 'model.pkl')













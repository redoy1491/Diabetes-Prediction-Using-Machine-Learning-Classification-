import pandas as pd
import numpy as np
import pickle


#preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

#Model
from sklearn.tree import DecisionTreeClassifier

#metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score

#Load dataset
df = pd.read_csv('diabetes.csv')

#sepearate x and y
x = df.drop('Outcome',axis = 1)
y = df['Outcome']

#Train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#tran best model with best paramaeters
dec_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42,
    class_weight='balanced'
    )

#fit the model
dec_tree.fit(x_train,y_train)

#evaluation
y_pred = dec_tree.predict(x_test)
print('Classification Report: \n',classification_report(y_test,y_pred))
print('Confusion_matrix: \n',confusion_matrix(y_test,y_pred))
print('Accuracy: ',accuracy_score(y_test,y_pred))
print('Precision: ',precision_score(y_test,y_pred))
print('Recall: ' ,recall_score(y_test,y_pred))
print('F1 Score: ',f1_score(y_test,y_pred))

#save descision tree model
with open('diabetes_model.pkl','wb') as file:
  pickle.dump(dec_tree,file)

print('\Descision Tree Model Trained Successfully and saved in pkl file')
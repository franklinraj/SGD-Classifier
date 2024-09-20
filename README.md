# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
```
step 1. Load the Iris Dataset
step 2. Create a DataFrame
step 3. Split Features and Target
step 4. Train-Test Split
step 5. Create and Train SGD Classifier
step 6. Make Predictions
step 7. Evaluate the Classifier
```
   
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: FRANKLIN RAJ G
RegisterNumber:  212223230058
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Iris data set
iris = load_iris()

# create a pandas dataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target

# display the first few rows and columns
print(df.head())

#split the data into features (X) and target(Y)
X = df.drop('target',axis=1)
y = df['target']

# split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# train the classifier on the training data
sgd_clf.fit(X_train, y_train)

# make predictions on the testing data
y_pred= sgd_clf.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# calculate the confusion matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
## Dataset:
![Unit-2 ex7 out1](https://github.com/user-attachments/assets/d7164e67-3f91-4a51-b3d9-4269522752c5)

## Accuracy & Confusion matrix:
![Unit -2 ex7 out2](https://github.com/user-attachments/assets/6c63ba0d-53ac-4f86-b39b-dae47682798c)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

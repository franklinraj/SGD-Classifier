# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
step 1.Start

step 2.Import Libraries: Import the necessary libraries - pandas, numpy, and matplotlib.pyplot.

step 3.Load Dataset: Load the dataset using pd.read_csv.

step 4.Remove irrelevant columns (sl_no, salary).

step 5.Convert categorical variables to numerical using cat.codes.

step 6.Separate features (X) and target variable (Y).

step 7.Define Sigmoid Function: Define the sigmoid function.

step 8.Define Loss Function: Define the loss function for logistic regression.

step 9.Define Gradient Descent Function: Implement the gradient descent algorithm to optimize the parameters.

step 10.Training Model: Initialize theta with random values, then perform gradient descent to minimize the loss and obtain the optimal parameters.

step 11.Define Prediction Function: Implement a function to predict the output based on the learned parameters.

step 12.Evaluate Accuracy: Calculate the accuracy of the model on the training data.

step 13.Predict placement status for a new student with given feature values (xnew).

step 14.Print Results: Print the predictions and the actual values (Y) for comparison.

step 15.Stop.
```
## Program:

## Program to implement the prediction of iris species using SGD Classifier.
## Developed by: ADITAAYAN M
## RegisterNumber: 212223040006
*/
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrics
import matplotlib.pyplot as plt
import seaborn as sns
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```


## Output:
![368835349-817e65d2-27c8-4758-b3cf-052b473b8f3a](https://github.com/user-attachments/assets/cc0b4d85-2d79-41b2-bb74-82ddf2abe26a)

![368835394-7fea0925-030c-4aa2-b1c8-e51c639c0f83](https://github.com/user-attachments/assets/e0b48a48-e3a8-43d5-a14d-f242ce7cf316)

![368827478-25a669dc-3146-482e-8aab-9306639f3292](https://github.com/user-attachments/assets/6101e353-e42d-4c06-90ba-4e558e95266d)




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

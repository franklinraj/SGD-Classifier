# Implementation-of-Logistic-Regression-Using-Gradient-Descent
### Date:
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step 1:
Import the necessary python packages
### Step 2:
Read the dataset.
### Step 3:
Define X and Y array.
### Step 4:
Define a function for costFunction,cost and gradient.
### Step 5:
Define a function to plot the decision boundary and predict the Regression value 

## Program:
Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: ADITAAYAN M

RegisterNumber:212223040006
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Placement_Data.csv")
df

df=df.drop("sl_no",axis=1)
df=df.drop("salary",axis=1)
df

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["gender"]=df["gender"].astype('category')
df["ssc_b"]=df["ssc_b"].astype('category')
df["hsc_b"]=df["hsc_b"].astype('category')
df["hsc_s"]=df["hsc_s"].astype('category')
df["degree_t"]=df["degree_t"].astype('category')
df["workex"]=df["workex"].astype('category')
df["specialisation"]=df["specialisation"].astype('category')
df["status"]=df["status"].astype('category')
df.dtypes

df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df

X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
Y

theta = np.random.random(X.shape[1]) # intitialise the model parameter
y=Y
# define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# define the loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*log(1-h))

#define the gradient descent algorithm
def gradient_descent(theta, X,y, alpha, num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-= alpha*gradient
    return theta

#train the model
theta = gradient_descent(theta,X,y,alpha = 0.01, num_iterations = 1000)
# Make predictions
def predict(theta, X):
    h= sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
y_pred

# evaluate the model
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy",accuracy)

print(y_pred)

print(Y)

xnew= np.array([[0,87,0,95,0,2,0,0,1,0,0,0]])
y_prednew=predict(theta,xnew)
y_prednew
```

## Output:

![368826439-6665f45c-702f-43f5-a598-42fe77819b8d](https://github.com/user-attachments/assets/c2ae88ff-d368-4a60-ae2d-478c9425d8c1)


![368826453-dd5d313d-0e0a-4d53-a025-653d95e70ac1](https://github.com/user-attachments/assets/d4933845-60b4-42e6-897c-8802bf2e0561)


![368826656-99754112-8505-49ad-94c1-26d23e4edd36](https://github.com/user-attachments/assets/25a669dc-3146-482e-8aab-9306639f3292)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.




# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas numpy and matplotlib
2.Upload a file that contains the required data
3. find x,y using sklearn
4. Use line chart and disply the graph and print the mse, mae,rmse


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Manoj karthik R
RegisterNumber: 212222240061 

/*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/csv.csv')
df.head()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,-1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:
![image](https://user-images.githubusercontent.com/119560395/230017225-c83264c5-ff8b-4633-8bed-3cf244b9f362.png)
![image](https://user-images.githubusercontent.com/119560395/230017294-959f9bf8-2a89-434b-a638-b11ef984eba9.png)
![image](https://user-images.githubusercontent.com/119560395/230017412-cc09e60f-663a-4a95-abf0-98e0059155eb.png)
![image](https://user-images.githubusercontent.com/119560395/230017505-eae29baa-e3ef-4d6b-9075-cfa06c175e45.png)
![image](https://user-images.githubusercontent.com/119560395/230017601-11385667-7c4e-4080-9f3b-3cf887f9e699.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

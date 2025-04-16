# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets
2. Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters
3. Train your model -Fit model to training data -Calculate mean salary value for each subset
4. Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance
5. Tune hyperparameters -Experiment with different hyperparameters to improve performance
6. Deploy your model Use model to make predictions on new data in real-world application.

## Program && Output:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: TAMIZHSELVAN B
RegisterNumber: 212223230225
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/8736c4ba-ac49-4601-ae61-0e582455b62c)

```
data.info()
```
![image](https://github.com/user-attachments/assets/04831b1d-9106-409f-9d84-b56e63b80a95)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/bf7264aa-a66b-4aa5-83ab-77a5db206ea8)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
![image](https://github.com/user-attachments/assets/dfcb15f1-bd85-4990-9a0a-8847eefbc716)

```
x=data[["Position","Level"]]
x.head()
```
![image](https://github.com/user-attachments/assets/86b328cd-117f-44c0-8085-fc816e65c8d0)

```
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
```

![image](https://github.com/user-attachments/assets/6fd88a12-07db-4f7b-a747-c2dc21dc7259)

```
r2=metrics.r2_score(y_test,y_pred)
r2
```

![image](https://github.com/user-attachments/assets/953d8c92-c4bb-4e72-b586-42723ef4c4ae)

```
dt.predict([[5,6]])
```

![image](https://github.com/user-attachments/assets/6672cc2f-1f76-4a84-ab56-0b78a1e4aeba)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

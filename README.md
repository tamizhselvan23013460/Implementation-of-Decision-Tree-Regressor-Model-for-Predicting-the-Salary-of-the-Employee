# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Start
2. Load the dataset from the CSV file.
3. Check the dataset for missing values and handle them if necessary.
4. Encode the categorical column (Position) using Label Encoding to convert text into numeric values.
5. Separate the dataset into:
6. Features (X): Position, Level
         Target (Y): Salary
7. Split the dataset into training and testing sets using train_test_split():
       Training set: 80%
       Testing set: 20%
8. Initialize the Decision Tree Regressor model.
9. Train the Decision Tree Regressor on the training data (x_train, y_train).
10. Predict the salary values for the testing data (x_test).
11. Evaluate the model performance:
12. Calculate Mean Squared Error (MSE) between actual and predicted salaries.
13. Calculate R² Score to check the goodness of fit.
14. Use the trained model to predict salary for new input values if needed.
15. End



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

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

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

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output :
### Preview datasets :

![EX_9_OUTPUT_1](https://github.com/user-attachments/assets/8146d5c6-ca79-4ff6-8cc2-3fb1a94d6862)

### Data Info :

![EX_9_OUTPUT_2](https://github.com/user-attachments/assets/917eaadf-2122-44bb-aa3a-8a7dbc3c5eab)


### Finding Null Values :

![EX_9_OUTPUT_3](https://github.com/user-attachments/assets/86ddfcf7-a738-4176-9efb-08f500c1dc5d)

### Label Encoder :

![EX_9_OUTPUT_4](https://github.com/user-attachments/assets/9f05b326-ce24-466b-ac9a-307b246d35d9)

### X-Initialize :
![EX_9_OUTPUT_5](https://github.com/user-attachments/assets/2d804af4-d9a9-415e-bed7-074f677edaf9)

### MSE :

![EX_9_OUTPUT_6](https://github.com/user-attachments/assets/82e0c0b2-8a7e-4f44-a8b6-ca7425691ecc)

### Value of R-Square :

![EX_9_OUTPUT_7](https://github.com/user-attachments/assets/ba30d13e-3954-4b49-b181-714968c69c68)

### Predict :

![EX_9_OUTPUT_8](https://github.com/user-attachments/assets/5cc3c0e5-71f5-4e13-ba7d-2cf065c026ba)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import joblib
mlflow.set_experiment("car-prediction")
os.makedirs("models",exist_ok=True)
df=pd.read_csv("data/updated.csv")

x=df.drop("price",axis=1)
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

with mlflow.start_run():
    model=LinearRegression()
    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)

    print(f'accuracy {accuracy}')

    mlflow.log_metric("accuracy",accuracy)
    joblib.dump(model,"models/model.pkl")

    mlflow.sklearn.log_model(model,registered_model_name="prediction-model")

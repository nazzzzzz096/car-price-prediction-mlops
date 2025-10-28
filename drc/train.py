from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import joblib



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri("https://dagshub.com/nazzzzzz096/car-price-prediction-mlops.mlflow")

mlflow.set_experiment("car-prediction")
df=pd.read_csv("data/updated.csv")

x=df.drop("price",axis=1)
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

with mlflow.start_run():
    model=LinearRegression()
    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)
    mse=mean_squared_error(y_test,y_pred)

    print(f'mean_squared_error {mse}')

    mlflow.log_metric("mean_squared_error",mse)
    print("model is saving")
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model,model_path)
    print("model save in the given path")

    mlflow.sklearn.log_model(model, "model")

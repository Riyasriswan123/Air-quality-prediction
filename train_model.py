import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data = pd.read_csv("air_quality.csv")

print("First 5 Rows:")
print(data.head())
X = data[["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]]
y = data["AQI"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
prediction = model.predict(X_test)

print("\nModel Performance")
print("---------------------------")
print("R2 Score :", r2_score(y_test, prediction))
print("MAE      :", mean_absolute_error(y_test, prediction))
print("MSE      :", mean_squared_error(y_test, prediction))
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nmodel.pkl has been created successfully!")
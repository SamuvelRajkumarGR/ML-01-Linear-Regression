import pandas as pd
from sklearn.linear_model import LinearRegression

data=pd.read_csv("dataset.csv")

X=data[["Hours_Studied"]]

y=data["Marks"]

model=LinearRegression()
model.fit(X,y)

hours=pd.DataFrame([[7]],columns=["Hours_Studied"])
predicted_marks=model.predict(hours)

print("Student Marks Prediction")
print("------------------------")
print("Hours Studied:", hours.iloc[0, 0])
print("Predicted Marks:",round(predicted_marks[0],2))
print("------------------------")

import matplotlib.pyplot as plt

plt.scatter(data["Hours_Studied"],data["Marks"])
plt.plot(data["Hours_Studied"],model.predict(X))

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")

plt.show()
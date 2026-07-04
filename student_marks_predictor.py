from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

current_folder = Path(__file__).parent
csv_file = current_folder / "student_marks.csv"

df = pd.read_csv(csv_file)

print(df)

X = df[["Hours"]]
y = df["Marks"]

print(X)
print(y)



model = LinearRegression()

model.fit(X, y)
predictions = model.predict(X)

plt.scatter(X, y, label="Actual Data")
plt.plot(X, predictions, color="red", label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression")

plt.legend()
plt.grid(True)

plt.savefig("output.png")
plt.show()

prediction = model.predict([[9]])

print(prediction)
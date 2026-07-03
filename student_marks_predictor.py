from pathlib import Path
import pandas as pd

current_folder = Path(__file__).parent
csv_file = current_folder / "student_marks.csv"

df = pd.read_csv(csv_file)

print(df)

X = df[["Hours"]]
y = df["Marks"]

print(X)
print(y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, y)
prediction = model.predict([[9]])

print(prediction)
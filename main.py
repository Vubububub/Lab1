import pandas as pd #https://www.kaggle.com/datasets/rishisukumar/student-screen-time-vs-cgpa-analysis-2026
df = pd.read_csv("D:/Student_Performance_2026.csv")
print(df.info())
print(df.isnull().sum())
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols = df.drop(columns=['student_ID','Gender']).columns
df[cols]= scaler.fit_transform(df[cols])
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
print(df.info())
print(df.head())
df.to_csv("Student_Performance_2026_processed.csv", index=False)
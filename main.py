import pandas as pd #https://www.kaggle.com/datasets/rishisukumar/student-screen-time-vs-cgpa-analysis-2026
df = pd.read_csv("D:/Student_Performance_2026.csv")
print(df.info())
print(df.isnull().sum())
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['previous_sem_CGPA']]= scaler.fit_transform(df[['previous_sem_CGPA']])
df[['current_sem_CGPA']]= scaler.fit_transform(df[['current_sem_CGPA']])
df[['attendance_percentage']]= scaler.fit_transform(df[['attendance_percentage']])
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
print(df.info())
print(df.head())
df.to_csv("Student_Performance_2026_processed.csv", index=False)
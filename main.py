import pandas as pd #https://www.kaggle.com/datasets/rishisukumar/student-screen-time-vs-cgpa-analysis-2026
import math
df = pd.read_csv("D:/Student_Performance_2026.csv")
print(df.info())
print(df.isnull().sum())
df=df.drop('previous_sem_CGPA', axis=1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['current_sem_CGPA']]= scaler.fit_transform(df[['current_sem_CGPA']])
df[['attendance_percentage']]= scaler.fit_transform(df[['attendance_percentage']])
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
print(df.info())
print(df.head())
df.to_csv("Student_Performance_2026_processed.csv", index=False)

X = df.drop(['current_sem_CGPA', 'student_ID'], axis=1)
y = df['current_sem_CGPA']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_predict_test = linear_model.predict(X_test)

r2 = r2_score(y_test, y_predict_test)
mse = mean_squared_error(y_test, y_predict_test)
mae = mean_absolute_error(y_test, y_predict_test)
print("R²:", r2, "RMSE:", math.sqrt(mse), "MAE:", mae)

threshold = df['current_sem_CGPA'].mean()
y_train_class = (y_train >= threshold)
y_val_class   = (y_val >= threshold)
y_test_class  = (y_test >= threshold)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

logreg_model = LogisticRegression(max_iter=100)
logreg_model.fit(X_train, y_train_class)
y_predict_test = logreg_model.predict(X_test)

accuracy = accuracy_score(y_test_class, y_predict_test)
cm = confusion_matrix(y_test_class, y_predict_test)
precision = precision_score(y_test_class, y_predict_test)
recall = recall_score(y_test_class, y_predict_test)
f1 = f1_score(y_test_class, y_predict_test)
report = classification_report(y_test_class, y_predict_test)
print("Accuracy:", accuracy,"Precision:", precision,"Recall:", recall,"F1:", f1)
print (report)
print("Confusion matrix:\n", cm)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
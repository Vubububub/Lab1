import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn import tree

df = pd.read_csv("D:/student_exam_performance_dataset.csv")
print(df.info())
print(df.isnull().sum())

df = df.drop(['age','student_id','grade_category'], axis=1)
df = pd.get_dummies(df, columns=['study_environment','parental_education','family_income'], drop_first=False)
df = pd.get_dummies(df, columns=['tutoring','pass_fail','gender','internet_access'], drop_first=True)

print(df.info())
df.to_csv("Student_Exam_Performance_2026_processed.csv", index=False)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df.drop(['final_exam_score','pass_fail_Pass'], axis=1)
y_train = train_df['final_exam_score']
X_test = test_df.drop(['final_exam_score','pass_fail_Pass'], axis=1)
y_test = test_df['final_exam_score']

reg_model = DecisionTreeRegressor(max_depth=6,min_samples_leaf=5,min_samples_split=10,random_state=42)
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("R²:", r2, "RMSE:", math.sqrt(mse), "MAE:", mae)

y_train_c = train_df['pass_fail_Pass']
y_test_c = test_df['pass_fail_Pass']

clf_model = DecisionTreeClassifier(max_depth=6,min_samples_leaf=5,min_samples_split=10, random_state=42)
clf_model.fit(X_train, y_train_c)
y_pred_class = clf_model.predict(X_test)
y_proba = clf_model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test_c, y_pred_class)
cm = confusion_matrix(y_test_c, y_pred_class)
precision = precision_score(y_test_c, y_pred_class)
recall = recall_score(y_test_c, y_pred_class)
f1 = f1_score(y_test_c, y_pred_class)
report = classification_report(y_test_c, y_pred_class)

print("Accuracy:", accuracy,"Precision:", precision,"Recall:", recall,"F1:", f1)
print (report)
print("Confusion matrix:\n", cm)


fpr, tpr, thresholds = roc_curve(y_test_c, y_proba)
roc_auc = auc(fpr, tpr)
print("\nROC-AUC:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.figure(figsize=(16,10))
tree.plot_tree(reg_model, filled=True, feature_names=X_train.columns)
plt.title("Decision Tree Regressor")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("D:/student_exam_performance_dataset.csv")
print(df.info())
print(df.isnull().sum())

df = df.drop(['age','student_id','grade_category','final_exam_score'], axis=1)
df = pd.get_dummies(df, columns=['study_environment','parental_education','family_income'], drop_first=False)
df = pd.get_dummies(df, columns=['tutoring','pass_fail','gender','internet_access'], drop_first=True)

print(df.info())
df.to_csv("Student_Exam_Performance_2026_processed.csv", index=False)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df.drop(['pass_fail_Pass'], axis=1)
y_train = train_df['pass_fail_Pass']
X_test = test_df.drop(['pass_fail_Pass'], axis=1)
y_test = test_df['pass_fail_Pass']

rf = RandomForestClassifier(n_estimators=200,oob_score=True,random_state=42,n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, rf_pred)
cm = confusion_matrix(y_test, rf_pred)
report = classification_report(y_test, rf_pred)

print("Accuracy:", accuracy,"OOB score:", rf.oob_score_)
print (report)
print("Confusion matrix:\n", cm)

ada = AdaBoostClassifier(n_estimators=200,learning_rate=0.5,random_state=42)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
ada_prob = ada.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, ada_pred)
cm = confusion_matrix(y_test, ada_pred)
report = classification_report(y_test, ada_pred)

print("Accuracy:", accuracy)
print (report)
print("Confusion matrix:\n", cm)

gb = GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=3,random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_prob = gb.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, gb_pred)
cm = confusion_matrix(y_test, gb_pred)
report = classification_report(y_test, gb_pred)

print("Accuracy:", accuracy)
print (report)
print("Confusion matrix:\n", cm)

fpr, tpr, thresholds = roc_curve(y_test, rf_prob)
roc_auc = auc(fpr, tpr)
print("\nROC-AUC:", roc_auc)

ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, ada_prob)
ada_roc_auc = auc(ada_fpr, ada_tpr)
print("\nROC-AUC:", ada_roc_auc)

gb_fpr, gb_tpr, gb_thresholds = roc_curve(y_test, gb_prob)
gb_roc_auc = auc(gb_fpr, gb_tpr)
print("\nROC-AUC:", gb_roc_auc)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC={roc_auc:.3f})")
plt.plot(ada_fpr, ada_tpr, label=f"AdaBoost (AUC={ada_roc_auc:.3f})")
plt.plot(gb_fpr, gb_tpr, label=f"Gradient Boosting (AUC={gb_roc_auc:.3f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.grid()
plt.show()
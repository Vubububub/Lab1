import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier


df = pd.read_csv("D:/spam_ham_dataset.csv")
print(df.info())
print(df.isnull().sum())
df=df.drop(columns=['Unnamed: 0','label'])


print(df.info())
X = df["text"]
y = df["label_num"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = Pipeline([ ("tfidf", TfidfVectorizer(lowercase=True,stop_words="english")),
                   ("nn", MLPClassifier(
                       hidden_layer_sizes=(64,32),
                        activation="relu",
                       solver="adam",
                       max_iter=5,
                       early_stopping=True,
                       random_state=42))])

model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred)


print (report)
print("Confusion matrix:\n", cm)

fpr, tpr, thresholds = roc_curve(y_test, prob)
roc_auc = auc(fpr, tpr)
print("\nROC-AUC:", roc_auc)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"(AUC={roc_auc:.3f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.grid()
plt.show()
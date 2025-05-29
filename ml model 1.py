import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
#data= pd.read_csv('final_with_smote.csv')
data= pd.read_csv('final_with_smote sscal.csv')
df= pd.DataFrame(data)
x = df.drop(columns=['Plant_Message_Type']) 
y = df["Plant_Message_Type"]  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200,class_weight="balanced", random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
print("Random Forest Classifier")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(x.shape[1]), importances[indices], align="center")
plt.xticks(range(x.shape[1]), x.columns[indices], rotation=10)
plt.xlim([-1, x.shape[1]])
plt.show()
#correlation matrix
import seaborn as sns
plt.figure(figsize=(20, 13))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
new_sample = [[0.3, 0.5, 1.2, 2.1, 0.7, 3.2, 25.5, 0.45, 1]] 

prediction = rf.predict(new_sample)
print("Predicted Plant Message Type:", prediction[0])


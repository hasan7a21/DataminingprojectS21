import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


heartDF = pd.read_csv('heart.csv')
print(f"HEAD OF DATA: \r\n{heartDF.head()}") #prints first 5 rows
print(f"TAIL OF DATA: \r\n{heartDF.tail()}") #prints last 5 rows 
print(f"Shape is: {heartDF.shape}") 

print(f"Info of Data: {heartDF.info()}") #prints summary of dataset

# Some statistics about the data in the dataset

targetCount = heartDF.target.value_counts()
heartLabels = "Doesn't have heart disease", "Has heart disease"
fig1,ax1 = plt.subplots()
ax1.pie(targetCount,explode=(0,0),labels=heartLabels,autopct='%1.1f%%') 
ax1.axis('equal')
#plt.show()

sexLabels = 'Male', 'Female'
fig1, ax1 = plt.subplots()
print(heartDF.sex.value_counts())
ax1.pie(heartDF.sex.value_counts(), explode=(0,0), labels=sexLabels, autopct='%1.1f%%')
ax1.axis('equal')
#plt.show()

fig = pd.crosstab(heartDF.sex, heartDF.target).plot(kind='bar')
plt.title('Heart disease count for each gender')
fig.set_xticklabels(labels=['Female', 'Male'], rotation=0)
plt.legend(['No heart disease','Heart disease'])
#plt.show()

plt.figure(figsize=(10,6))
plt.scatter(heartDF.age[heartDF.target==1], 
            heartDF.chol[heartDF.target==1], 
            c="purple")
plt.scatter(heartDF.age[heartDF.target==0], 
            heartDF.chol[heartDF.target==0], 
            c="orange")
plt.title("Heart Disease with respect to Age and Serum Cholestoral")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Serum Cholestoral in mg/dl")
#plt.show()

#for preprocesssing dataset
print(f"Preprocessing Check: \r\n{heartDF.isna().sum()}")
print(f"No Null Values, no need for preprocessing")

#since output of print heartDF.isna().sum() shows that there are no null values, we do not need to preprocess our data

dataColumn = heartDF.iloc[:, 0:-1] #rest of the columns
targetColumn = heartDF.iloc[:, -1] #target column

x_train, x_test, y_train, y_test = train_test_split(dataColumn, targetColumn, test_size = 0.25, random_state = 17)

#Naive Bayes Classification
naive=GaussianNB()
naive.fit(x_train,y_train)
predicted_naive=naive.predict(x_test)
cm_naive=metrics.confusion_matrix(y_test,predicted_naive)
print(f"Naive Bayes Predicted Target Column: {predicted_naive}")
print("\r\n")
accuracy_naive=metrics.accuracy_score(y_test,predicted_naive)

#SVM Classification
svmclass = svm.SVC(kernel = 'linear',random_state = 13)
svmclass.fit(x_train, y_train)
svcpred = svmclass.predict(x_test)
cm_svc = metrics.confusion_matrix(y_test,svcpred)
#svc_score = svc_clf.score(x_test, y_test)
print(f"SVC Predicted Target Column: {svcpred}")
print("\r\n")
svcaccuracy = metrics.accuracy_score(y_test, svcpred)

#Logistic Regression Classification
logreg = LogisticRegression(max_iter = 1000, random_state = 19)
logreg.fit(x_train, y_train)
logregpred = logreg.predict(x_test)
cm_logreg = metrics.confusion_matrix(y_test,logregpred)
print(f"Logistic Regression Predicted Target Column: {logregpred}")
print("\r\n")
yaccuracy2 = metrics.accuracy_score(y_test, logregpred)


#checking precision and recall of these classification models
naivePrecision = metrics.precision_score(y_test, predicted_naive)
naiveRecall = metrics.recall_score(y_test, predicted_naive)
print(f"Naive Bayes Accuracy: {accuracy_naive}")
print("Naive Bayes Precision:",naivePrecision)
print("Naive Bayes Recall:",naiveRecall)
print(f"Naive Bayes F1 Score: {2 * (naivePrecision * naiveRecall) / (naivePrecision + naiveRecall)}")
print("\r\n")

svPrecision =  metrics.precision_score(y_test, svcpred)
svRecall = metrics.recall_score(y_test, svcpred)
print(f"SVM Classification Accuracy: {svcaccuracy}")
print("SVM Classification Precision:",svPrecision)
print("SVM Classification Recall:", svRecall)
print(f"SVM Classification F1 Score: {2 * (svPrecision * svRecall) / (svPrecision + svRecall)}")
print("\r\n")

logregPrecision = metrics.precision_score(y_test, logregpred)
logregRecall = metrics.recall_score(y_test, logregpred)
print(f"Logistic Regression Accuracy: {yaccuracy2}")
print("Logistic Regression Precision:",logregPrecision)
print("Logistic Regression Recall:",logregRecall)
print(f"Logistic Regression F1 Score: {2 * (logregPrecision * logregRecall) / (logregPrecision + logregRecall)}")
print("\r\n")

#confusion matrix of these models
print(f"Naive Bayes Confusion Matrix: \r\n{cm_naive}\r\n")
print(f"SVM Confusion Matrix: \r\n{cm_svc}\r\n")
print(f"Logistic Regression Confusion Matrix: \r\n{cm_logreg}\r\n")
 
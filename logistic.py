# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:06:14 2024

@author: sahil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

claimants = pd.read_csv("C:/DS2/9_Logistic_Regression/claimants.csv")

# There are CLMAGE and LOSS are having continous data rest are
# verify the dataset, where CASENUM is not really useful so droping

c1 = claimants.drop('CASENUM',axis=1)
c1.head(11)
c1.describe()
# let us check whether there are null values
c1.isna().sum()
# There are several NULL values
# if we use dropna() function we will lose 290 data points
# hence we will go for imputation
c1.dtypes
mean_value = c1.CLMAGE.mean()
mean_value
# Now let us impute the same
c1.CLMAGE = c1.CLMAGE.fillna(mean_value)
c1.CLMAGE.isna().sum()
#hence all null values of CLMAGE has been fill by mean values
# for columns where there are discrete values. We will apply 
# MOde Imputation
mode_CLMSEX = c1.CLMSEX.mode()
mode_CLMSEX
c1.CLMSEX = c1.CLMSEX.fillna((mode_CLMSEX)[0])
c1.CLMSEX.isna().sum()

# CLMINSUR is also categrical data hence Mode inputation is used
mode_CLMINSUR = c1.CLMINSUR.mode()
mode_CLMINSUR
c1.CLMINSUR = c1.CLMINSUR.fillna((mode_CLMINSUR)[0])
c1.CLMINSUR.isna().sum()

# SEATBELT is categorical data hence go for mode imputation
mode_SEATBELT = c1.SEATBELT.mode()
mode_SEATBELT
c1.SEATBELT = c1.SEATBELT.fillna((mode_SEATBELT)[0])
c1.SEATBELT.isna().sum()

# Now the person we met an accident will hire the Atternev or not
# let us build the model
logit_model = sm.logit('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data = c1).fit()
logit_model.summary()
# in logistic regression  we do not have R square values, on;y check p=values
# SEATBELT is stastistically insignificant ignore and proceed
logit_model.summary2()
# here we are going to check AIC value, it stands for Akaike information
# is mathematical method for evaluation how well the model fit the data
# A lower score more the better model, Aic score is only useful in 
# with other AIC score for same dataset

#Now lets do prediction
pred = logit_model.predict(c1.iloc[:,1:])
# here we are applying all rows columns from 1,as columsn 0 is ATTORNEY
# terget value

# Let us check the perfomance of the model
fpr,tpr,thresholds = roc_curve(c1.ATTORNEY,pred)
#
#
#
#
optimal_idx = np.argmax(tpr-fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))

roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i),
                    'tpr': pd.Series(tpr, index=i),
                    '1-fpr': pd.Series(1-fpr,index=i),
                    'tf': pd.Series(tpr-(1-fpr), index=i),
                    'thresholds': pd.Series(thresholds, index=i)})


plt.plot(fpr,tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
roc.iloc[(roc.tf-0).abs().argsort()[:1]]
roc_auc = auc(fpr,tpr)
print("Area under the Curve: %f"% roc_auc)


fig, ax = pl.subplots()
pl.plot(roc['tpr'],color = 'red')
pl.plot(roc['1-fpr'],color = 'blue')
plt.xlabel("1-False positive rate")
plt.ylabel("True positive rate")
plt.title('Receiver operating characteristics')
ax.set_xticklabels([])



# filling all the cells with zeros
c1['pred'] = np.zeros(1340)
c1.loc[pred > optimal_threshold,"pred"]= 1

classification = classification_report(c1["pred"],c1["ATTORNEY"])
classification

#Splitting the data into train and test
train_data, test_data = train_test_split(c1,test_size=0.3)

# Model Building
model = sm.logit('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data = train_data).fit()

model.summary()

model.summary2()



# Now Time for Prediction
test_pred = logit_model.predict(test_data)

test_data["test_pred"] = np.zeros(402)
test_data.loc[test_pred > optimal_threshold,"test_pred"] = c1

# confusion Matrix
confusion_matrix = pd.crosstab(test_data.test_pred,test_data.ATTORNEY)
confusion_matrix

accuracy_test = (143+151)/(402)
accuracy_test


#clssification report
classification_test = classification_report(test_data["test_pred"],test_data["ATTORNEY"])
classification_test

fpr,tpr,thresholds = metrics.roc_curve(test_data["ATTORNEY"],test_pred)

#plot the curve
plt.plot(fpr,tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
#Area under the curve
roc_auc_test = metrics.auc(fpr,tpr)
roc_auc_test

#prediction onthe train data
train_pred = logit_model.predict(train_data)
train_data["trian_pred"] = np.zeros(938)
train_data.loc[train_pred > optimal_threshold,"train_pred"] = 1

# confusion matrix
confusion_matrix = pd.crosstab(train_data.train_pred,train_data.ATTORNEY)
confusion_matrix

accuracy_train = (315+347)/(938)
accuracy_train

#CLASSIFICATION REPORT
classification_train = classification_report(train_data["trian_pred"],train_data["ATTORNEY"])

import os
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
#Import Libraries for decision tree
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


os.getcwd()
os.chdir('/Users/maneeshagvs/Documents/datasets')
train = pd.read_csv('Train_dataset.csv')
test = pd.read_csv('Test_dataset.csv')
test_id = test.ID_code
Y =train['target']
X = train.drop(columns=['target','ID_code'])
test = test.drop(columns=['ID_code'])
error_table = pd.DataFrame()
def missingvalues(df):
    count = 0
    for i in df.columns:
        if(df[i].isnull().sum() != 0):
            print("The missing values are present for",df[i].isnull().sum(),i)
            count+=1
    if(count == 0):
        print("There are no missing values present")
    return count

def constants(df):
    count = 0 
    for i in df.columns:
        if(len(df[i].unique()) == 1):
            print("The value is constant",len(df[i].unique()),i)
            count+=1
    if(count == 0):
        print("There are no constant present")
    return count

print("missing values train count : ",missingvalues(train))
print("missing values test count : ",missingvalues(test))
print("constant values count : ",constants(train))
print("constant values count : ",constants(test))


def Remove_duplicate():
    remove = []
    cols = train.columns
    for i in range(len(cols)-1):
        v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    
def accuracy_cal(i,actual,predicted):
    CM = pd.crosstab(actual,predicted)
    #let us save TP, TN, FP, FN
    TN = CM.iloc[0,0]
    FN = CM.iloc[1,0]
    TP = CM.iloc[1,1]
    FP = CM.iloc[0,1]
    error_table.loc['Accuracy',i]=(((TP+TN))/(TP+TN+FP+FN))
    error_table.loc['Precision',i]=((TP)/(TP+FP))
    error_table.loc['FalsePositiveRate',i]=((FP)/(FP+TN))
    error_table.loc['Specificity',i]= 1 - error_table.loc['FalsePositiveRate',i]
    error_table.loc['Recall or Sensitivity',i]=((TP)/(TP+FN))
    print("Accuracy , FalsePositive:",error_table.loc['Accuracy',i],error_table.loc['FalsePositiveRate',i])
    auc = metrics.roc_auc_score(actual,predicted)
    error_table.loc['ROC',i]=auc
    
Remove_duplicate()
    
train0 = train[ train['target']==0 ].copy()
train1 = train[ train['target']==1 ].copy()


# In[3]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
s = sns.countplot(train["target"], 
                   order = train["target"].value_counts().index)
for p, label in zip(s.patches, train["target"].value_counts()):
    s.annotate(label, (p.get_x()+0.375, p.get_height()+0.15))

plt.show()


# In[60]:


train.head()


# In[61]:


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y, test_size=0.3, random_state=40)
RF_model = RandomForestClassifier(n_estimators = 20).fit(X1_train, Y1_train)
RF_Predictions = RF_model.predict(X1_test)
accuracy_cal('RandomForest',Y1_test,RF_Predictions)


# In[62]:


train.info()


# In[63]:


(pd.Series(RF_model.feature_importances_, index=X.columns).nlargest(20).plot(kind='bar'))


# In[64]:


feature_importance = pd.DataFrame(pd.Series(RF_model.feature_importances_, index=X.columns).nlargest(20))
train_imp = train.loc[:,feature_importance.index]
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = train_imp.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[65]:


plt.figure(figsize=(8,5))
for i in train_imp.columns:
    sns.distplot(train0[i], label = 't=0')
    sns.distplot(train1[i], label = 't=1')
    plt.legend()
    plt.xlabel(i)
    plt.show()


# In[66]:


train[train.columns[2:]].mean().plot('hist');plt.title('Mean Frequency');


# In[67]:


###################################### NAIVE_BAYES ###################################################################
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
#Naive Bayes implementation
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y, test_size=0.3, random_state=40)

NB_model = GaussianNB().fit(X2_train, Y2_train)

#predict test cases
NB_Predictions = NB_model.predict(X2_test)

accuracy_cal('NaviesBayes',Y2_test,NB_Predictions)


# In[68]:


#################################### DECISION TREE ##################################################################
#Decision Tree
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X, Y, test_size=0.3,random_state=40)

C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X4_train, Y4_train)

#predict new test cases
C50_Predictions = C50_model.predict(X4_test)
accuracy_cal('DecisionTree',Y4_test,C50_Predictions) 


# In[69]:


# split X and y into training and testing sets

X5_train,X5_test,Y5_train,Y5_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[70]:


################################## LOGISTIC REGRESSION ##############################################################

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X5_train,Y5_train)

#
Y5_pred=logreg.predict(X5_test)

accuracy_cal('LogisticRegression',Y5_test,Y5_pred) 


# In[71]:


cnf_matrix = confusion_matrix(Y5_test,Y5_pred)


# In[72]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[73]:


y_pred_proba = logreg.predict_proba(X5_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y5_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y5_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[74]:


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y, test_size=0.3, random_state=40)
## # Feature selection
clf = ExtraTreesClassifier(random_state=1729)
selector = clf.fit(X1_train, Y1_train)


# In[75]:


####################################XGBOOST ############################################################################

fs = SelectFromModel(selector, prefit=True)
X1_train = fs.transform(X1_train)
X1_test = fs.transform(X1_test)
test = fs.transform(test)

print(X1_train.shape, X1_test.shape, test.shape)

## # Train Model
# classifier from xgboost
m2_xgb = xgb.XGBClassifier(n_estimators=110, nthread=-1, max_depth = 4,seed=1729)
m2_xgb.fit(X1_train,Y1_train, eval_metric="auc", verbose = False,eval_set=[(X1_test, Y1_test)])

# calculate the auc score
print("Roc AUC:", metrics.roc_auc_score(Y1_test, m2_xgb.predict_proba(X1_test)[:,1],
             average='macro'))

auc = metrics.roc_auc_score(Y1_test, m2_xgb.predict_proba(X1_test)[:,1])
plt.plot(fpr,tpr,label="data 1,auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[80]:


accuracy_cal('XGBoost',Y1_test,m2_xgb.predict_proba(X1_test)[:,1]) 


# In[76]:


## final Submission
probs = m2_xgb.predict_proba(test)
submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
submission.to_csv("submission.csv", index=False)


# In[85]:


###over error metrics
print(error_table)


# In[93]:


# In[ ]:





# In[ ]:





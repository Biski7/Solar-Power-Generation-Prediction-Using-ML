
######################################################
#
# imports for the program
#
######################################################

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_predict
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier as ovr
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble import VotingClassifier


######################################################
#
# reading the dataset, creating dataframe
#
######################################################
df = pd.read_csv('C:/Users/bisha/Desktop/3rd try/BT.csv')


######################################################
#
# Dividing the ouput into 5 lables
#
######################################################

df['Power Generated'] = np.where(df['Power Generated'].between(0,100), 0, df['Power Generated'])
df['Power Generated'] = np.where(df['Power Generated'].between(100,10000), 1, df['Power Generated'])
df['Power Generated'] = np.where(df['Power Generated'].between(10000,20000), 2, df['Power Generated'])
df['Power Generated'] = np.where(df['Power Generated'].between(20000,30000), 3, df['Power Generated'])
df['Power Generated'] = np.where(df['Power Generated'].between(30000,40000), 4, df['Power Generated'])


######################################################
#
# Checking the balance of the classes
#
######################################################
value_0 = df['Power Generated'].value_counts()[0]
value_1 = df['Power Generated'].value_counts()[1]
value_2 = df['Power Generated'].value_counts()[2]
value_3 = df['Power Generated'].value_counts()[3]
value_4 = df['Power Generated'].value_counts()[4]

df_0 = df[df.iloc[:,-1]==0]
df_1 = df[df.iloc[:,-1]==1]
df_2 = df[df.iloc[:,-1]==2]
df_3 = df[df.iloc[:,-1]==3]
df_4 = df[df.iloc[:,-1]==4]


######################################################
#
# Upsampling the minority classes
#
######################################################

df_2_upsampled = resample(df_2, replace=True, n_samples=738)
df_3_upsampled = resample(df_3, replace=True, n_samples=738)
df_4_upsampled = resample(df_4, replace=True, n_samples=738)


######################################################
#
# Downsampling the majority class
#
######################################################
df_0_downsampled = resample(df_0, replace=True,n_samples=738)


######################################################
#
# Combining the data after resampling 
#
######################################################

df_new = pd.concat([df_0_downsampled, df_1, df_2_upsampled, df_3_upsampled, df_4_upsampled])
df = df_new

######################################################
#
# Separating the data & target information
#
######################################################

X = df.iloc[:, :-1]  # we only take the first two features.
y = df.iloc[:, -1]


######################################################
#
# Using Label Encoder
#
######################################################

le = LabelEncoder()
df['Is Daylight'] = le.fit_transform(df['Is Daylight'])


######################################################
#
# Data splitting for training and testing
#
######################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


######################################################
#
# Standardizing
#
######################################################

sc = StandardScaler()
sc.fit(X_train)
sc.fit(X_test)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

######################################################
#
# Using LDA
#
######################################################

lda = LDA(n_components=2)      
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)


######################################################
#
# Creating LR, SVM and Decision Tree Classifier
#
######################################################
clf1 = LogisticRegression(penalty='l2', C= 1.0, random_state=1, solver = 'lbfgs')                                                
clf2 = SVC(C= 100, kernel= 'rbf', gamma = 1000, random_state=1, probability = True)                                                        
clf3 = DecisionTreeClassifier(max_depth=20, criterion='entropy', random_state = 1)

######################################################
#
# Implementing Pipeline
#
######################################################

pipe1 = Pipeline([['sc', StandardScaler()],['clf', clf2]])
pipe2 = Pipeline([['sc', StandardScaler()],['clf', clf2]])
clf_labels = ['Logistic regression', 'SVM', 'Decision tree']
              

######################################################
#
# Implementing Majority Voting
#
######################################################

mv_clf = VotingClassifier(estimators=[('lr', pipe1), ('svm', pipe2), ('dt', clf3)], voting='soft')
clf_labels += ['Majority voting - ArgMax']
all_clf = [pipe1, pipe2, clf3, mv_clf]


######################################################
#
# Writing best score, best parameter and accuracy to a file
#
######################################################

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train_lda, y=y_train, cv=10, scoring='roc_auc_ovr')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

mv_clf.fit(X_train_lda, y_train)



######################################################
#
# Plotting the classification and saving it
#
###################################################### 
X_combined=np.vstack((X_train_lda,X_test_lda))
y_combined=np.hstack((y_train,y_test))                 
plot_decision_regions(X_combined,
                      y_combined,
                      clf=mv_clf, legend=2)
# plt.savefig('classification_majority_voting.png')



######################################################
#
# ROC AUC curve
#
######################################################

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
y_pred_proba=mv_clf.predict_proba(X_test_lda)
fpr = {}
tpr = {}
thresh ={}

number_of_classifier = 3
for i in range(number_of_classifiers):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_proba[:,i], pos_label=i)
    
  
# ######################################################
# #
# # Plotting the ROC_AUC curve and saving it
# #
# ######################################################

# plt.plot(fpr[0], tpr[0], linestyle='-',color='orange', label='LR')
# plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='SVM')
# plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Decision Trees')
# plt.plot([0,1], [0,1], linestyle = '--', color = (0.6, 0.6, 0.6), label = 'Random Guessing')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive rate')
# plt.legend(loc='best')
# plt.savefig('ROC')
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    y_pred = clf.fit(X_train_lda, y_train).predict_proba(X_test_lda)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray',
linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
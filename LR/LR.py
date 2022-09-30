######################################################
#
# imports for the program
#
######################################################

import pandas as pd
import numpy as np
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
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_predict
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.metrics import recall_score, precision_score, f1_score


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

X = df.iloc[:, 5:-1] 
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
# Implementing Pipeline 
#
######################################################

pipe_lr = make_pipeline( LogisticRegression(random_state=1,solver='lbfgs'))



######################################################
#
# Parameter list for Grid Search 
#
######################################################

inverse_lambda = [0.001, 0.01,0.1,1,2,10,100]


######################################################
#
# Creating a dictionary for Parameter C
#
######################################################

param_grid = {"logisticregression__C":inverse_lambda}


######################################################
#
# Implementing Cross validation, GridSearch and fitting data
#
######################################################

cross_validation = list(StratifiedKFold(n_splits=10, random_state=1, shuffle=True).split(X_train_lda, y_train))


######################################################
#
# Implementing GridSearch and fitting data
#
######################################################

gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='accuracy',cv=cross_validation, n_jobs=-1, refit=True, return_train_score=True)
gs = gs.fit(X_train_lda, y_train)


######################################################
#
# Determining best score and best parameters
#
######################################################

best_score = gs.best_score_
param = gs.best_params_
best_param = param['logisticregression__C']
y_pred = cross_val_predict(pipe_lr, X_test_lda, y_test)


######################################################
#
# Printing the best test and train accuracy
#
######################################################
print(' Training Accuracy: %.5f' % gs.score(X_train_lda, y_train))
print('\n')
print(' Test Accuracy: %.5f' % gs.score(X_test_lda, y_test))
print('\n')


######################################################
#
# Printing the recall, precision and f1 score 
#
######################################################
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
f1_score = f1_score(y_test, y_pred, average='macro')

print('Recall : ', recall)
print('Precision: ', precision)
print('F1_score: ', f1_score)


######################################################
#
# Writing best parameters and accuracy to file
#
######################################################

with open("BT_LR.txt", "w") as f:
  f.write('Best Score %.5f' %best_score)
  f.write('\n')
  f.write('Best Param C %.5f: '  %best_param) 
  f.write('\n')
  f.write('Training Accuracy: %.5f' % gs.score(X_train_lda, y_train))
  f.write('\n')
  f.write('Test Accuracy: %.5f' % gs.score(X_test_lda, y_test))
  f.write('\n')
  f.write('Recall: %.5f '  % recall) 
  f.write('\n')
  f.write('Precision: %.5f' % precision)
  f.write('\n')
  f.write('f1_score: %.5f' % f1_score)
  f.write('\n')
  f.write('*****************************************************************\n\n')



######################################################
#
# Plotting the classification and saving it
#
###################################################### 

X_combined=np.vstack((X_train_lda,X_test_lda))
y_combined=np.hstack((y_train,y_test))                 
plot_decision_regions(X_combined,
                      y_combined,
                      clf=gs, legend=2)
plt.legend(loc='best')
plt.title('Classification - LR')
plt.savefig('Logistic_Regression_plot.png')



######################################################
#
# Plotting the confusion matrix and saving
#
######################################################

cm = confusion_matrix(y_test, y_pred)
labels=['0','1','2','3','4']
plt.figure(figsize=(9,9))
sns.heatmap(cm, cbar=False, xticklabels=labels, yticklabels=labels, fmt='d', annot=True, cmap=plt.cm.Blues)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Confusion Matrix.png')
plt.show()


######################################################
#
# ROC AUC curve
#
######################################################

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
y_pred_proba=gs.predict_proba(X_test_lda)
fpr = {}
tpr = {}
thresh ={}

number_of_classes = 5
for i in range(number_of_classes):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_proba[:,i], pos_label=i)
    
  
######################################################
#
# Plotting the ROC_AUC curve and saving it
#
######################################################

plt.plot(fpr[0], tpr[0], linestyle='-',color='orange', label='0-100')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='100-1K')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='1K-2K')
plt.plot(fpr[3], tpr[3], linestyle='--',color='yellow', label='2K-3K')
plt.plot(fpr[4], tpr[4], linestyle='--',color='red', label='3K-4K')
plt.plot([0,1], [0,1], linestyle = '--', color = (0.6, 0.6, 0.6), label = 'Random Guessing')
plt.title('LR ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('ROC')









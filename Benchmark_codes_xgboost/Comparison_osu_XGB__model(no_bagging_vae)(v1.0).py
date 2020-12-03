
# coding: utf-8

# # Load packages and use tensorflow as backend

# In[4]:


######################################################
# Install a pip package in the current Jupyter kernel
# import system level packages
#!{sys.executable} -m pip install numpy
#!{sys.executable} -m pip install requests
#import requests
######################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import tensorflow as tf
import keras
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
from keras import utils
import pydot
import graphviz
from keras.utils import plot_model
from keras_tqdm import TQDMNotebookCallback
#from .tqdm_callback import TQDMNotebookCallback
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.python.client import device_lib

########################################################
#importing necessary libraries for scikit-learn

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
#from sklearn.grid_search import GridSearchCV   #!!!the grid search package that has issue, dont use it
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from scipy import interp
from scipy import stats

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing


# In[5]:


##################################################
#test tensorflow, remember to change the kernel
#using kernel that supports GPU computing
#simple test to confirm tensorflow is actually working
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print("10 + 32 = ", sess.run(a + b))
#manually set the random seed to define a replication
r_seed = 42

#manually set the number for cross validation
num_cv = 5

print("current random seed is: ", r_seed)


# # check the system information

# In[6]:


#######################################################################################################
#check the system information, check if cuda and gpu computing for tensorflow is installed properly
#######################################################################################################
print("whether tensorflow is built with cuda: ", tf.test.is_built_with_cuda())
print("whether gpu computing is available for tensorflow: ", tf.test.is_gpu_available())
print("using keras version: ", keras.__version__)
print("using tensorflow version: ", tf.__version__)
print("using python version: ", sys.version)

print("\n")
print("Device details:\n", device_lib.list_local_devices())
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


# # Reading files

# In[7]:


##########################################################
# Reading files/documents                                #
# !!! Need to change to file location on your local drive#
##########################################################

#vae file
compress_path = 'counts_data/vae_compressed_wLabels/encoded_BRCA_rnaseq_vae_binaryLabels(L2,perSp,a1.0,unlabel,0.2_var,10LF3kX4,z100,minmax,ep700).txt'

#vae with grade file
#compress_path = 'counts_data/vae_compressed_with_grade/TCGA_4cancers_(BLCA_perSP_minmax_3labels_6LF6k_z200)_with_grade.txt'
# vae with stage file
#compress_path = 'counts_data/vae_compressed_with_stage/TCGA_4cancers_(PAAD_perSP_minmax_3labels_6LF6k_z50)_with_stage.txt'


#open(path).readline()
#gene expression RNAseq, Batch effects normalized mRNA data

og_data = pd.read_csv(compress_path, sep = "\t", index_col = 0)
og_data = og_data.dropna(axis='columns')
#ExprAlldata.columns = ["Gene", "Counts"]
print("dimension of the input data: ", og_data.shape)
og_data.head(5)


# ## Number of cases in each category

# In[8]:


df_count = og_data.groupby('response_group')['Ensembl_ID'].nunique()
print(df_count)
#df_count.nlargest(10)


# In[9]:


###################################################
#store the raw data, and use ensembl id as index
##################################################
df_raw = og_data.iloc[:, 0:]
df_raw = df_raw.set_index('Ensembl_ID')

#notice the last column is the response_group
#df_raw.shape
df_raw.head(3)


# In[10]:


#####################################!#################################
#here begins full data
################################
#full data, 4 labels analysis
#Complete Response    21
#Clinical Progressive Disease    10
#Radiographic Progressive Disease     7
#Stable Disease     7

#features
df_raw_coln = len(df_raw.columns)
X = df_raw.iloc[:,0:(df_raw_coln-1)]
X = X.values

#label/target
y = df_raw.loc[:, 'response_group']
y = y.values

#!!!!!!!
#check to confirm the last column is not response group, only y contains response group information
col = X.shape[1]
#print(X[:,(col-1)])

class_names = np.unique(y)
print("unique labels from y: ", class_names)


# # Load necessary methods

# In[11]:


#########################################################################################
#plot confusion matrix
#inputs: cm, confusion matrix from cross_val_predict
#        normalize, whether to use normalize for each sections 
#        title, input the title name for the figure
#        cmap, color map using blue as default
#output: a confusion matrix plot with true label as y axis, and predicted label as x axis
#########################################################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[12]:


##############################################################
#plot area under curve graph
#input: actual, true labels/target without one hot encoding
#       probs, predicted probabilities
#       n_classes, number of unique classes in target
#       title, input the title name for the figure
#output: a roc curve plot for multi class task
###############################################################
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle

def plot_multiclass_roc_auc(actual, probs, n_classes, title = 'multi-class roc'):
    lb = LabelBinarizer()
    lb.fit(actual)
    actual = lb.transform(actual)
    y_prob = probs
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(actual[:, i], y_prob[:, i])
        #please notice the difference between auc() and roc_auc_score()
        #also auc() only works on monotonic increasing or monotonic
        #decreasing input x
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        colors = cycle(['blue', 'red', 'green', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
        label='ROC curve of class {0} (area = {1:0.10f})'
            ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data using '+title)
    plt.legend(loc="lower right")
    #commented thus being able to use fig save function
    #plt.show()


# In[13]:


#######################################################
#Random search CV method
#and
#Multi class roc_auc score method
########################################################
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer, roc_auc_score

###########################################################################################
#Multi class roc_auc score method
#input: y_test, true labels from test fold
#       y_prob, predicted probability on test fold
#       average, string, [None, ‘micro’, ‘macro’ (default), ‘samples’, ‘weighted’]
#                'macro': Calculate metrics for each label, and find their unweighted mean. 
#                This does not take label imbalance into account.
#                'weighted': Calculate metrics for each label, and find their average, 
#                weighted by support (the number of true instances for each label).
#output: auroc value for each class
#multiclass_score, an implemented scoring method for multi class task
#!!!
#Notice that by default,needs_proba : boolean, default=False
#thus the multiclass_score will try to use the predicted label instead of predicted probability to calculate roc
#which is not correct, and will causing the tuning process to not find the best parameters
##############################################################################################
def multiclass_roc_auc_score(y_test, y_prob, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    #y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_prob, average=average)

#!!!
#Notice that by default,needs_proba : boolean, default=False
#thus the multiclass_score will try to use the predicted label instead of predicted probability to calculate roc
#which is not correct, and will causing the tuning process to not find the best parameters
multiclass_score = make_scorer(multiclass_roc_auc_score, needs_proba = True)

###############################################################################################
#Binary class roc auc score method
#input: y_true, true labels from test fold
#       y_score, predicted probability on test fold
#       average, string, [None, ‘micro’, ‘macro’ (default), ‘samples’, ‘weighted’]
#                'macro': Calculate metrics for each label, and find their unweighted mean. 
#                This does not take label imbalance into account.
#                'weighted': Calculate metrics for each label, and find their average, 
#                weighted by support (the number of true instances for each label).
#output: auroc value for each class
#############################################################################################
def binary_class_roc_auc_score(y_true, y_score, average="weighted"):

    return roc_auc_score(y_true, y_score, average=average)

binaryclass_score = make_scorer(binary_class_roc_auc_score, needs_threshold = True)

###################################################################################
#Random search CV method
#input: est, input estimator/classifier
#       p_distr, the grid of parameters to search on
#       nbr_iter, numbers of iteration on random search
#       X, feature, y, true labels
#output: ht_estimator, best estimator based on mean value of all folds
#        ht_params, best parameters
#
################################################################################################
def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    #seed = 42
    cv = StratifiedKFold(n_splits = 3, random_state = r_seed, shuffle = True)
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr, scoring=multiclass_score,
                                  n_jobs=-1, n_iter=nbr_iter, cv=cv, return_train_score = True, verbose =10)
    #CV = Cross-Validation ( here using Stratified KFold CV) #,random_state = seed
    start = time()
    rdmsearch.fit(X,y)
    print('hyper-tuning time : %d seconds' % (time()-start))
    start = 0
   # ht_train_mean = rdmsearch.cv_results_['mean_train_score']
   # ht_train_std = rdmsearch.cv_results_['std_train_score']
   # ht_test_mean_sp0 = rdmsearch.cv_results_['split0_test_score']
   # ht_test_mean_sp1 = rdmsearch.cv_results_['split1_test_score']
   # ht_test_mean_sp2 = rdmsearch.cv_results_['split2_test_score']
    #ht_train_mean_sp3 = rdmsearch.cv_results_['split3_train_score']
    #ht_train_mean_sp4 = rdmsearch.cv_results_['split4_train_score']
    #ht_best_loc = np.where(rdmsearch.cv_results_['rank_test_score'] == 1)
    
    ht_cv_results = rdmsearch.cv_results_
    ht_estimator = rdmsearch.best_estimator_
    ht_params = rdmsearch.best_params_
    #ht_score = rdmsearch.best_score_
    
    return ht_estimator, ht_params, ht_cv_results


# # Grid search

# In[14]:


###########################################################
#Grid search Tune learning rate, n_estimators, and booster
#
##########################################################
param_test_this_loop = {
 'learning_rate':[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
 'n_estimators':[i for i in range(1,40)],
 'booster':['gbtree'],
 #'booster':['gbtree','gblinear','dart'],
    'silent':[True],
    'random_state':[r_seed]
}
cv = StratifiedKFold(n_splits = num_cv, random_state = r_seed, shuffle = True)

gsearch_loop1 = GridSearchCV(estimator = XGBClassifier(booster = 'gbtree'), 
 param_grid = param_test_this_loop, scoring="roc_auc",n_jobs=-1,iid=False, cv=cv,verbose=10)
gsearch_loop1.fit(X,y)
gsearch_loop1.grid_scores_, gsearch_loop1.best_params_, gsearch_loop1.best_score_


# In[15]:


##########################################################
#output grid scores, and save to a file
#
#xgb_grid_scores = pd.DataFrame(gsearch1.grid_scores_)
#xgb_grid_file = os.path.join("Tuning_insights", "xgb_grid_socres(lr&n_estimators).tsv")
#xgb_grid_scores.to_csv(xgb_grid_file, sep='\t')

# Test to make sure the parameters are correct
#gsearch_loop1.best_params_
#gsearch_loop1.best_params_["learning_rate"]
#gsearch_loop1.best_params_["booster"]
#gsearch_loop1.best_params_["n_estimators"]


# In[16]:


#################################################
#Grid search Tune max_depth and min_child_weight
#default
#################################################
param_test_this_loop = {
 'max_depth':[i for i in range(1,10)],
 'min_child_weight':[i for i in range(0,10)],
    'silent':[True],
    'random_state':[r_seed]
}
cv = StratifiedKFold(n_splits = num_cv, random_state = r_seed, shuffle = True)

gsearch_loop2 = GridSearchCV(estimator = XGBClassifier(learning_rate =gsearch_loop1.best_params_["learning_rate"], 
                                                  n_estimators=gsearch_loop1.best_params_["n_estimators"], 
                                                  booster = gsearch_loop1.best_params_["booster"]), 
param_grid = param_test_this_loop, scoring="roc_auc",n_jobs=-1,iid=False, cv=cv,verbose=10)
gsearch_loop2 .fit(X,y)
gsearch_loop2 .grid_scores_, gsearch_loop2 .best_params_, gsearch_loop2 .best_score_


# In[17]:


##########################################
#Grid search Tune subsample and colsample
#
##########################################
param_test_this_loop = {
             'subsample':[i/100.0 for i in range(10,110,10)],
             'colsample_bytree':[i/100.0 for i in range(10,110,10)],
             
    'silent':[True],
    'random_state':[r_seed]
}
cv = StratifiedKFold(n_splits = num_cv, random_state = r_seed, shuffle = True)

gsearch_loop3 = GridSearchCV(estimator = XGBClassifier(learning_rate =gsearch_loop1.best_params_["learning_rate"], 
                                                      n_estimators=gsearch_loop1.best_params_["n_estimators"], 
                                                      booster = gsearch_loop1.best_params_["booster"],
                                                      max_depth =gsearch_loop2.best_params_["max_depth"],
                                                      min_child_weight=gsearch_loop2.best_params_["min_child_weight"]), 
  param_grid = param_test_this_loop, scoring="roc_auc",n_jobs=-1,iid=False, cv=cv,verbose=10)
gsearch_loop3.fit(X,y)
gsearch_loop3.grid_scores_, gsearch_loop3.best_params_, gsearch_loop3.best_score_


# In[18]:


##########################################
#Grid search Tune subsample and colsample
#
##########################################
param_test_this_loop = {
             'reg_alpha':[i for i in range(0,3)],
             'reg_lambda':[i for i in range(1,100)],
    'silent':[True],
    'random_state':[r_seed]
}
cv = StratifiedKFold(n_splits = num_cv, random_state = r_seed, shuffle = True)

gsearch_loop4 = GridSearchCV(estimator = XGBClassifier(learning_rate =gsearch_loop1.best_params_["learning_rate"], 
                                                      n_estimators=gsearch_loop1.best_params_["n_estimators"], 
                                                      booster = gsearch_loop1.best_params_["booster"],
                                                      max_depth =gsearch_loop2.best_params_["max_depth"],
                                                      min_child_weight=gsearch_loop2.best_params_["min_child_weight"], 
                                                    subsample = gsearch_loop3.best_params_["subsample"],
                                                 colsample_bytree = gsearch_loop3.best_params_["colsample_bytree"]), 
  param_grid = param_test_this_loop, scoring="roc_auc",n_jobs=-1,iid=False, cv=cv,verbose=10)
gsearch_loop4.fit(X,y)
gsearch_loop4.grid_scores_, gsearch_loop4.best_params_, gsearch_loop4.best_score_


# In[19]:


#################################################################
# Use random search to train a XGBoost model
# with bagging
###################################################
#est = XGBClassifier()
#param_test_rand = {
#            'learning_rate':[0.4,0.5,0.7],
#            'n_estimators':[i for i in range(12,20)],
#            #'n_estimators':[i for i in range(1,10)],
#            'max_depth':[i for i in range(12,20)],
#            'min_child_weight':[i for i in range(0,5)],
#             'booster':['gbtree','gblinear','dart'],
#             'subsample':[i/100.0 for i in range(70,90)],
#             'colsample_bytree':[i/100.0 for i in range(40,60)],
#             'reg_alpha':[i for i in range(0,5)],
#             'reg_lambda':[i for i in range(100,120)],
#             'silent':[True],
#             'random_state':[42]   

 #'learning_rate':[0.1, 0.01, 0.001, 0.0001],
 #'n_estimators':[i for i in range(1,1000,10)],
 #'n_estimators':[i for i in range(1,10)],
 #'max_depth':[i for i in range(1,30)],
 #'min_child_weight':[i for i in range(0,10)],
 #'booster':['gbtree','gblinear','dart'],
 #'subsample':[i/100.0 for i in range(1,100,1)],
 #'colsample_bytree':[i/100.0 for i in range(1,100,1)],
 #'reg_alpha':[i for i in range(0,10)],
 #'reg_lambda':[i for i in range(0,100)],
 #'silent':[True],
 #'random_state':[42]
#}

#gb_estimator, gb_parameters, gb_cv_results = hypertuning_rscv(est, param_test_rand, 6000, X, y)

#print(gb_parameters)

#print('Hyper-tuned model score :')
#print(gb_ht_score*100)

#print('Hyper-tuned model training mean :')
#print(gb_train_mean)
#print('Hyper-tuned model training std :')
#print(gb_train_std)

#print('Hyper-tuned model training mean for split1 :')
#print(gb_train_mean_sp0)

#print('Hyper-tuned model training mean for split2 :')
#print(gb_train_mean_sp1)

#print('Hyper-tuned model training mean for split3 :')
#print(gb_train_mean_sp2)


# In[20]:


#print(gb_cv_results["split1_train_score"])
#print(gb_cv_results["split1_test_score"])


# # Training the XGBoost model with the best parameters

# In[21]:


###########################
# training a XGBoost model
##########################
# if using the randomSearch method
#xgb = gb_estimator

# if using GridSearch method
xgb = gsearch_loop4.best_estimator_
cv = StratifiedKFold(n_splits = num_cv, random_state = r_seed, shuffle = True)

##!!!!
#notice that mean of auroc of each fold is different from the auroc calculated by all the predicted probability
#svm_scores = cross_val_score(svm_model_linear, X, y, cv = cv, scoring=multiclass_score)
y_xgb_prob = cross_val_predict(xgb, X, y, cv = cv, method = 'predict_proba')

# calculate the auroc by directly using the multiclass_roc_auc_score scorer
#xgb_multiclass_auroc = multiclass_roc_auc_score(y, y_xgb_prob, average="weighted")

# calculate the auroc by directly using the binaryiclass_roc_auc_score scorer
xgb_multiclass_auroc = binary_class_roc_auc_score(y, y_xgb_prob[:,1], average="weighted")

#print(xgb)
#print("Auroc across all folds: %0.5f" % (xgb_multiclass_auroc))


# In[22]:


#print("Predicted labels are:")
#print(xgb_pred)

#output predicted labels
#XGboost
#xgb_pred_df = pd.DataFrame(xgb_pred)
#xgb_pred_file = os.path.join("predicted_labels", "xgb_pred.tsv")
#xgb_pred_df.to_csv(xgb_pred_file, sep='\t')


# In[23]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

cv = StratifiedKFold(n_splits = num_cv, random_state = r_seed)
y_xgb_pred = cross_val_predict(xgb, X, y, cv = cv)
xgb_conf_mat = confusion_matrix(y,y_xgb_pred)

#from sklearn.metrics import roc_auc_score
#print(roc_auc_score(y, y_xgb_pred))


# ## Save and plot feature importance

# In[24]:


#for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
#    print('%s: ' % importance_type, xgb.get_booster().get_score(importance_type=importance_type))


# In[34]:


#count the importance of features, and see actually how many are useful
print("Number of features have importance greater than zero: ", np.count_nonzero(xgb.feature_importances_))


# In[26]:


#######################################################################
#feature_names = xgb.get_booster().feature_names

#record_list = []
#for names in feature_names:
#    record_list.append([names, 0])
    
#feature_important_dict = xgb.get_booster().get_score(importance_type='cover')

#for key,value in feature_important_dict.items():
#    if record_list.index([key,0]) >= 0:
#        index = record_list.index([key,0])
#        record_list.remove([key,0])
#        record_list.insert(index, [key, value])
        
#print(record_list)


# In[27]:


#import csv
#with open("feature_importance(cover_seed?).csv", "w", newline="") as f:
#    writer = csv.writer(f)
#    writer.writerows(record_list)


# In[28]:


#xgb.feature_importances_
#np.savetxt("feature_importance(seed?).csv", xgb.feature_importances_, delimiter=",")


# In[29]:


#######################################
#plot feature importance
###########################################
import xgboost
#xgboost.plot_importance(xgb)
#plt.rcParams['figure.figsize'] = [10, 30]
#plt.savefig('counts_data/(0806)Feature_Importance(deep10+3L_0.1t_0.2var)(BLCA,seed9).png')
#plt.show()


# ## Print out roc auc figures

# In[30]:


########################################
#print out binary class roc auc figure
############################################
fpr, tpr, threshold = metrics.roc_curve(y,y_xgb_prob[:,1])
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[31]:


##########################
#########################


# In[32]:


########################################
#print out multiclass roc auc figure
############################################
#from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import confusion_matrix

#cv = StratifiedKFold(n_splits = num_cv, random_state = r_seed)
#y_xgb_prob = cross_val_predict(xgb, X, y, cv = cv, method = 'predict_proba')

#import matplotlib.pyplot as plt
#plt.figure(figsize = (10, 8))
#plot_multiclass_roc_auc(y, y_xgb_prob, n_classes = 3, title = "xgb, SARC_high_var_0.2, 4 layers")
#plt.savefig('(0606)3class_roc_auc_xgb(4layers12k_0.1test)(SARC_high_var_0.2).png')


# ## Print out results for a given random seed

# In[37]:


print(xgb)
print("Auroc across all folds: %0.5f" % (xgb_multiclass_auroc))
print("Random seed is: ", r_seed)
print("The confusion martix is:\n", xgb_conf_mat)


# In[ ]:


##############################


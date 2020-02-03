# Classification_Algorithms_Python
The intend of this research is to put together the 7 most commonly used classification algorithms along with the python code: 
1)	Logistic Regression, 
2)	Naïve Bayes,
3)	Stochastic Gradient Descent, 
4)	K-Nearest Neighbours, 
5)	Decision Tree, 
6)	Random Forest, 
7)	Support Vector Machine


I have used following Python libraries for while writing codes
import math
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-whitegrid')

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
## Structured Data Classification :
Classification can be performed on structured or unstructured data. Classification is a technique where we categorize data into a given number of classes. 
The main goal of a classification problem is to identify the category/class to which a new data will fall under.

hours_per_week_bin
occupation_bin
msr_bin
capital_gl_bin
race_sex_bin
education_num_bin
education_bin
workclass_bin
age_bin
flag [Train/Test]

y  [Income]    “>50K”  y=1 and “<=50K”  y=0
Normally we are keeping [Train : Test] ratio   70 : 30.

The dataset contains salaries. The following is a description of our dataset:
•	Classes: 2 (‘>50K’ and ‘<=50K’) --- y = 1 or 0
•	attributes (Columns): 7
•	instances (Rows): 48,842



## Logistic Regresssion
#logistic Regression
lr =  LogisticRegression()
lr.fit(x_train, y_train)

make class predictions for the testing set
y_pred=lr.predict(x_test)

Confusion Matrix:
•	Classes: 2 (‘>50K’ and ‘<=50K’) --- y = 1 or 0
Label : Y = 1     ‘>50K’
             Y = 0    ‘<=50K’
N= 16281	Predicted = 0	Predicted = 1
Actual = 0	TN = 11602	FP = 833
Actual = 1	FN = 1676	TP = 2170

Basic Terminology :
True Positives (TP) : We correctly Predicted People do have Salary > 50K [ 2170]
True Negatives (TN) : We correctly predicted that People don’t have salary <= 50K
False Positives (FP) : We incorrectly predicted that People do have salary > 50K [833]  (“Type I Error”)
False Negatives (FN) : We incorrectly predicted that People  don’t have salary <=50K

Accuracy Score: 0.8458939868558443 [ Accuracy = TN+TP / TN+FN+FP+TP]
precision: 0.7226107226107226 [ Precision = TP / TP+FP ]  [1,1] / [1,1] + [0,1]
recall: 0.5642225689027561  [Recall or True Positive Rate or Sensitivity = TP / TP + FN]  [1,1] / [1,1] + [1,0]
specificity: 0.9330116606353036 [ Specificity = TN / TN + FP]  [0,0] / [0,0] + [0,1]
f1 score: 0.6336691487808439

Precision: When a positive value is predicted, how often is the prediction correct?
•	How "precise" is the classifier when predicting positive instances?
•	precision = TP / float(TP + FP) 

Sensitivity: When the Actual value is positive, how often is the prediction correct?
•	Something we want to maximize
•	How "sensitive" is the classifier to detecting positive instances?
•	Also known as "True Positive Rate" or "Recall"
•	TP / all positive
	all positive = TP + FN

•	sensitivity = TP / float(FN + TP)



Specificity: When the actual value is negative, how often is the prediction correct?
•	Something we want to maximize
•	How "specific" (or "selective") is the classifier in predicting positive instances?
•	TN / all negative
	all negative = TN + FP

F1 Score
p = precision & r = recall
f1 score = (2*p*r) / (p+r)


## Naïve Bayes Classifier :
Learning a Naive Bayes classifier is just a matter of counting how many times each attribute co-occurs with each class
GaussianNB ()

## Stochastic Gradient Descent :
While in GD, you have to run through ALL the samples in your training set to do a single update for a parameter in a particular iteration, in SGD, on the other hand, you use ONLY ONE or SUBSET of training sample from your training set to do the update for a parameter in a particular iteration. If you use SUBSET, it is called Minibatch Stochastic gradient Descent.

SGDClassifier 
SGD sikit-learn documentation
loss="hinge": (soft-margin) linear Support Vector Machine, loss="modified_huber": smoothed hinge loss, loss="log": logistic regression




## K- Nearest Neighbours
No Training Step: K-NN does not explicitly build any model, it simply tags the new data entry based learning from historical data. New data entry would be tagged with majority class in the nearest neighbor.

Variety of distance criteria to be choose from: K-NN algorithm gives user the flexibility to choose distance while building K-NN model.
1.	Euclidean Distance
2.	Hamming Distance
3.	Manhattan Distance
4.	Minkowski Distance
K-NN slow algorithm: K-NN might be very easy to implement but as dataset grows efficiency or speed of algorithm declines very fast.

Decision Tree

Max depth in decision tree
Max Depth. Controls the maximum depth of the tree that will be created. It can also be described as the length of the longest path from the tree root to a leaf. The root node is considered to have a depth of 0.


max_depth_options = 10  highest pick in chart
max_depth_options = None  Normalise bar plot on same average compare with others
min_samples_leaf_options = 15  Chart took the pick on 15 then again started dropping


## Random Forest :

N_estimators :   n_estimators represents the number of trees in the forest. Usually the higher the number of trees the better to learn the data. However, adding a lot of trees can slow down the training process considerably, therefore we do a parameter search to find the sweet spot.

Max_features : max_features represents the number of features to consider when looking for the best split.

Min_samples_leaf :  min_samples_leaf is The minimum number of samples required to be at a leaf node. This parameter is similar to min_samples_splits, however, this describe the minimum number of samples of samples at the leafs, the base of the tree.

Oob_score = True
n_estimators = 70
max_features = None
min_samples_leaf = 30




## Support Vecot Machine :
Simple SVM : linear 
In case of linearly separable data in two dimensions, as shown in Fig. 1, a typical machine learning algorithm tries to find a boundary that divides the data in such a way that the misclassification error can be minimized. If you closely look at Fig. 1, there can be several boundaries that correctly divide the data points. The two dashed lines as well as one solid line classify the data correctly.
SVM differs from the other classification algorithms in the way that it chooses the decision boundary that maximizes the distance from the nearest data points of all the classes. An SVM doesn't merely find a decision boundary; it finds the most optimal decision boundary.
The most optimal decision boundary is the one which has maximum margin from the nearest points of all the classes. The nearest points from the decision boundary that maximize the distance between the decision boundary and the points are called support vectors as seen in Fig 2. The decision boundary in case of support vector machines is called the maximum margin classifier, or the maximum margin hyper plane.
 Kernel SVM 
SVM Kernel Types :
Polynomial kernel (poly),  Gaussian Kernel (rbf), Sigmoid Kernel (sigmoid), 

SVM Conclusion : If we compare the performance of the different types of kernels we can clearly see that the linear SVM [Simple SVM] giving us best result comparative to others. As in our data 2 output classes are their so linear performed better.

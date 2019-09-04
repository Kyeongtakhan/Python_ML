import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline

df = pd.read_csv('loan_train.csv')

#Convert to date time object
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])

#Data visualization and pre-processing
#!conda install -c anaconda seaborn -y
import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

#Pre-processing: Feature selection/extraction
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()

#Convert Categorical features to numerical values
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
#Lets convert male to 0 and female to 1
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

#Feature selection, lets defind feature sets, X
X = Feature
y = df['loan_status'].values
#Data Standardization give data zero mean and unit variance (technically should be done after train test split )
X= preprocessing.StandardScaler().fit(X).transform(X)

# KNN
from sklearn.neighbors import KNeighborsClassifier
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 4)
Ks = 15
mean_acc = np.zeros(Ks - 1)
std_acc = np.zeros(Ks - 1)
confusionMX = []
for n in range (1,Ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)
    std_acc[n-1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])
neigh = KNeighborsClassifier(n_neighbors = mean_acc.argmax() +1).fit(X_train,y_train)
yhat = neigh.predict(X_test)
print("KNN jaccard is ",jaccard_similarity_score(yhat,y_test))
print("KNN F1-score is ",f1_score(y_test,yhat,average = None))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
pre_y = preprocessing.LabelEncoder()
pre_y.fit(['PAIDOFF', 'COLLECTION'])
y = pre_y.transform(df['loan_status'].values) 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 4)
dc_tree = DecisionTreeClassifier(criterion = "entropy")
dc_tree.fit(X_train,y_train)
prediction_tree = dc_tree.predict(X_test)

print("Decision tree jaccard is ",jaccard_similarity_score(prediction_tree,y_test))
score = f1_score(prediction_tree, y_test)
print("Decision tree f1 score is ", score)

#Support Vector Machine
from sklearn import svm
pre_y = preprocessing.LabelEncoder()
pre_y.fit(['PAIDOFF', 'COLLECTION'])
y = pre_y.transform(df['loan_status'].values) 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 4)
svm_model = svm.SVC(gamma='scale', kernel = 'sigmoid')
svm_model.fit(X_train,y_train)
yhat = svm_model.predict(X_test)

print("SVM jaccard is ", jaccard_similarity_score(yhat,y_test))
print("SVM f1 score is ", f1_score(yhat,y_test))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 4)
LR_model = LogisticRegression(C = 0.9, solver = 'liblinear').fit(X_train,y_train)
yhat = LR_model.predict(X_test)

print("Logistic Regression jaccard is ", jaccard_similarity_score(yhat,y_test) )
print("Logistic Regression f1 score is ", f1_score(yhat,y_test))
print("Logistic Regression log loss is ", log_loss(yhat,y_test))



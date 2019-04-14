# iris-data-set

# Versions of Python and libraries
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# correlation matrix
corr_matrix = dataset.corr()
print(corr_matrix)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 0
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
nsplits = 10
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=nsplits, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
  
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions with k-nearest neighbors on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print('Accuracy score:', accuracy_score(Y_validation, predictions))
print('Confusion matrix:\n', confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Petal-length and petal-width are highly correlated. Run the algorithm again without petal-width
# Split-out validation dataset
array = dataset.values
X2 = array[:,0:3]
Y2 = array[:,4]
validation_size = 0.20
seed = 0
X_train2, X_validation2, Y_train2, Y_validation2 = model_selection.train_test_split(X2, Y2, test_size=validation_size, random_state=seed)
nsplits = 10
scoring = 'accuracy'
models2 = []
models2.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models2.append(('LDA', LinearDiscriminantAnalysis()))
models2.append(('KNN', KNeighborsClassifier()))
models2.append(('CART', DecisionTreeClassifier()))
models2.append(('NB', GaussianNB()))
models2.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results2 = []
names2 = []
for name, model in models2:
	kfold2 = model_selection.KFold(n_splits=nsplits, random_state=seed)
	cv_results2 = model_selection.cross_val_score(model, X_train2, Y_train2, cv=kfold2, scoring=scoring)
	results2.append(cv_results2)
	names2.append(name)
	msg = "%s: %f (%f)" % (name, cv_results2.mean(), cv_results2.std())
	print(msg)
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results2)
ax.set_xticklabels(names2)
plt.show()
# Make predictions with k-nearest neighbors on validation dataset
knn2 = KNeighborsClassifier()
knn2.fit(X_train2, Y_train2)
predictions2 = knn2.predict(X_validation2)
print('Accuracy score:', accuracy_score(Y_validation2, predictions2))
print('Confusion matrix:\n', confusion_matrix(Y_validation2, predictions2))
print(classification_report(Y_validation2, predictions2))

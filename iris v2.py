# Credits to Jason Brownlee at
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Python version
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

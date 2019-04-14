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

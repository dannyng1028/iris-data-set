#### First data science practice on the famous iris data set

##### Data description:
###### Consists of 150 entries and  5 columns
###### sepal length, sepal width, petal length, petal width, class
###### Target is to classify the flower as setosa, versicolor or virginica.
###### Took 80% (120 entries) of the data set as training data and leave the rest (30) for validation.
###### supervised learning

##### Algorithms applied:
###### logistic regression, linear discriminant analysis, k-nearest neighborhood, CART (classification and regression tree), naive bayesian and support vector machines

###### iris.py dealt with the full 4 columns for classification
###### iris v2.py applied the same classification models but omitted petal-width because it is highly correlated with petal-length that appears not very meaningful to use in training

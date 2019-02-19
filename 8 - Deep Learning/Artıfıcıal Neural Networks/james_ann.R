#importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

#encoding the categorical feature as factor
dataset$Gender = as.numeric(factor(dataset$Gender, levels = c('Male','Female'), labels = c(1,2)))
dataset$Geography = as.numeric(factor(dataset$Geography, levels = c('France','Germany', 'Spain'), labels = c(1,2,3)))

#splitting the dataset into the training set and the test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#feature scaling - 100% compulsary for ANN
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

#fitting ANN classifier to training set
#install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited', training_frame = as.h2o(training_set), activation = 'Rectifier', hidden = c(6,6), epochs = 100, train_samples_per_iteration = -2)

#predicting the test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (prob_pred > 0.5)

#making the confusion matrix
cm = table(test_set[,11], y_pred > 0.5)


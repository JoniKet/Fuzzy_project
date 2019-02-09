# Support Vector Machine (SVM)


# Importing the dataset
train = read.csv('data_train.csv')
test = read.csv('data_test.csv')

# Feature Scaling

train[4:10] = scale(train[4:10])
test[4:10] = scale(test[4:10])

# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = y ~ .,
                 data = train,
                 type = 'C-classification',
                 kernel = 'linear',
                 scale = FALSE)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test)

# Making the Confusion Matrix
cm = table(test[,48], y_pred)

acc = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
sen = cm[2,2]/(cm[2,2]+cm[2,1])
spe = cm[1,1]/(cm[1,1]+cm[1,2])


# Predicting the Train set results
y_pred2 = predict(classifier, newdata = train)
cm = table(train[,48], y_pred2)

acc = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
sen = cm[2,2]/(cm[2,2]+cm[2,1])
spe = cm[1,1]/(cm[1,1]+cm[1,2])
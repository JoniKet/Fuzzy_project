# Fuzzy data analysis course decision tree classification for the data

# Importing the dataset
train = read.csv('data_train.csv')
test = read.csv('data_test.csv')

# Feature Scaling

train[4:10] = scale(train[4:10])
test[4:10] = scale(test[4:10])

# checking differend sizes of trees

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
init_cp = 0.03
library(rpart)
library(rpart.plot)
classifier = rpart(formula = y ~ .,
                   data = train,
                   method = 'class',
                   cp = init_cp) # the complexity parameter is used to find optimal tree size

# Predicting the Test set results
y_pred = predict(classifier, newdata = test[-48],type = 'class')

# Making the Confusion Matrix
cm = table(test[, 48], y_pred)

acc = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
sen = cm[2,2]/(cm[2,2]+cm[2,1])
spe = cm[1,1]/(cm[1,1]+cm[1,2])

# plotting the decision tree
prp(classifier, main = paste('Complexity parameter =',toString(init_cp),'ACC:',toString(round(acc,4)), sep = " "),
    extra = 106,
    fallen.leaves = TRUE)




# Fuzzy data analysis course practical assignment logistic regression classification

# Importing the dataset
train = read.csv('data_train.csv')
test = read.csv('data_test.csv')

# Feature Scaling

train[4:10] = scale(train[4:10])
test[4:10] = scale(test[4:10])


# Fitting Logistic Regression to the Training set
classifier = glm(formula = y ~ .,
                 family = binomial,
                 data = train)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test[-48])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix FOR TEST SET
cm = table(test[, 48], y_pred > 0.5)

acc = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
sen = cm[2,2]/(cm[2,2]+cm[2,1])
spe = cm[1,1]/(cm[1,1]+cm[1,2])


#Predicting the Train set results

prob_pred = predict(classifier, type = 'response', newdata = train[-48])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix FOR TEST SET
cm = table(train[, 48], y_pred > 0.5)

acc = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
sen = cm[2,2]/(cm[2,2]+cm[2,1])
spe = cm[1,1]/(cm[1,1]+cm[1,2])
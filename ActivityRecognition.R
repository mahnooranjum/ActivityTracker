
#===========================================================================
#   Author : MAHNOOR ANJUM
#   Description : Human Activity Recognition
#   Using multiple algorithms and plotting their results
#   Obtaining Classification reports
#
#   References:
#   SuperDataScience
#   Official Documentation
#
#   Data Source:
#   https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones
#===========================================================================
# Importing the dataset
dataset = read.csv('train.csv')

#Encoding
dataset$Activity = factor(dataset$Activity,
                          levels = c('STANDING', 'SITTING','LAYING','WALKING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS'),
                          labels = c(0, 1,2,3,4,5))
# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Activity, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling
training_set[-563] = scale(training_set[-563])
test_set[-563] = scale(test_set[-563])



#PCA
#install.packages('e1071')
#TO VISUALIZE, KINDLY UNCOMMENT THIS CODE
library(e1071)
library(caret)
pca = preProcess(x = training_set[-563], method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c(2, 3, 1)]
test_set = predict(pca, test_set)
test_set = test_set[c(2, 3, 1)]

#THIS IS THE K NEAREST NEIGHBORS ALGORITHM, CURRENTLY THE NUMBER
# OF NEIGHBORS IS 20, YOU CAN USE ANY NUMBER YOU WANT, TRIAL AND RUN
library(class)
knnpred = knn(train = training_set[, -3],
             test = test_set[, -3],
             cl = training_set[, 3],
             k = 20,
             prob = TRUE)
cmknn = table(test_set[, 3], knnpred)

# Fitting Naive Bayes to the Training set============================================
#install.packages('e1071')
#library(e1071)
nb = naiveBayes(x = training_set[-3],
                        y = training_set$Activity)

# Predicting the Test set results
nbpred = predict(nb, newdata = test_set[-3])

# Making the Confusion Matrix
cmnb = table(test_set[, 3], nbpred)





#TO VISUALIZE WE NEED TO USE PRINCIPLE COMPONENT ANALYSIS
#UNCOMMENT THE PCA CODE
#RERUN THE WHOLE THING FOR THIS TO WORK
#VISUALIZING NAIVE BAYES 
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.1)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.1)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(nb, newdata = grid_set)
plot(set[, -3], main = '(Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 5, 'aquamarine', ifelse(y_grid == 4, 'coral1', ifelse(y_grid == 3, 'thistle', ifelse(y_grid == 2, 'darkgoldenrod3', ifelse(y_grid == 1, 'darkorchid1', 'olivedrab1'))))))
points(set, pch = 21, bg = ifelse(set[, 3] == 5, 'aquamarine3', ifelse(set[, 3] == 4, 'coral3', ifelse(set[, 3] == 3, 'thistle4', ifelse(set[, 3] == 2, 'darkgoldenrod1', ifelse(set[, 3] == 1, 'darkorchid4', 'olivedrab3'))))))


#VISUALIZING KNN 

library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.080)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.080)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = knn(train = training_set[, -3], test = grid_set, cl = training_set[, 3], k = 5)
plot(set[, -3],
     main = 'K-NN (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 5, 'aquamarine', ifelse(y_grid == 4, 'coral1', ifelse(y_grid == 3, 'thistle', ifelse(y_grid == 2, 'darkgoldenrod3', ifelse(y_grid == 1, 'darkorchid1', 'olivedrab1'))))))
points(set, pch = 21, bg = ifelse(set[, 3] == 5, 'aquamarine3', ifelse(set[, 3] == 4, 'coral3', ifelse(set[, 3] == 3, 'thistle4', ifelse(set[, 3] == 2, 'darkgoldenrod1', ifelse(set[, 3] == 1, 'darkorchid4', 'olivedrab3'))))))

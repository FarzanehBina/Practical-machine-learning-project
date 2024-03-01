## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt,
forearm, arm, and dumbell of 6 participants to predict the manner in
which they did the exercise.

## Data Preprocessing

    library(caret)

    ## Warning: package 'caret' was built under R version 4.3.2

    ## Loading required package: ggplot2

    ## Loading required package: lattice

    library(rpart)
    library(rpart.plot)

    ## Warning: package 'rpart.plot' was built under R version 4.3.2

    library(randomForest)

    ## Warning: package 'randomForest' was built under R version 4.3.2

    ## randomForest 4.7-1.1

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    library(corrplot)

    ## Warning: package 'corrplot' was built under R version 4.3.2

    ## corrplot 0.92 loaded

### Download the Data

    trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    trainFile <- "./data/pml-training.csv"
    testFile  <- "./data/pml-testing.csv"
    if (!file.exists("./data")) {
      dir.create("./data")
    }
    if (!file.exists(trainFile)) {
      download.file(trainUrl, destfile=trainFile, method="curl")
    }
    if (!file.exists(testFile)) {
      download.file(testUrl, destfile=testFile, method="curl")
    }

### Read the Data

After downloading the data from the data source, we can read the two csv
files into two data frames.

    trainRaw <- read.csv("./data/pml-training.csv")
    testRaw <- read.csv("./data/pml-testing.csv")
    dim(trainRaw)

    ## [1] 19622   160

    dim(testRaw)

    ## [1]  20 160

The training data set contains 19622 observations and 160 variables,
while the testing data set contains 20 observations and 160 variables.
The “classe” variable in the training set is the outcome to predict.

### Clean the data

In this step, we will clean the data and get rid of observations with
missing values as well as some meaningless variables.

    sum(complete.cases(trainRaw))

    ## [1] 406

First, we remove columns that contain NA missing values.

    trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
    testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 

Next, we get rid of some columns that do not contribute much to the
accelerometer measurements.

    classe <- trainRaw$classe
    trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
    trainRaw <- trainRaw[, !trainRemove]
    trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
    trainCleaned$classe <- classe
    testRemove <- grepl("^X|timestamp|window", names(testRaw))
    testRaw <- testRaw[, !testRemove]
    testCleaned <- testRaw[, sapply(testRaw, is.numeric)]

Now, the cleaned training data set contains 19622 observations and 53
variables, while the testing data set contains 20 observations and 53
variables. The “classe” variable is still in the cleaned training set.

### Slice the data

Then, we can split the cleaned training set into a pure training data
set (70%) and a validation data set (30%). We will use the validation
data set to conduct cross validation in future steps.

    set.seed(22519) # For reproducibile purpose
    inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
    trainData <- trainCleaned[inTrain, ]
    testData <- trainCleaned[-inTrain, ]

## Data Modeling

We fit a predictive model for activity recognition using **Random
Forest** algorithm because it automatically selects important variables
and is robust to correlated covariates & outliers in general. We will
use **5-fold cross validation** when applying the algorithm.

    controlRf <- trainControl(method="cv", 5)
    modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
    modelRf

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10988, 10989, 10989, 10991, 10991 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9912654  0.9889499
    ##   27    0.9916291  0.9894104
    ##   52    0.9842766  0.9801110
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 27.

\## Appendix: Figures 1. Correlation Matrix Visualization

    corrPlot <- cor(trainData[, -length(names(trainData))])
    corrplot(corrPlot, method="color")

![](WLE-project_files/figure-markdown_strict/unnamed-chunk-9-1.png) 2.
Decision Tree Visualization

    treeModel <- rpart(classe ~ ., data=trainData, method="class")
    prp(treeModel) # fast plot

![](WLE-project_files/figure-markdown_strict/unnamed-chunk-10-1.png)

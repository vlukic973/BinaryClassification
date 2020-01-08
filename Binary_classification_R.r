# 1. Read the dataset into R

path_data<-"/Users/vesna/Downloads/Training.csv"

data<-read.csv(path_data,header=TRUE,sep=";")

# 2. Clean data. v17, v29 and v19 are both delimited by comma, fix up these columns

split_17<-t(as.data.frame(strsplit(as.character(data$v17), ",")))
split_19<-t(as.data.frame(strsplit(as.character(data$v19), ",")))
split_29<-t(as.data.frame(strsplit(as.character(data$v29), ",")))

data<-cbind(data,split_17,split_19,split_29)

names(data)[20]<-'v17_1'
names(data)[21]<-'v17_2'
names(data)[22]<-'v19_1'
names(data)[23]<-'v19_2'
names(data)[24]<-'v29_1'
names(data)[25]<-'v29_2'

data<-data[,-c(2,3,8)]

data$v17_1<- as.numeric(data$v17_1)
data$v17_2<- as.numeric(data$v17_2)
data$v19_1<- as.numeric(data$v19_1)
data$v19_2<- as.numeric(data$v19_2)
data$v29_1<- as.numeric(data$v29_1)
data$v29_2<- as.numeric(data$v29_2)

# 3. Data exploration

summary(data)

rownames(data)<-1:nrow(data)

Training<-data

missing=sapply(Training,function(x) sum(is.na(x)))

#### There is approx 57% missing data, remove this column

Training=subset(Training, select=-c(v35))

Training=Training[complete.cases(Training), ]

Training2 = as.data.frame(sapply(Training, as.numeric))

Training2$classLabel=Training2$classLabel-1

#### Shuffle rows

Training2 <- Training2[sample(nrow(Training2)),]

##### 4. Make training and validation sets from the original dataset

Training = Training2[0:round(0.8*nrow(Training2)),]
Validation = Training2[(round(0.8*nrow(Training2))+1):nrow(Training2),]

# 5. Try models

########### Try random forest

library(randomForest)
library(caret)
library(e1071)

trControl <- trainControl(method = "cv", number = 10, search = "grid")

# Set a seed to ensure reproducibility of results

set.seed(1234)

# Train the random forest model
rf_default <- train(as.factor(classLabel)~., data = Training, method = "rf", metric = "Accuracy", trControl = trControl)

# Print the results
print(rf_default)

#### We observe good performance on the training set. Remove the outcome column from the validation data in order to apply the trained model

Validation2 <- Validation

Validation <- Validation[ , !(names(Validation) %in% c('classLabel'))]

##### Predict the labels of the validation set based on trained model

fitted.results <- predict(rf_default,newdata=Validation,type='raw')

# Calculate classification accuracy

misClasificError <- mean(fitted.results != Validation2$classLabel)
print(paste('Accuracy',1-misClasificError))

#### Observed perfect classification accuracy. Generate an ROC curve as well, as classification accuracy alone is inadequate, especially since the dataset is imbalanced

library(ROCR)
p <- predict(rf_default, newdata=Validation, type='raw')
pr <- prediction(as.integer(p)-1, Validation2$classLabel)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

# Calculate the area under the ROC curve

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

######## Try the Gradient Boosted Machine model to see if we get the same results as when using the random forest

gbm_Model <- train(as.factor(classLabel)~., data = Training, method = "gbm", metric = "Accuracy", trControl = trControl)

gbm_fitted <- predict(gbm_Model,newdata=Validation,type='raw')

misClasificError <- mean(gbm_fitted != Validation2$classLabel)
print(paste('Accuracy',1-misClasificError))

summary(gbm_Model)

# We find that the v7 column is perfectly correlated with the outcome variable, and there is no visible correlation between the other columns and the outcome variable.

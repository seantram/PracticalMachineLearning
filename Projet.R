

setwd("D:\\OneDrive\\Coursera\\Data Science\\Practical Machine Learning")

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url, "pml-training.csv",  quiet = FALSE, mode = "w", cacheOK = TRUE)
download.file(url2, "pml-testing.csv",  quiet = FALSE, mode = "w", cacheOK = TRUE)



data <- read.csv("pml-training.csv", na.strings=c("NA","","#DIV/0!"), strip.white=TRUE)

data <- subset( data, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp ) )

data2 <- data[, colMeans(is.na(data)) <= .50]


library(caret)


set.seed(123)
inTrain  <- createDataPartition(y=data2$classe, p=0.6, list=FALSE)
training <- data2[inTrain,]
testing  <- data2[-inTrain,]


#par <- trainControl(allowParallel=TRUE, method="cv", number=5)


ctrl <- trainControl(allowParallel=TRUE, method = "cv", repeats = 2)

modFit <- train(classe ~ ., method ="rf", data=training, prox=TRUE, trControl=ctrl)

##########################################################################
library(doSNOW)
library(foreach)
library(randomForest)
library(RColorBrewer)

y <- as.factor(training[, 55])
x <-  subset( training, select = -classe)

cl <- makeCluster(12, type="SOCK")
registerDoSNOW(cl)

rf.mod    <- foreach(ntree=rep(500, 3), .combine=combine, .multicombine=TRUE,
                     .packages='randomForest') %dopar% {
                         randomForest(x, y, ntree=ntree, cv.fold=4)
                     }

#####################################################################
color <- brewer.pal(n	= 8, "Dark2")
imp	  <- as.data.frame(rf.mod$importance[order(rf.mod$importance),])
barplot(t(imp), col=color[1])
points(which(imp==imp['rand',]),0.6, col=color[2], type='h', lwd=2)


prediction <- predict(rf.mod, testing)


confusionMatrix <- confusionMatrix(prediction, testing$classe)

with(rf.mod, plot( error.cv, log="x", type="o", lwd=2))


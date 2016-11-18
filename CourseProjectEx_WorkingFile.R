# PREDICT 422 Practical Machine Learning

# Course Project - Working R Script File

# OBJECTIVE: A charitable organization wishes to develop a machine learning
# model to improve the cost-effectiveness of their direct marketing campaigns
# to previous donors. - test

# 1) Develop a classification model using data from the most recent campaign that
# can effectively capture likely donors so that the expected net profit is maximized.

# 2) Develop a prediction model to predict donation amounts for donors - the data
# for this will consist of the records for donors only.


# load the data, we will have to each add our own code here
charity <- read.csv("/Users/paulbertucci/Desktop/MSPA/PRED 422/charity.csv") # load the "charity.csv" file
#charity <- read.csv("/Users/mexic_000/Dropbox/Courses/Northwestern/422/Final_Group_Project/charity.csv")

#transformation summary
par(mfcol=c(1,2))
hist(charity$avhv)
hist(log(charity$avhv))

hist(charity$incm)
hist(log(charity$incm))

hist(charity$inca)
hist(log(charity$inca))

hist(charity$tgif)
hist(log(charity$tgif))

hist(charity$agif)
hist(log(charity$agif))

hist(charity$tdon)
hist(log(charity$tdon))

hist(charity$npro)
hist((charity$npro)^(2/3))  #this one may not be worthwhile

hist(charity$plow)
hist(charity$plow^(1/3)) #this does not look normal, but definitely better

hist(charity$lgif)
hist(charity$lgif^(1/5))

hist(charity$rgif)
hist(charity$rgif^(1/7))

hist(charity$tlag)
hist(charity$tlag^(1/5))
par(mfcol=c(1,1))


# predictor transformations

charity.t <- charity
charity.t$avhv <- log(charity.t$avhv)
# add further transformations if desired
# for example, some statistical methods can struggle when predictors are highly skewed

#these are based on the histograms above
charity.t$incm_log <- log(charity.t$incm)
charity.t$inca_log <- log(charity.t$inca)
charity.t$tgif_log <- log(charity.t$tgif)
charity.t$agif_log <- log(charity.t$agif)
charity.t$tdon_log <- log(charity.t$tdon)
charity.t$npro_pwr <- (charity.t$npro)^(2/3)
charity.t$plow_pwr <- charity.t$plow^(1/3)
charity.t$lgif_pwr <- charity.t$lgif^(1/5)
charity.t$rgif_pwr <- charity.t$rgif^(1/7)
charity.t$tlag_pwr <- charity.t$tlag^(1/5)


# set up data for analysis
# added tranformed variables into the data set by replacing 2;21 with c(2:21,25:34)

data.train <- charity.t[charity$part=="train",]
# x.train <- data.train[,c(2:21,25:34)]
x.train <- data.train[,2:21]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.t[charity$part=="valid",]
# x.valid <- data.valid[,c(2:21,25:34)]
x.valid <- data.valid[,2:21]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.t[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
# x.test <- data.test[,c(2:21,25:34)]
x.test <- data.test[,2:21]

x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # Standardized data to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) #  Standardized data to predict damt when donr=1

#Validation set
x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)



##### CLASSIFICATION MODELING ######

# linear discriminant analysis

library(MASS)

model.lda1 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c) # include additional terms on the fly using I()

# Note: strictly speaking, LDA should not be used with qualitative predictors,
# but in practice it often is if the goal is simply to find a good predictive model

post.valid.lda1 <- predict(model.lda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.lda1 <- cumsum(14.5*c.valid[order(post.valid.lda1, decreasing=T)]-2)
plot(profit.lda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda1)) # report number of mailings and maximum profit
# 1329.0 11624.5

cutoff.lda1 <- sort(post.valid.lda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda1 <- ifelse(post.valid.lda1>cutoff.lda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda1, c.valid) # classification table
#               c.valid
#chat.valid.lda1   0   1
#              0 675  14
#              1 344 985
# check n.mail.valid = 344+985 = 1329
# check profit = 14.5*985-2*1329 = 11624.5

# logistic regression

model.log1 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c, family=binomial("logit"))

post.valid.log1 <- predict(model.log1, data.valid.std.c, type="response") # n.valid post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit
# 1291.0 11642.5

cutoff.log1 <- sort(post.valid.log1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.log1 <- ifelse(post.valid.log1>cutoff.log1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1, c.valid) # classification table
#               c.valid
#chat.valid.log1   0   1
#              0 709  18
#              1 310 981
# check n.mail.valid = 310+981 = 1291
# check profit = 14.5*981-2*1291 = 11642.5
###################
# KNN
###################

library(class)
set.seed(1)
knn.pred=knn(data.train.std.c[,-21],data.valid.std.c[,-21],data.train.std.c[,21],k=5,prob=TRUE)
knn.pred.cv=knn.cv(data.train.std.c[,-21],data.train.std.c[,21],k=3,prob=TRUE)

for (i in 1:10)
{
  knn.pred=knn(data.train.std.c[,-21],data.valid.std.c[,-21],data.train.std.c[,21],k=i,prob=TRUE)
  success<-(mean(knn.pred==c.valid))
  print(success)
}

#confusion matrix
table(knn.pred,c.valid)
mean(knn.pred==c.valid)

knn.attributes<-attributes(knn.pred)
knn.pred.num<-as.numeric(knn.pred)-1
post.valid.knn<-knn.attributes$prob*knn.pred.num

profit.knn <- cumsum(14.5*c.valid[order(post.valid.knn, decreasing=T)]-2)
plot(profit.knn)
n.mail.valid <- which.max(profit.knn) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.knn)) # report number of mailings and maximum profit
n.mail.valid
max(profit.knn)

cutoff.knn <- sort(post.valid.knn, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.knn <- ifelse(post.valid.knn>cutoff.knn, 1, 0) # mail to everyone above the cutoff
table(knn.pred, c.valid) # classification table
table(chat.valid.knn, c.valid) # sanity check



###################
# Dec Tree Model
###################
library(tree)

tree.fit=tree(factor(data.train.std.c$donr) ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
              data=data.train.std.c)

plot(tree.fit)
text(tree.fit,pretty=1)

cv.tree.fit=cv.tree(tree.fit,FUN=prune.misclass)
names(cv.tree.fit)
cv.tree.fit
par(mfrow=c(1,2))
plot(cv.tree.fit$size,cv.tree.fit$dev,type="b")
plot(cv.tree.fit$k,cv.tree.fit$dev,type="b")



# We now apply the prune.misclass() function in order to prune the tree to the nodes with the lowest dev
lowest.dev.node<-cv.tree.fit$size[which.min(cv.tree.fit$dev)]
prune.tree=prune.misclass(tree.fit,best=lowest.dev.node)
plot(prune.tree)
text(prune.tree,pretty=1)

post.valid.tree <- predict(prune.tree, data.valid.std.c, type="vector")[,2] # n.valid post probs

profit.tree <- cumsum(14.5*c.valid[order(post.valid.tree, decreasing=T)]-2)
plot(profit.tree) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.tree) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.tree)) # report number of mailings and maximum profit
n.mail.valid
max(profit.tree)

cutoff.tree <- sort(post.valid.tree, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.tree <- ifelse(post.valid.tree>cutoff.tree, 1, 0) # mail to everyone above the cutoff
table(chat.valid.tree, c.valid) # classification table



###################
# Bagging Model
###################
model.bag <- randomForest::randomForest(factor(data.train.std.c$donr) ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                                          avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                                        data.train.std.c,mtry=16,importance=TRUE) 
randomForest::importance(model.bag)
randomForest::varImpPlot(model.bag)

post.valid.bag <- predict(model.bag, data.valid.std.c,type='prob')[,2] # n.valid.c post probs
# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.bag <- cumsum(14.5*c.valid[order(post.valid.bag, decreasing=T)]-2)
plot(profit.bag) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.bag) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.bag)) # report number of mailings and maximum profit

cutoff.bag <- sort(post.valid.bag, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.bag <- ifelse(post.valid.bag>cutoff.bag, 1, 0) # mail to everyone above the cutoff
table(chat.valid.bag, c.valid) # classification table




###################
# Random Forest Model
###################
model.rf <- randomForest::randomForest(factor(data.train.std.c$donr) ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                                         avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,
                                       data.train.std.c) 

# Exploring Variable Selection - this code takes too long
# # ensure the results are repeatable
# set.seed(1)
# # load the library
# library(mlbench)
# library(caret)
# # define the control using a random forest selection function
# control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# # run the RFE algorithm
# results_fs <- rfe(data.train.std.c[,1:20], data.train.std.c[,21], sizes=c(1:8), rfeControl=control)
# # summarize the results
# print(results_fs)
# # list the chosen features
# predictors(results_fs)
# # plot the results
# plot(results_fs, type=c("g", "o"))

# model.rf <- randomForest::randomForest(factor(data.train.std.c$donr) ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
#                                          avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,
#                                        data.train.std.c) 

randomForest::importance(model.rf)
randomForest::varImpPlot(model.rf)

post.valid.rf <- predict(model.rf, data.valid.std.c,type='prob')[,2] # n.valid.c post probs
# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.rf <- cumsum(14.5*c.valid[order(post.valid.rf, decreasing=T)]-2)
plot(profit.rf) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.rf) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.rf)) # report number of mailings and maximum profit

cutoff.rf <- sort(post.valid.rf, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.rf <- ifelse(post.valid.rf>cutoff.rf, 1, 0) # mail to everyone above the cutoff
table(chat.valid.rf, c.valid) # classification table



###################
# Boosting Model
###################
library(gbm)
# We run gbm() with the option distribution="gaussian" since this is a regression problem; if it were a binary
# classification problem, we would use distribution="bernoulli". 
# The argument n.trees=5000 indicates that we want 5000 trees, and the option interaction.depth=4 limits the depth of each tree.

model.boost=gbm(data.train.std.c$donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                data=data.train.std.c,distribution="bernoulli",n.trees=5000,interaction.depth=1)

# The summary() function produces a relative influence plot and also outputs the relative influence statistics.
summary(model.boost)

post.valid.boost <- predict(model.boost, data.valid.std.c,type='response',n.trees=5000) # n.valid.c post probs
# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.boost <- cumsum(14.5*c.valid[order(post.valid.boost, decreasing=T)]-2)
plot(profit.boost) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.boost) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.boost)) # report number of mailings and maximum profit

cutoff.boost <- sort(post.valid.boost, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.boost <- ifelse(post.valid.boost>cutoff.boost, 1, 0) # mail to everyone above the cutoff
table(chat.valid.boost, c.valid) # classification table

###################
# SVM
###################

library(e1071)

svmfit=svm(data.train.std.c$donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
             avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
           data=data.train.std.c, kernel="radial",  gamma=.5, cost=10)
plot(svmfit, data.train.std.c)
summary(svmfit)

# Still working on this code, attempting to tune the model use cv, this code takes too long
# set.seed(1)
# tune.out=tune(svm,donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
#                 avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
# data=data.train.std.c, kernel="radial", ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
# summary(tune.out)


post.valid.svm <- predict(svmfit, data.valid.std.c) # n.valid.c post probs
profit.svm <- cumsum(14.5*c.valid[order(post.valid.svm, decreasing=T)]-2)
plot(profit.svm) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.svm) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.svm)) # report number of mailings and maximum profit

cutoff.svm <- sort(post.valid.svm, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.svm <- ifelse(post.valid.svm>cutoff.svm, 1, 0) # mail to everyone above the cutoff
table(chat.valid.svm, c.valid) # classification table


################
#Results DataFrame
################
model<-c("LDA1","Log1")
n.mail<-c(which.max(profit.lda1),which.max(profit.log1))
profit<-c(max(profit.lda1),max(profit.log1))


results<-data.frame(model,n.mail,profit,stringsAsFactors = FALSE)
# adding results for each new model
results<-rbind(c("RandomForest",which.max(profit.rf),max(profit.rf)),results)
results<-rbind(c("Dec Tree",which.max(profit.tree),max(profit.tree)),results)
results<-rbind(c("Bagging",which.max(profit.bag),max(profit.bag)),results)
results<-rbind(c("Boosting",which.max(profit.boost),max(profit.boost)),results)
results<-rbind(c("KNN",which.max(profit.knn),max(profit.knn)),results)
results<-rbind(c("SVM",which.max(profit.svm),max(profit.svm)),results)


results


# Results

# n.mail Profit  Model
# 1329   11624.5 LDA1
# 1291   11642.5 Log1

# select model.log1 since it has maximum profit in the validation sample

post.test <- predict(model.log1, data.test.std, type="response") # post probs for test data

# Oversampling adjustment for calculating number of mailings for test set

n.mail.valid <- which.max(profit.log1)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)
#    0    1 
# 1676  331
# based on this model we'll mail to the 331 highest posterior probabilities

# See below for saving chat.test into a file for submission



##### PREDICTION MODELING ######

# Least squares regression

model.ls1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)

pred.valid.ls1 <- predict(model.ls1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
# 1.867523
sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y) # std error
# 0.1696615

# drop wrat for illustrative purposes
model.ls2 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)

pred.valid.ls2 <- predict(model.ls2, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls2)^2) # mean prediction error
# 1.867433
sd((y.valid - pred.valid.ls2)^2)/sqrt(n.valid.y) # std error
# 0.1696498

# Results

# MPE  Model
# 1.867523 LS1
# 1.867433 LS2

# select model.ls2 since it has minimum mean prediction error in the validation sample

yhat.test <- predict(model.ls2, newdata = data.test.std) # test predictions




# FINAL RESULTS

# Save final results for both classification and regression

length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt

ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="ABC.csv", row.names=FALSE) # use your initials for the file name

# submit the csv file in Canvas for evaluation based on actual test donr and damt values

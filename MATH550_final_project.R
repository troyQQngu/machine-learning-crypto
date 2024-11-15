library(readr)
cpto <- read_csv("Desktop/Summer 2022/Research/stocks_crypto_data/crypto_return.csv")
snp <- read_csv("Desktop/Summer 2022/Research/stocks_crypto_data/s&p500_return.csv")
cpto <- cpto[ , colSums(is.na(cpto))==0]
snp <- snp[ , colSums(is.na(snp))==0]
cpto$Date <- as.Date(cpto$Date,format = "%Y/%m/%d")
snp$Date <-as.Date(snp$Date,format = "%Y-%m-%d")

df <- merge(x=cpto,y=snp,by="Date",all.y = FALSE)
# Question 1: Which crypto currency correlates with the Bitcoin the most in terms of daily return? How about S&P 500 stocks?

Bcor_cc <- cor(cpto$BTC,cpto[,3:58])
lst1 <- sort(abs(Bcor_cc),decreasing=TRUE,index.return=TRUE)
lst1$x[1:10]
Bcor_cc[,lst1$ix[1:10]]

Bcor_sp <- cor(df$BTC,df[,(ncol(cpto)+1):ncol(df)])
lst2 <- sort(abs(Bcor_sp),decreasing=TRUE,index.return=TRUE)
lst2$x[1:10]
Bcor_sp[,lst2$ix[1:10]]

# Question 2: Can we predict the next-day return of the Bitcoin with these 20 stocks/cryptos? 
#   Sub-question 1: What methods can we use? subset selection, LASSO, ridge(?), PCR(?), decision trees(Boost), RNN(?)
#   Sub-question 2: How should we split the data set? Realistically, we would be given the historical data to predict the future returns. 
#     Does that suggest we should also split the data chronologically? Would RNN be a better method in this case?
#   Sub-question 3: Does correlation imply prediction power?

# subset selections
library(leaps)

y <- df$BTC[2:860]

regfit.fwd <- regsubsets(y=y,x=df[1:859,2:555],method="forward",nvmax=20)
fwd.summary <- summary(regfit.fwd)
coef(regfit.fwd,20)
fwd.idx <- which(fwd.summary$which[20,],arr.ind = TRUE)

regfit.bwd <- regsubsets(y=y,x=df[1:859,2:555],method="backward",nvmax=20)
bwd.summary <- summary((regfit.bwd))
coef(regfit.bwd,20)
bwd.idx <- which(bwd.summary$which[20,],arr.ind = TRUE)

myidx <- union(fwd.idx,bwd.idx)[c(-1)]
mydat <- df[1:859,myidx] 

# LASSO 
library(glmnet)
set.seed(12)
idx <- sample(1:nrow(mydat),0.7*nrow(mydat))
x.train <- data.matrix(mydat[idx,])
x.test <- data.matrix(mydat[-idx,])
y.train <- y[idx]
y.test <- y[-idx]

set.seed(12)
cv_model <- cv.glmnet(x.train, y.train, alpha = 1)
best_lambda <- cv_model$lambda.min

best_model <- glmnet(x.train, y.train, alpha = 1, lambda = best_lambda)
coef(best_model)
plot(cv_model)

y_pred <- predict(best_model, s = best_lambda, newx = x.test)
sst <- sum((y.test - mean(y.test))^2)
sse <- sum((y_pred - y.test)^2)

rsq_L <- 1 - sse/sst
rsq_L

# Ridge regression
set.seed(19)
cv_model <- cv.glmnet(x.train, y.train, alpha = 0)
best_lambda <- cv_model$lambda.min

best_model <- glmnet(x.train, y.train, alpha = 1, lambda = best_lambda)
coef(best_model)
plot(cv_model)

y_pred <- predict(best_model, s = best_lambda, newx = x.test)
sst <- sum((y.test - mean(y.test))^2)
sse <- sum((y_pred - y.test)^2)

rsq_R <- 1 - sse/sst
rsq_R

# Elastic Net
set.seed(19)
cv_model <- cv.glmnet(x.train, y.train, alpha = 1)
best_lambda <- cv_model$lambda.min

best_model <- glmnet(x.train, y.train, alpha = 1, lambda = best_lambda)
coef(best_model)

y_pred <- predict(best_model, s = best_lambda, newx = x.test)
sst <- sum((y.test - mean(y.test))^2)
sse <- sum((y_pred - y.test)^2)

rsq_EN <- 1 - sse/sst
rsq_EN
# Conclusion: LASSO, Ridge, and Elastic Net are not working under the randomized sampling.

# Try RNN
library(keras) 
library(tidyverse)
library(tensorflow)

xdata <- data.matrix(mydat)
xdata <- scale(xdata)
lagm <- function(x, k = 1) {
  n <- nrow(x)
  pad <- matrix(NA, k, ncol(x))
  rbind(pad, x[1:(n - k), ]) 
}
arframe <- data.frame(btc_pred = xdata[, "BTC"], L1 = lagm(xdata, 1), L2 = lagm(xdata, 2),L3 = lagm(xdata, 3), L4 = lagm(xdata, 4),L5 = lagm(xdata, 5))

arframe <- arframe[-(1:5), ]
arfit <- lm(btc_pred ~ ., data = arframe[idx, ])
arpred <- predict(arfit, arframe[-idx, ])
V0 <- var(arframe[-idx, "btc_pred"])
1 - mean((arpred - arframe[-idx, "btc_pred"])^2) / V0


n <- nrow(arframe)
xrnn <- data.matrix(arframe[, -1])
xrnn <- array(xrnn, c(n, 30, 5))
xrnn <- xrnn[,, 5:1]
xrnn <- aperm(xrnn, c(1, 3, 2))

# model <- keras_model_sequential() %>%
#  layer_simple_rnn(units = 12,input_shape = list(5, 30),dropout = 0.1, recurrent_dropout = 0.1) %>%
#  layer_dense(units = 1)
# model %>% compile(optimizer = optimizer_rmsprop(),loss = "mse")
# history <- model %>% fit(
# xrnn[idx,, ], arframe[idx, "btc_pred"], batch_size = 64, epochs = 200, 
# validation_data = list(xrnn[-idx,, ], arframe[-idx, "btc_pred"]) )
# 1 - mean((kpred - arframe[-idx, "btc_pred"])^2) / V0

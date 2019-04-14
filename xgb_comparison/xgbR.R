library(mlr)
library(dplyr)
library(mltools)
library(DALEX)

##R

test <- select(read.csv("titanic_test.csv"), -c(X))
train <- select(read.csv("titanic_train.csv"), -c(X))
titanic_test_X <- select(read.csv("titanic_test.csv"), -c(X, survived))
titanic_test_Y <- select(read.csv("titanic_test.csv"), c(survived))
titanic_train_X <- select(read.csv("titanic_train.csv"), -c(X, survived))
titanic_train_Y <- select(read.csv("titanic_train.csv"), c(survived)) 
y <- unclass(titanic_train_Y)
y <- y$survived


custom_predict <- function(object, newdata) {pred <- predict(object, newdata=newdata)
response <- pred$data[,3]
return(response)}

task <- makeClassifTask(id = "test", data = train, target = "survived")
learner <- makeLearner("classif.xgboost", 
                       par.vals = list(booster = "gbtree", eta = 0.1, num_parallel_tree = 100,
                                       gamma = 0, min_child_weight = 1, max_depth = 3,
                                       max_delta_step = 0, subsample = 1, alpha = 0,
                                       lambda = 1), 
                       predict.type = "prob")
model <- train(learner, task)
preds <- predict(model, newdata = test)

r_explain <- DALEX::explain(model, data = train,
                      y = y, label = "R",
                      predict_function = custom_predict)

preds$data$prob.1


model_performance(r_explain)

auc_roc(preds$data$prob.1, test$survived)


## Python

library(iBreakDown)
library(reticulate)




rf <- py_load_object("pima.pickle.pkl", pickle = "pickle")

predict_function <- function(model, newdata){
  model$predict_proba(newdata)[,2]
}



rf_explain <- explain(rf, data = titanic_train_X,
                      y = y, label = "python",
                      predict_function = predict_function)


## Plots

plot(variable_importance(rf_explain), variable_importance(r_explain))
plot(model_performance(rf_explain), model_performance(r_explain))


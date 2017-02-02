---
layout: post
title:  "Predicting House Prices"
date:   2016-10-11 18:54:42 +0300
categories: ["data-science"]
excerpt: "We will try to predict the sale price of houses in King County, USA using a decision tree model. Later, we will improve our predictions using Gradient Boosted trees. Using R!"
---

So you have a house for sale or buying one? What is a fair price for it? Can we predict it correctly?

Let's use the "House Sales in King County" data available at [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction) to answer that question. Each row of the dataset contains information about a home sold between May 2014 and May 2015 along with the price in US dollars. Some of the other features include:

* bedrooms - number of bedrooms
* bathrooms - number of bathrooms
* floors - number of floors
* yr_built - year built
* zipcode
* long - longitude
* lat - latitude
* condition - building condition (ordered categorical variable in the range 1 - 5)
* grade - construction quality of improvements (ordered categorical variable in the range 1 - 13)

If not interested in house prices you still can learn something about regression, classification trees, and extreme gradient boosting.


# Fire up R and load some libraries


```R
library(ggplot2)
library(reshape2)
library(plyr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)
library(doMC)
library(scales)
library(GGally)
```

Load our utility functions, make results reproducible and instruct R to use all our CPU cores (my PC has 8 cores, you might want to revise that value for yours).


```R
source("utils.R")

set.seed(42)
theme_set(theme_minimal())
registerDoMC(cores = 8)
options(warn=-1)
```

# Load and preprocess the dataset


```R
df <- read.csv("data/kc_house_data.csv", stringsAsFactors = FALSE)
```


```R
print(paste("rows:", nrow(df), "cols:", ncol(df)))
```

    [1] "rows: 21613 cols: 21"


Remove id and date columns and instruct R to interpret condition, view, grade and waterfront as factors.


```R
df <- df[-c(1, 2)]
df$condition <- as.factor(df$condition)
df$view <- as.factor(df$view)
df$grade <- as.factor(df$grade)
df$waterfront <- as.factor(df$waterfront)
```


# Exploration

Do we have missing data?


```R
ggplot_missing(df)
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_11_1.png)


It looks like everything is in here! Great!

## Maps

The following awesome maps were created by [Thierry Ellena](https://harlfoxem.github.io/). Let's have a look at them:

[House locations](https://harlfoxem.github.io/houses.html) <br/>
[Number of houses by zipcode](https://harlfoxem.github.io/count.html) <br/>
[Price by zipcode](https://harlfoxem.github.io/price.html)

Let's look at the distribution of house condtion, grade and price:


```R
p1 <- qplot(condition, data=df, geom = "bar",
    main="Number of houses by condition")

p2 <- qplot(grade, data=df, geom = "bar",
    main="Number of houses by grade")

p3 <- ggplot(df, aes(price)) + geom_density() + 
    scale_y_continuous(labels = comma) +
    scale_x_continuous(labels = comma, limits = c(0, 2e+06)) +
    xlab("price") +
    ggtitle("Price distribution")

multiplot(p1, p2, p3)
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_14_0.png)


And a look at price (log10) vs other features: 


```R
ggplot(df, aes(x=log10(price), y=sqft_living)) +
    geom_smooth() +
    scale_y_continuous(labels = comma) +
    scale_x_continuous(labels = comma) +
    ylab("sqft of living area") + 
    geom_point(shape=1, alpha=1/10) +
    ggtitle("Price (log10) vs sqft of living area")
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_16_1.png)



```R
ggplot(df, aes(x=grade, y=log10(price))) +
    geom_boxplot() +
    scale_y_continuous(labels = comma) +
    coord_flip() +
    geom_point(shape=1, alpha=1/10) +
    ggtitle("Price (log10) vs grade")
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_17_1.png)



```R
ggplot(df, aes(x=condition, y=log10(price))) +
    geom_boxplot() +
    scale_y_continuous(labels = comma) +
    coord_flip() +
    geom_point(shape=1, alpha=1/10) +
    ggtitle("Price (log10) vs condition")
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_18_1.png)



```R
ggplot(df, aes(x=as.factor(floors), y=log10(price))) +
    geom_boxplot() +
    scale_y_continuous(labels = comma) +
    xlab("floors") +
    coord_flip() +
    geom_point(shape=1, alpha=1/10) +
    ggtitle("Price (log10) vs number of floors")
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_19_1.png)


How different features correlate?


```R
ggcorr(df, hjust = 0.8, layout.exp = 1) + 
    ggtitle("Correlation between house features")
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_21_1.png)


# Splitting the data

We will split the data using the <code>caret</code> package. 90% will be used for training and 10% for testing.


```R
train_idx = createDataPartition(df$price, p=.9, list=FALSE)

train <- df[train_idx, ]
test <- df[-train_idx, ]
```

We will extract the labels (true values) from our test dataset.


```R
test_labels <- test[, 1]
```

# First attempt of building a model

Let's build a decision tree with the <code>rpart</code> package using all features (except price) as predictors:


```R
tree_fit <- rpart(price ~ ., data=df)
tree_predicted <- predict(tree_fit, test)
```

And the results of our model:


```R
summary(tree_predicted)
```


       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
     315400  315400  462800  542000  654900 5081000 



```R
summary(test_labels)
```


       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      82000  322100  450000  536300  643900 3419000 



```R
cor(tree_predicted, test_labels)
```


0.814574873081786



```R
rmse(tree_predicted, test_labels)
```


197634.31260839


#### How the actual and predicted distributions compare to each other?


```R
res <- data.frame(price=c(tree_predicted, test_labels), 
          type=c(replicate(length(tree_predicted), "predicted"), 
                 replicate(length(test_labels), "actual")))

ggplot(res, aes(x=price, colour=type)) +
    scale_x_continuous(labels = comma, limits = c(0, 2e+06)) +
    scale_y_continuous(labels = comma) +
    geom_density()
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_34_1.png)


Not very good, eh? Let's dig a bit deeper.

#### How does our model looks like?


```R
rpart.plot(tree_fit, digits = 4, fallen.leaves = TRUE,
             type = 3, extra = 101)
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_36_0.png)


It seems that the grade, location (lat, long), square feet are important factors for deciding the price of a house.

# Fitting a xgbTree model

That was a good first attempt. Ok, it wasn't even good. So, can we do better? Let's try an ensemble of boosted trees. For good intro to boosted trees see: [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/latest/model.html).

First, we will set up the resampling method used by <code>caret</code>. 10 cross-validation passes should do (preferably in parallel).


```R
ctrl = trainControl(method="cv", number=10, allowParallel = TRUE)
```

Our next step is to find good parameters for <code>XGBoost</code>. See the references below to find out how to tune the parameters for your particular problem. Those are the parameters I've tried:


```R
param_grid <-  expand.grid(eta = c(0.3, 0.5, 0.8), 
                        max_depth = c(4:10), 
                        gamma = c(0), 
                        colsample_bytree = c(0.5, 0.6, 0.7),
                        nrounds = c(120, 140, 150, 170), 
                        min_child_weight = c(1))
```

After trying them out, the following were chosen:


```R
param_grid <- expand.grid(eta=c(0.3), 
                          max_depth= c(6), 
                          gamma = c(0), 
                          colsample_bytree = c(0.6), 
                          nrounds = c(120),
                          min_child_weight = c(1))
```

Finally, time to train our model using *root mean squared error* as score metric:


```R
xgb_fit = train(price ~ ., 
            data=df, method="xgbTree", metric="RMSE",
            trControl=ctrl, subset = train_idx, tuneGrid=param_grid)

xgb_predicted = predict(xgb_fit, test, "raw")
```

and the results:


```R
summary(xgb_predicted)
```


       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
     143100  323700  464700  542100  649700 6076000 



```R
summary(test_labels)
```


       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      82000  322100  450000  536300  643900 3419000 



```R
cor(xgb_predicted, test_labels)
```


0.926845603997267



```R
rmse(xgb_predicted, test_labels)
```


132324.026367212


comparison of actual and predicted distributions:


```R
res <- data.frame(price=c(xgb_predicted, test_labels), 
            type=c(replicate(length(xgb_predicted), "predicted"), 
                  replicate(length(test_labels), "actual")))

ggplot(res, aes(x=price, colour=type)) +
    scale_x_continuous(labels = comma, limits = c(0, 2e+06)) +
    scale_y_continuous(labels = comma) +
    geom_density()
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_52_1.png)


The distributions look much more similar compared to the one produced by the decision tree model.

#### What are the most important features according to our model?


```R
imp <- varImp(xgb_fit, scale = FALSE)

imp_names = rev(rownames(imp$importance))
imp_vals = rev(imp$importance[, 1])

var_importance <- data_frame(variable=imp_names,
                             importance=imp_vals)
var_importance <- arrange(var_importance, importance)
var_importance$variable <- factor(var_importance$variable, 
        levels=var_importance$variable)

var_importance_top_15 = var_importance[with(var_importance, 
        order(-importance)), ][1:15, ]

ggplot(var_importance_top_15, aes(x=variable, weight=importance)) +
 geom_bar(position="dodge") + ggtitle("Feature Importance (Top 15)") +
 coord_flip() + xlab("House Attribute") + ylab("Feature Importance") +
 theme(legend.position="none")
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_55_1.png)


# Compare distributions of predictions

Let's see how the tree distributions compare to each other:


```R
res <- data.frame(price=c(tree_predicted, xgb_predicted, test_labels), 
                  type=c(replicate(length(tree_predicted), "tree"), 
                         replicate(length(xgb_predicted), "xgb"),
                         replicate(length(test_labels), "actual")
                        ))

ggplot(res, aes(x=price, colour=type)) +
    scale_x_continuous(labels = comma, limits = c(0,2e+06)) +
    scale_y_continuous(labels = comma) +
    geom_density()
```


{:.center}
![png]({{site.url}}/assets/3.predicting_house_prices_files/2.predicting_house_prices_57_1.png)


Again, we can confirm that the Boosted Trees model provides much more accurate distribution with its predictions.

# How well we did, really?

Let's randomly choose 10 rows and look at the difference between predicted and actual price:


```R
test_sample <- sample_n(test, 10, replace=FALSE)
test_predictions <- predict(xgb_fit, test_sample, "raw")
actual_prices <- round(test_sample$price, 0)
predicted_prices <- round(test_predictions, 0)
data.frame(actual=actual_prices, 
    predicted=predicted_prices, 
    difference=actual_prices-predicted_prices)
```

|actual|predicted|difference|
|--- |--- |--- |
|680000|566726|113274|
|1400000|1502961|-102961|
|400000|465854|-65854|
|468000|382870|85130|
|220000|208510|11490|
|525000|553434|-28434|
|404000|559599|-155599|
|327000|316226|10774|
|475000|460288|14712|
|443000|431310|11690|


Is this good? Well, personally I expected more. However, there are certainly more things to try if you are up to it. One interesting question that arises after receiving prediction is: How sure the model is that the price is what he tells us it is? But that is a topic for another post.

# References

[RMSE explained](https://www.youtube.com/watch?v=IBOXR05NMfc) <br/>
[Gradient Boosting explained](https://www.youtube.com/watch?v=WZvPUGNJg18) <br/>
[Dataset attributes explained](http://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r) <br/>
[More information for the attributes](https://www.kaggle.com/forums/f/1447/house-sales-in-king-county-usa/t/23194/variable-explanation)

### XGBoost

[Introduction to XGBoost](https://www.r-bloggers.com/an-introduction-to-xgboost-r-package/) <br/>
[Optimizing XGBoost](http://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1) <br/>
[Parameter tuning in XGBoost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

### caret

[Tuning parameters in caret](https://topepo.github.io/caret/model-training-and-tuning.html#alternate-tuning-grids)


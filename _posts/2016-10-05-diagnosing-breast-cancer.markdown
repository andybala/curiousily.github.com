---
layout: post
title:  "Diagnosing Breast Cancer from image data"
date:   2016-10-05 11:47:43 +0300
categories: ["data-science"]
excerpt: "We will examine a dataset containing features of digitized biopsy images from patients with breast lumps. We will train K-means clustering model to classify the type of tumor. All this using R!"
---

Detecting breast (or any other type of) cancer before noticing symptoms is a key first step in fighting the disease. The process involves examining breast tissue for lumps or masses. Fine needle aspirate (FNA) biopsy is performed if such irregularity is found. The extracted tissue is then examined under a microscope by a clinician.

Can a machine help the clinician do a better job? Can the doctor focus more on treating the disease rather than detecting it? Recently, Deep Learning (DL) has seen major advances in the area of computer vision. Naturally, some scientists tried to apply it to breast cancer detection - and [did so with great success](https://blogs.nvidia.com/blog/2016/09/19/deep-learning-breast-cancer-diagnosis/)!

Here, we will look at a dataset created by [Dr. William H. Wolberg, W. Nick Street and Olvi L. Mangasarian](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) from the University of Wisconsin. Each row describes features of the cell nuclei present in the digitized image of the FNA along with the diagnosis (M = malignant, B = benign) and ID of a patient with a lump in her breast.

Here is a list of the measured cell nuclei features:

* radius (mean of distances from center to points on the perimeter)
* texture (standard deviation of gray-scale values)
* perimeter
* area
* smoothness (local variation in radius lengths)
* compactness (perimeter^2 / area - 1.0)
* concavity (severity of concave portions of the contour)
* concave points (number of concave portions of the contour)
* symmetry
* fractal dimension ("coastline approximation" - 1)

Can we predict whether the lump is benign or malignant?

{:.center}
![jpeg]({{ site.url }}/assets/1.diagnosing_breast_cancer_files/biopsy.jpg)

{:.center}
*Sample image from which the cell nuclei features are extracted*

# Fire up R and load some libraries


```R
library(ggplot2)
library(Amelia)
library(class)
library(gmodels)

set.seed(42)
```

# Exploration


```R
df <- read.csv("data/breast_cancer.csv", stringsAsFactors = FALSE)
```


```R
print(paste("rows:", nrow(df), "cols:", ncol(df)))
```

    [1] "rows: 569 cols: 32"


Let's remove the ID column and recode the diagnosis.


```R
df <- df[-1]
df$diagnosis <- factor(df$diagnosis, levels = c("B", "M"),
                         labels = c("Benign", "Malignant"))
```

Do we have missing data?


```R
missmap(df, main="Missing Data Map", col=c("#FF4081", "#3F51B5"), 
        legend=FALSE)
```


{:.center}
![png]({{ site.url }}/assets/1.diagnosing_breast_cancer_files/1.diagnosing_breast_cancer_9_0.png)


Nope. That's good! What is the distribution for the both types of cancer?


```R
barplot(table(df$diagnosis), xlab = "Type of tumor", ylab="Numbers per type")
```


{:.center}
![png]({{ site.url }}/assets/1.diagnosing_breast_cancer_files/1.diagnosing_breast_cancer_11_0.png)


Let's see if we can differentiate between tumor types using some features (randomly chosen?):


```R
qplot(radius_mean, data=df, colour=diagnosis, geom="density",
      main="Radius mean for each tumor type")
```


{:.center}
![png]({{ site.url }}/assets/1.diagnosing_breast_cancer_files/1.diagnosing_breast_cancer_13_1.png)



```R
qplot(smoothness_mean, data=df, colour=diagnosis, geom="density",
      main="Smoothness mean for each tumor type")
```


{:.center}
![png]({{ site.url }}/assets/1.diagnosing_breast_cancer_files/1.diagnosing_breast_cancer_14_1.png)



```R
qplot(concavity_mean, data=df, colour=diagnosis, geom="density",
      main="Concavity mean for each tumor type")
```


{:.center}
![png]({{ site.url }}/assets/1.diagnosing_breast_cancer_files/1.diagnosing_breast_cancer_15_1.png)


# Preprocess the data

Let's normalize (scale every value in our dataset in the range [0:1]) our data. This will become handy when we try to classify the tumor type for each patient.


```R
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

df_normalized <- as.data.frame(lapply(df[2:31], normalize))
```

Additionaly, let's create a scaled version of our dataset too! The formula for scaling is the following:

$$\frac{x - mean(x)}{\sigma(x)}$$

where $x$ is a vector that contains real numbers.


```R
df_scaled <- as.data.frame(scale(df[-1]))
```

# Splitting our data

Now, let's split our dataset into 3 new - training, test and validation. First, let's put aside 150 rows for test/validation and use the rest for training:


```R
train_idx <- sample(nrow(df_normalized), nrow(df_normalized) - 150, 
                    replace = FALSE)
df_normalized_train <- df_normalized[train_idx, ]
```

Let's use 100 of the rest for testing and 50 for validation:


```R
test_validation_idx <- seq(1:nrow(df_normalized))[-train_idx]
test_idx <- sample(test_validation_idx, 100, replace = FALSE)
validation_idx <- test_validation_idx[-test_idx]

df_normalized_test <- df_normalized[test_idx, ]
df_normalized_validation <- df_normalized[validation_idx, ]
```

# Predicting tumor type

We will use simple k-means clustering algorithm to predict whether a patient has a benign or malignant tumor.


```R
df_train_labels <- df[train_idx, 1]
df_test_labels <- df[test_idx, 1]
df_validation_labels <- df[validation_idx, 1]

df_normalized_pred_labels <- knn(train = df_normalized_train, 
                                 test = df_normalized_test, 
                                 cl = df_train_labels, 
                                 k = 21)
```

Ok, that was quick. How did we do? Let's evaluate our model using a cross table and see:


```R
evaluate_model <- function(expected_labels, predicted_labels) {
    CrossTable(x = expected_labels, y = predicted_labels, prop.chisq=FALSE)
    true_predctions <- table(expected_labels == predicted_labels)["TRUE"]
    correct_predictions <- true_predictions / length(predicted_labels)
    print(paste("Correctly predicted: ", correct_predictions))
}
```


```R
evaluate_model(df_test_labels, df_normalized_pred_labels)
```

    
     
       Cell Contents
    |-------------------------|
    |                       N |
    |           N / Row Total |
    |           N / Col Total |
    |         N / Table Total |
    |-------------------------|
    
     
    Total Observations in Table:  100 
    
     
                    | predicted_labels 
    expected_labels |    Benign | Malignant | Row Total | 
    ----------------|-----------|-----------|-----------|
             Benign |        60 |         0 |        60 | 
                    |     1.000 |     0.000 |     0.600 | 
                    |     0.952 |     0.000 |           | 
                    |     0.600 |     0.000 |           | 
    ----------------|-----------|-----------|-----------|
          Malignant |         3 |        37 |        40 | 
                    |     0.075 |     0.925 |     0.400 | 
                    |     0.048 |     1.000 |           | 
                    |     0.030 |     0.370 |           | 
    ----------------|-----------|-----------|-----------|
       Column Total |        63 |        37 |       100 | 
                    |     0.630 |     0.370 |           | 
    ----------------|-----------|-----------|-----------|
    
     
    [1] "Correctly predicted:  0.97"


Not bad, only 3 errors. Can we do better? Let's use our scaled dataset:


```R
df_scaled_train <- df_scaled[train_idx, ]
df_scaled_test <- df_scaled[test_idx, ]
df_scaled_validation <- df_scaled[validation_idx, ]
```


```R
df_scaled_pred_labels <- knn(train = df_scaled_train, 
                             test = df_scaled_test, 
                             cl = df_train_labels, 
                             k = 21)
```


```R
evaluate_model(df_test_labels, df_scaled_pred_labels)
```

    
     
       Cell Contents
    |-------------------------|
    |                       N |
    |           N / Row Total |
    |           N / Col Total |
    |         N / Table Total |
    |-------------------------|
    
     
    Total Observations in Table:  100 
    
     
                    | predicted_labels 
    expected_labels |    Benign | Malignant | Row Total | 
    ----------------|-----------|-----------|-----------|
             Benign |        60 |         0 |        60 | 
                    |     1.000 |     0.000 |     0.600 | 
                    |     0.938 |     0.000 |           | 
                    |     0.600 |     0.000 |           | 
    ----------------|-----------|-----------|-----------|
          Malignant |         4 |        36 |        40 | 
                    |     0.100 |     0.900 |     0.400 | 
                    |     0.062 |     1.000 |           | 
                    |     0.040 |     0.360 |           | 
    ----------------|-----------|-----------|-----------|
       Column Total |        64 |        36 |       100 | 
                    |     0.640 |     0.360 |           | 
    ----------------|-----------|-----------|-----------|
    
     
    [1] "Correctly predicted:  0.96"


Huh, even worse! Let's try different k values:


```R
train_and_evaluate <- function(train, test, train_labels, test_labels, k) {
    predicted_labels <- knn(train = train, test = test, 
                            cl = train_labels, k = k)
    evaluate_model(test_labels, predicted_labels)
}
```


```R
train_and_evaluate(df_normalized_train, df_normalized_test, 
                   df_train_labels, df_test_labels, 1)
```

    
     
       Cell Contents
    |-------------------------|
    |                       N |
    |           N / Row Total |
    |           N / Col Total |
    |         N / Table Total |
    |-------------------------|
    
     
    Total Observations in Table:  100 
    
     
                    | predicted_labels 
    expected_labels |    Benign | Malignant | Row Total | 
    ----------------|-----------|-----------|-----------|
             Benign |        60 |         0 |        60 | 
                    |     1.000 |     0.000 |     0.600 | 
                    |     0.952 |     0.000 |           | 
                    |     0.600 |     0.000 |           | 
    ----------------|-----------|-----------|-----------|
          Malignant |         3 |        37 |        40 | 
                    |     0.075 |     0.925 |     0.400 | 
                    |     0.048 |     1.000 |           | 
                    |     0.030 |     0.370 |           | 
    ----------------|-----------|-----------|-----------|
       Column Total |        63 |        37 |       100 | 
                    |     0.630 |     0.370 |           | 
    ----------------|-----------|-----------|-----------|
    
     
    [1] "Correctly predicted:  0.97"



```R
train_and_evaluate(df_normalized_train, df_normalized_test, 
                   df_train_labels, df_test_labels, 5)
```

    
     
       Cell Contents
    |-------------------------|
    |                       N |
    |           N / Row Total |
    |           N / Col Total |
    |         N / Table Total |
    |-------------------------|
    
     
    Total Observations in Table:  100 
    
     
                    | predicted_labels 
    expected_labels |    Benign | Malignant | Row Total | 
    ----------------|-----------|-----------|-----------|
             Benign |        60 |         0 |        60 | 
                    |     1.000 |     0.000 |     0.600 | 
                    |     0.952 |     0.000 |           | 
                    |     0.600 |     0.000 |           | 
    ----------------|-----------|-----------|-----------|
          Malignant |         3 |        37 |        40 | 
                    |     0.075 |     0.925 |     0.400 | 
                    |     0.048 |     1.000 |           | 
                    |     0.030 |     0.370 |           | 
    ----------------|-----------|-----------|-----------|
       Column Total |        63 |        37 |       100 | 
                    |     0.630 |     0.370 |           | 
    ----------------|-----------|-----------|-----------|
    
     
    [1] "Correctly predicted:  0.97"



```R
train_and_evaluate(df_normalized_train, df_normalized_test, 
                   df_train_labels, df_test_labels, 15)
```

    
     
       Cell Contents
    |-------------------------|
    |                       N |
    |           N / Row Total |
    |           N / Col Total |
    |         N / Table Total |
    |-------------------------|
    
     
    Total Observations in Table:  100 
    
     
                    | predicted_labels 
    expected_labels |    Benign | Malignant | Row Total | 
    ----------------|-----------|-----------|-----------|
             Benign |        60 |         0 |        60 | 
                    |     1.000 |     0.000 |     0.600 | 
                    |     0.952 |     0.000 |           | 
                    |     0.600 |     0.000 |           | 
    ----------------|-----------|-----------|-----------|
          Malignant |         3 |        37 |        40 | 
                    |     0.075 |     0.925 |     0.400 | 
                    |     0.048 |     1.000 |           | 
                    |     0.030 |     0.370 |           | 
    ----------------|-----------|-----------|-----------|
       Column Total |        63 |        37 |       100 | 
                    |     0.630 |     0.370 |           | 
    ----------------|-----------|-----------|-----------|
    
     
    [1] "Correctly predicted:  0.97"


Not much change. Let's see how our model performs on the validation set:


```R
train_and_evaluate(df_normalized_train, df_normalized_validation, 
                   df_train_labels, df_validation_labels, 21)
```

    
     
       Cell Contents
    |-------------------------|
    |                       N |
    |           N / Row Total |
    |           N / Col Total |
    |         N / Table Total |
    |-------------------------|
    
     
    Total Observations in Table:  127 
    
     
                    | predicted_labels 
    expected_labels |    Benign | Malignant | Row Total | 
    ----------------|-----------|-----------|-----------|
             Benign |        79 |         0 |        79 | 
                    |     1.000 |     0.000 |     0.622 | 
                    |     0.929 |     0.000 |           | 
                    |     0.622 |     0.000 |           | 
    ----------------|-----------|-----------|-----------|
          Malignant |         6 |        42 |        48 | 
                    |     0.125 |     0.875 |     0.378 | 
                    |     0.071 |     1.000 |           | 
                    |     0.047 |     0.331 |           | 
    ----------------|-----------|-----------|-----------|
       Column Total |        85 |        42 |       127 | 
                    |     0.669 |     0.331 |           | 
    ----------------|-----------|-----------|-----------|
    
     
    [1] "Correctly predicted:  0.952755905511811"


Our final accuracy is about 95%. What does this mean? If our model was to replace a doctor it would missclassify 6 malignant tumors as benign. This is bad! The other type of error (missclassifying benign tumor as malignant) is pretty bad too! So, improvement to the accuracy (in any way) might save lives! Can you improve the model?

P.S. This post was written as ipython notebook. Download it from [here](https://github.com/curiousily/curiousily.github.com/blob/master/_data/notebooks/1.diagnosing_breast_cancer.ipynb). The dataset can be download from [here](https://github.com/curiousily/curiousily.github.com/blob/master/_data/notebooks/data/breast_cancer.csv).

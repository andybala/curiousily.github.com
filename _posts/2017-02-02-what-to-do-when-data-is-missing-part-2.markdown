---
layout: post
title:  "What to do when data is missing? - Part II"
date:   2017-02-02 22:16:15 +0300
categories: ["data-science"]
excerpt: "Let's use a Deep Autoencoder to impute missing categorical data from a dataset describing physical characteristics of mushrooms. How well can we do it? Let's try it with Keras in Python."
---

Mushrooms, anyone? What if you have lots of data on mushrooms, yet some of it is missing? One important question you might want to answer is whether or not a particular specimen is edible or poisonous. Of course, your understanding of what a poisonous mushroom is might be quite different (hi to all from Netherlands), but I digress.

The dataset of interest will be (you guessed it) all about mushrooms. It describes physical characteristics of *8124* mushroom instances. The number of variables is 23, all of which are categorical. More information about the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Mushroom).

# 1. Autoencoders

{:.center}
![png]({{site.url}}/assets/12.what_to_do_when_data_is_missing_part_ii_files/mushroom_encoder.png)
*Which one do you like better?*

Strangely enough, an autoencoder is a model that given input data tries to predict it. It is used for unsupervised learning (That might not be entirely correct). Puzzling? First time I heard the concept I thought it must be a misunderstanding on my part. It wasn't.

More specifically, let's take a look at Autoencoder Neural Networks. This autoencoder tries to learn to approximate the following identity function:

$$\textstyle f_{W,b}(x) \approx x$$

While trying to do just that might sound trivial at first, it is important to note that we want to learn a compressed representation of the data, thus find structure. This can be done by limiting the number of hidden units in the model. Those kind of autoencoders are called *undercomplete*.

## 1.1 Choosing loss function

In order to learn something meaningful, autoencoders should try to minimize some function, called *reconstruction error*. The traditional *squared error* is often used:

$$\textstyle L(x,x') = ||\, x - x'||^2$$

## 2.1 Encoding the data

Our dataset contains categorical variables exclusively. We will use a standard approach for such cases - one-hot encoding. Furthermore, we have to handle cells with missing values. We will create a missing mask vector and append it to our one-hot encoded values. Missing values will be filled with some constant.

Let's take a look at this sample data:


```python
[
    ['blue stem', 'large cap'],
    [None, 'large cap'],
    ['green stem', None]
]
```

Our encoded data looks like this:

```python
[
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0], # no missing values
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 0], # missing value in 1st variable
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 1]  # missing value in 2nd variable
]
```


## 2.2 Reconstruction error

Our reconstruction error will be the mean squared error which hopefully will work for categorical data. Since we are using Keras, our function must adhere to some rules.

```python
def make_reconstruction_loss(n_features):

    def reconstruction_loss(input_and_mask, y_pred):
        X_values = input_and_mask[:, :n_features]
        X_values.name = "$X_values"

        missing_mask = input_and_mask[:, n_features:]
        missing_mask.name = "$missing_mask"
        observed_mask = 1 - missing_mask
        observed_mask.name = "$observed_mask"

        X_values_observed = X_values * observed_mask
        X_values_observed.name = "$X_values_observed"

        pred_observed = y_pred * observed_mask
        pred_observed.name = "$y_pred_observed"

        return mse(y_true=X_values_observed, y_pred=pred_observed)
    return reconstruction_loss
```

Additionally, we will use slightly modified mean squared error for assessing our progress during training. The function takes into account the mask of the input data:


```python
def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))
```

## 2.3 Getting everything just right

Of course, a couple of imports are needed to make everything work. Make sure you have the proper libraries installed (all available via pip install).


```python
import random
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from keras.objectives import mse
from keras.models import Sequential
from keras.layers.core import Dropout, Dense
from keras.regularizers import l1l2

from collections import defaultdict
```

```python
%matplotlib inline
```

## 2.4 Creating the model (the fun part)

The following implementation is heavily based on the one provided by [fancyimpute](https://github.com/hammerlab/fancyimpute/). Our NN has 3 hidden layers (with ReLU activation functions) with dropout probability set to 0.5. You can also choose two regularizers coefficients - the sum of the weights (L1) and sum the squares of the weights (L2). These are 0. The activation function for the output layer is sigmoid. It appears to work better than linear for our use case.

Here is the code for it, don't be afraid to fiddle with it:


```python
class Autoencoder:

    def __init__(self, data,
                 recurrent_weight=0.5,
                 optimizer="adam",
                 dropout_probability=0.5,
                 hidden_activation="relu",
                 output_activation="sigmoid",
                 init="glorot_normal",
                 l1_penalty=0,
                 l2_penalty=0):
        self.data = data.copy()
        self.recurrent_weight = recurrent_weight
        self.optimizer = optimizer
        self.dropout_probability = dropout_probability
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.init = init
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        
    def _get_hidden_layer_sizes(self):
        n_dims = self.data.shape[1]
        return [
            min(2000, 8 * n_dims),
            min(500, 2 * n_dims),
            int(np.ceil(0.5 * n_dims)),
        ]

    def _create_model(self):

        hidden_layer_sizes = self._get_hidden_layer_sizes()
        first_layer_size = hidden_layer_sizes[0]
        n_dims = self.data.shape[1]
        
        model = Sequential()

        model.add(Dense(
            first_layer_size,
            input_dim= 2 * n_dims,
            activation=self.hidden_activation,
            W_regularizer=l1l2(self.l1_penalty, self.l2_penalty),
            init=self.init))
        model.add(Dropout(self.dropout_probability))

        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(
                layer_size,
                activation=self.hidden_activation,
                W_regularizer=l1l2(self.l1_penalty, self.l2_penalty),
                init=self.init))
            model.add(Dropout(self.dropout_probability))

        model.add(Dense(
            n_dims,
            activation=self.output_activation,
            W_regularizer=l1l2(self.l1_penalty, self.l2_penalty),
            init=self.init))

        loss_function = make_reconstruction_loss(n_dims)

        model.compile(optimizer=self.optimizer, loss=loss_function)
        return model

    def fill(self, missing_mask):
        self.data[missing_mask] = -1

    def _create_missing_mask(self):
        if self.data.dtype != "f" and self.data.dtype != "d":
            self.data = self.data.astype(float)

        return np.isnan(self.data)

    def _train_epoch(self, model, missing_mask, batch_size):
        input_with_mask = np.hstack([self.data, missing_mask])
        n_samples = len(input_with_mask)
        n_batches = int(np.ceil(n_samples / batch_size))
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = input_with_mask[indices]

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_data = X_shuffled[batch_start:batch_end, :]
            model.train_on_batch(batch_data, batch_data)
        return model.predict(input_with_mask)

    def train(self, batch_size=256, train_epochs=100):
        missing_mask = self._create_missing_mask()
        self.fill(missing_mask)
        self.model = self._create_model()

        observed_mask = ~missing_mask

        for epoch in range(train_epochs):
            X_pred = self._train_epoch(self.model, missing_mask, batch_size)
            observed_mae = masked_mae(X_true=self.data,
                                    X_pred=X_pred,
                                    mask=observed_mask)
            if epoch % 50 == 0:
                print("observed mae:", observed_mae)

            old_weight = (1.0 - self.recurrent_weight)
            self.data[missing_mask] *= old_weight
            pred_missing = X_pred[missing_mask]
            self.data[missing_mask] += self.recurrent_weight * pred_missing
        return self.data.copy()
```

Whew, that was a large chunk of code. The important part is updating our data where values are missing. We use some predefined weight along with the predictions of our NN to update only the missing value cells. Here is a diagram of our model:

{:.center}
![jpeg]({{site.url}}/assets/12.what_to_do_when_data_is_missing_part_ii_files/autoencoder.png)
*The architecture of our Autoencoder*

# 3. Evaluation

Let's see how well our Autoencoder can impute missing data, shall we?

## 3.1 Preparing the data

We will use the encoding technique for our categorical data as discussed above. Let's load (and drop a column with missing values from) our dataset.


```python
df = pd.read_csv("data/mushrooms.csv")
df = df.drop(['sroot'], axis=1)
```

### 3.1.1 Making data dissapear

We will use MCAR as a driving process behind making missing data. Each data point in our dataset has 0.1 probability of being set to *NaN*.


```python
prob_missing = 0.1
df_incomplete = df.copy()
ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
for row, col in random.sample(ix, int(round(prob_missing * len(ix)))):
    df_incomplete.iat[row, col] = np.nan
```

### 3.1.2 Encoding

Let's one-hot encode our dataset using panda's <code>get_dummies</code> and apply our missing data points accordingly:


```python
missing_encoded = pd.get_dummies(df_incomplete)

for col in df.columns:
    missing_cols = missing_encoded.columns.str.startswith(str(col) + "_")
    missing_encoded.loc[df_incomplete[col].isnull(), missing_cols] = np.nan
```

## 3.2 Working out our Encoder (or just training it)

Finally, it is time to use our skillfully crafted model. We will train it for 300 epochs with a batch size of 256. All else is left to it's default options.


```python
imputer = Autoencoder(missing_encoded.values)
complete_encoded = imputer.train(train_epochs=300, batch_size=256)
```

    observed mae: 0.230737793827
    observed mae: 0.0371880909997
    observed mae: 0.0268712357369
    observed mae: 0.0232342579129
    observed mae: 0.0210763975367
    observed mae: 0.0191748421237

That's all folks! Just kidding. The error seems to decrease so it might be wise to train for a few (or much more) epochs. 

That run surprisingly fast on my 2.6 Ghz i7 (2013) with 16 Gb DDR3 ram. Should definitely try it out on a CUDA-enabled machine.

## 3.3 How well we did?

We have to do a little more housekeeping before answering that one. Here is an example of a row from our imputed dataset:


```python
complete_encoded[10, :10]
```




    array([  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             0.00000000e+00,   1.00000000e+00,   4.18084731e-17,
             5.83600606e-20])



Not what you expected? That's all right. Let's use maximum likelihood estimation (MLE) to pick winning prediction for each missing data point.


```python
def mle(row):
    res = np.zeros(row.shape[0])
    res[np.argmax(row)] = 1
    return res

col_classes = [len(df[c].unique()) for c in df.columns]

dummy_df = pd.get_dummies(df)

mle_complete = None

for i, cnt in enumerate(col_classes):
    start_idx = int(sum(col_classes[0:i]))
    col_true = dummy_df.values[:, start_idx:start_idx+cnt]
    col_completed = complete_encoded[:, start_idx:start_idx+cnt]
    mle_completed = np.apply_along_axis(mle, axis=1, arr=col_completed)
    if mle_complete is None:
        mle_complete = mle_completed
    else:
        mle_complete = np.hstack([mle_complete, mle_completed])
```

Let's see what we've got:


```python
mle_complete[10, :10]
```




    array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.])



That's more like it! Now we just have to reverse that <code>get_dummies</code> encoding...


```python
def reverse_dummy(df_dummies):
    pos = defaultdict(list)
    vals = defaultdict(list)

    for i, c in enumerate(df_dummies.columns):
        if "_" in c:
            k, v = c.split("_", 1)
            pos[k].append(i)
            vals[k].append(v)
        else:
            pos["_"].append(i)

    df = pd.DataFrame({k: pd.Categorical.from_codes(
                              np.argmax(df_dummies.iloc[:, pos[k]].values, axis=1),
                              vals[k])
                      for k in vals})

    df[df_dummies.columns[pos["_"]]] = df_dummies.iloc[:, pos["_"]]
    return df

rev_df = reverse_dummy(pd.DataFrame(data=mle_complete, columns=dummy_df.columns))
rev_df = rev_df[list(df.columns)]
```

How much incorrect data points we have?


```python
incorrect = (rev_df != df)
incorrect_cnts = incorrect.apply(pd.value_counts)
incorrect_sum = incorrect_cnts.sum(axis=1)
incorrect_sum[1]
```




    3807.0



Whoops, that sounds like a lot. Maybe we didn't do well overall? But how much are missing?


```python
missing = df_incomplete.apply(pd.isnull)
missing_cnts = missing.apply(pd.value_counts)
missing_sum = missing_cnts.sum(axis=1)
missing_sum[1]
```




    17873



Whew, that is a lot, too! Okay, I'll stop teasing you, here is the accuracy of our imputation:


```python
accuracy = 1.0 - (incorrect_sum[1] / missing_sum[1])
print(accuracy)
```

    0.786997146534


Roughly 79%. Not bad considering the amount of data we have. But we haven't compared that to other approaches for data imputation (well, I have).

# 4. Conclusion(s)

<div class="center">
    <iframe width="100%" height="360" src="https://www.youtube.com/embed/5FmL80fVMxo" frameborder="0" allowfullscreen></iframe>
</div>

And once again, the day is saved thanks to the powerp... the Autoencoder. It was quite fun and easy to build this model using Keras. Probably it will be even easier in the near future since Keras will be integrated directly into TensorFlow. Some questions that require further investigation remain unanswered: 
* Is this model effective for categorical data imputation? 
* Was our data highly correlated and easy to predict?
* Can other models perform better? If so, when?
* Can we incorporate uncertainty into our imputation? So we have a degree of belief of how good the imputed value is.

# References

[Keras](https://keras.io/) - The library we used to build the Autoencoder<br/>
[fancyimpute](https://github.com/hammerlab/fancyimpute/) - Most of the Autoencoder code is taken from this awesome library<br/>
[Autoencoders](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/) - Unsupervised Feature Learning and Deep Learning on Autoencoders<br/>
[Denoising Autoencoders](http://deeplearning.net/tutorial/dA.html) - Tutorial on Denoising Autoencoders with short review on Autoencoders<br/>
[Data Imputation on Electronic Health Records](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5144587/) - Great use of Deep Autoencoders to impute medical records data<br/>
[Data Imputation using Autoencoders in Biology](http://www.biorxiv.org/content/biorxiv/early/2016/06/07/054775.full.pdf) - Another great use of Autoencoders for imputation

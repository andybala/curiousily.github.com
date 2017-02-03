---
layout: post
title:  "What to do when data is missing? - Part II"
date:   2017-02-02 22:16:15 +0300
categories: ["data-science"]
excerpt: "Let's use a Deep Autoencoder to impute missing categorical data from a dataset describing physical characteristics of mushrooms. How well can we do it? Let's try it with Keras in Python."
---

Mushrooms, anyone? What if you have lots of data on mushrooms, yet some of it is missing? One important question you might want to answer is whether or not a particular specimen is edible or poisonous. Of course, your understanding of what a poisonous mushroom is might be quite different (hi to all from Netherlands), but I digress.

The dataset of interest will be (you guessed it) all about mushrooms. It describes physical characteristics of *8124* mushroom instances. The number of variables is 23, all of which are categorical. More information about the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Mushroom).

# Autoencoders

{:.center}
![png]({{site.url}}/assets/12.what_to_do_when_data_is_missing_part_ii_files/mushroom_encoder.png)
*Which one do you like better?*

Strangely enough, an autoencoder is a model that given input data tries to predict it. It is used for unsupervised learning (That might not be entirely correct). Puzzling? First time I heard the concept I thought it must be a misunderstanding on my part. It wasn't.

More specifically, let's take a look at Autoencoder Neural Networks. This autoencoder tries to learn to approximate the following identity function:

$$\textstyle f_{W,b}(x) \approx x$$

While trying to do just that might sound trivial at first, it is important to note that we want to learn a compressed representation of the data, thus find structure. This can be done by limiting the number of hidden units in the model. Those kind of autoencoders are called *undercomplete*.

## Choosing loss function

In order to learn something meaningful, autoencoders should try to minimize some function, called *reconstruction error*. The traditional *squared error* is often used:

$$\textstyle L(x,x') = ||\, x - x'||^2$$

# Creating an Autoencoder

We will use Keras to create a simple Deep Autoencoder. Before getting to the fun part, though, we have some housekeeping to do. How should we encode the categorical data along with the missing values? Will that affect our reconstruction error?

## Encoding the data

Our dataset contains categorical variables exclusively. We will use a standard approach for such cases - one-hot encoding. Furthermore, we have to handle cells with missing values. We will create a missing mask vector and append it to our one-hot encoded values. Missing values will be filled with some constant.

Let's take a look at this sample data:


```python
[
    ['blue stem', 'large cap'],
    [np.nan, 'large cap'],
    ['green stem', np.nan]
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

## Reconstruction error

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

## Getting everything just right

Of course, a couple of imports are needed to make everything work. Make sure you have the proper libraries installed (all available via pip install).


```python
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

## The fun part

Here is the code for it, don't be afraid to fiddle with it:


```python
class Autoencoder:

    def __init__(self, data,
                 recurrent_weight = 0.5):
        self.data = data
        self.recurrent_weight = recurrent_weight

    def _create_model(self):
        n_dims = self.data.shape[1]
        hidden_layer_sizes = [
            min(2000, 8 * n_dims),
            min(500, 2 * n_dims),
            int(np.ceil(0.5 * n_dims)),
        ]

        first_layer_size = hidden_layer_sizes[0]

        hidden_activation = 'relu'
        output_activation = 'sigmoid'
        init="glorot_normal"
        l1_penalty = 0
        l2_penalty = 0
        dropout_probability=0.5

        model = Sequential()

        model.add(Dense(
            first_layer_size,
            input_dim= 2 * n_dims,
            activation=hidden_activation,
            W_regularizer=l1l2(l1_penalty, l2_penalty),
            init=init))
        model.add(Dropout(dropout_probability))

        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(
                layer_size,
                activation=hidden_activation,
                W_regularizer=l1l2(l1_penalty, l2_penalty),
                init=init))
            model.add(Dropout(dropout_probability))

        model.add(Dense(
            n_dims,
            activation=output_activation,
            W_regularizer=l1l2(l1_penalty, l2_penalty),
            init=init))

        loss_function = make_reconstruction_loss(n_dims)

        optimizer = "adam"
        model.compile(optimizer=optimizer, loss=loss_function)
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
        model = self._create_model()

        observed_mask = ~missing_mask

        for _ in range(train_epochs):
            X_pred = self._train_epoch(model, missing_mask, batch_size)
            observed_mae = masked_mae(X_true=self.data,
                                    X_pred=X_pred,
                                    mask=observed_mask)
            print("observed mae:", observed_mae)

            old_weight = (1.0 - self.recurrent_weight)
            self.data[missing_mask] *= old_weight
            pred_missing = X_pred[missing_mask]
            self.data[missing_mask] += self.recurrent_weight * pred_missing
        return self.data
```


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

def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))

def mle(row):
    res = np.zeros(row.shape[0])
    res[np.argmax(row)] = 1
    return res
```


```python
df = pd.read_csv("data/mushrooms.csv")
df = df.drop(['sroot'], axis=1)
df_incomplete = df.copy()
import random
ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
for row, col in random.sample(ix, int(round(.1*len(ix)))):
    df_incomplete.iat[row, col] = np.nan
dummy_df = pd.get_dummies(df)
dummy_incomplete_df = pd.get_dummies(df_incomplete)

for col in df.columns:
    dummy_incomplete_df.loc[df_incomplete[col].isnull(), dummy_incomplete_df.columns.str.startswith(str(col) + "_")] = np.nan

imputer = Autoencoder(dummy_incomplete_df.copy().values)
imputer.train(train_epochs=10)

dummy_completed_df = imputer.data
```

    observed mae: 0.249136511569
    observed mae: 0.1506028165
    observed mae: 0.12032281598
    observed mae: 0.107268109797
    observed mae: 0.097324605074
    observed mae: 0.0945542617275
    observed mae: 0.0858770286434
    observed mae: 0.0807686920289
    observed mae: 0.0759420838568
    observed mae: 0.0723345458635



```python
col_classes = [len(df[c].unique()) for c in df.columns]

mle_complete_df = None

for i, cnt in enumerate(col_classes):
    start_idx = int(sum(col_classes[0:i]))
    col_true = dummy_df.values[:, start_idx:start_idx+cnt]
    col_completed = dummy_completed_df[:, start_idx:start_idx+cnt]
    mle_completed = np.apply_along_axis(mle, axis=1, arr=col_completed)
    if mle_complete_df is None:
        mle_complete_df = mle_completed
    else:
        mle_complete_df = np.hstack([mle_complete_df, mle_completed])

rev_df = reverse_dummy(pd.DataFrame(data=mle_complete_df, columns=dummy_df.columns))
rev_df = rev_df[list(df.columns)]
incorrect = (rev_df != df)
incorrect_cnts = incorrect.apply(pd.value_counts)
incorrect_sum = incorrect_cnts.sum(axis=1)

missing = df_incomplete.apply(pd.isnull)
missing_cnts = missing.apply(pd.value_counts)
missing_sum = missing_cnts.sum(axis=1)

accuracy = 1.0 - (incorrect_sum[1] / missing_sum[1])
print(accuracy)
```

    0.769876349801


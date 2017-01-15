---
layout: post
title:  "What do when data is missing? - Part I"
date:   2017-01-15 12:43:00 +0300
categories: ["data-science"]
excerpt: "Yes! You've got the coolest dataset on your hard drive. Countless hours of fun are waiting for you. Except, some rows have missing values and your model might not be happy with those. But you have the perfect solution! You can just ignore them (nobody said delete them, right?)! Now, why this might not be the best idea? Let's find out using R."
---

Let's begin by you trying to watch this video:

<div class="center">
    <iframe width="100%" height="360" src="https://www.youtube.com/embed/SyWydWLmszE" frameborder="0" allowfullscreen></iframe>
</div>

This video has 300k+ views with around 1.6k positive reactions. It might be worth watching, considering it is in some cryptic foreign language (Know Bulgarian? That's how it feels to have some serious superpowers!). But you have a problem, right? You don't know what the video is all about!

Well, you are missing some data. In your case, subtitles might be a good start. However, knowing the language, experiencing the emotions that the actor is trying to convey might be much better! So, it is not enough to have some data, it is best to have the real thing.

# 1. Problem definition

Let's try to translate the problem into R

Let's say we are trying to find the mean weight of all women in New York City. We weight 1000 women and write down their weights. The first 5 begin as follows:


```R
weights <- c(55.4, 48.5, 58.5, 63.4, 67.4)
```

We can easily get the mean of that sample:


```R
mean(weights_sample)
```


58.64


But let's say that the last value was missing and we are trying to get the mean of the sample:


```R
weights_missing <- c(55.4, 48.5, 58.5, 63.4, NA)
mean(weights_missing)
```


    [1] NA


Huh? The result is undefined. What did you expect? Can we hack our way around this? Of course, let's be sneaky and just ignore the missing value.


```R
mean(weights_missing, na.rm = TRUE)
```


56.45


Yep, we've done it! Good job! And this is the end of the post. Or is it?

Just imagine this title in your favorite newspaper tomorrow morning:

"Scientists found that the average weight of a woman in New York City is 56.45 kilograms"

Is this correct? Can we just ignore the values that are missing? And the answer is - it depends.

# 2. Come on, this can't be that hard!

So, what is the best way to handle missing data? As [Orchard and Woodbury (1972, p. 697)](#cite-orchard1972missing) remarked:

`Obviously, the best way to treat missing data is not to have them.`

Of course, they knew that this was impossible ideal to achieve in practice.

[Allison (2002, p. 1)](#cite-allison2001missing) observed that:

`Sooner or later (usually sooner), anyone who does statistical analysis runs into problems with missing data.`

So, how should we deal with this?

## 2.1 MCAR, MAR and MNAR

Before answering the question above we have to familiarize ourselves with the strange terms described above.

[Rubin (1976)](#cite-rubin1976inference) defined three categories of missing data problems. He stated that every data point has some probability of being missing. What defines those probabilities is a process called the missing data mechanism or response mechanism. A model for this process is called missing data model or response model.

### 2.1.1 MCAR

When the probability of a data point being missing is the same, the data are said to be missing completely at random (MCAR). For example, let's say we go to every woman in NYC, flip a coin and weight her only if the coin show heads. Each data point in this will have the same probability of being missing, e.g. the probability of the coin landing heads.

### 2.1.2 MAR

If the probability of being missing is the same only within the groups defined by the observed data, then the data are missing at random (MAR). Let's take our previous example and modify it a bit. Let's say we weigh only women that are between 30 and 40 years of age.

MAR is a generalization of MCAR. Thus, much more useful in practice.

### 2.1.3 MNAR (or NMAR)

You might already know what this category represents. Missing not at random (or Not missing at random) means that the probability of being missing is unknown. For example, the heavier ladies from the example above might not be so eager to weigh themselves, but this might not be known by us.

MNAR is really hard to handle in practice. Strategies to handle this case include finding the cause of missing data or provide hypothesis (what-if analysis) and see how sensitive the results are.

## 2.2 Simple fixes

### 2.2.1 Listwise deletion

Just delete the rows that have missing data. It is a really convenient way to handle the issue. Under MCAR this might be an ok thing to do. However, [Schafer and Graham (2002)](#cite-schafer2002missing) demonstrated using an elegant simulation the bias of listwise deletion under MAR and MNAR.

### 2.2.2 Pairwise deletion (Available-case analysis)

This method tries to be a bit smarter than listwise deletion. Pairwise deletion works by calculating the means and (co)variances on all observed data. Given two variables $$X$$ and $$Y$$, the mean of $$X$$ is based on the observed $$X$$ values, the same goes for $$Y$$. To calculate correlation and covariance, pairwise deletion takes all data rows where $$X$$ and $$Y$$ are present.


```R
library(tidyverse)

heights_missing <- c(171, 165, NA, 173, 172)
df <- tibble(heights = heights_missing, weights = weights_missing)
```


```R
lapply(df, mean, na.rm = T)
```


<dl>
	<dt>$heights</dt>
		<dd>170.25</dd>
	<dt>$weights</dt>
		<dd>56.45</dd>
</dl>




```R
cor(df, use = "pair")
```


<table>
<thead><tr><th></th><th scope="col">heights</th><th scope="col">weights</th></tr></thead>
<tbody>
	<tr><th scope="row">heights</th><td>1.0000000</td><td>0.9480866</td></tr>
	<tr><th scope="row">weights</th><td>0.9480866</td><td>1.0000000</td></tr>
</tbody>
</table>




```R
cov(df, use = "pair")
```


<table>
<thead><tr><th></th><th scope="col">heights</th><th scope="col">weights</th></tr></thead>
<tbody>
	<tr><th scope="row">heights</th><td>12.91667</td><td>29.43333</td></tr>
	<tr><th scope="row">weights</th><td>29.43333</td><td>38.93667</td></tr>
</tbody>
</table>



This method is an easy way to compute the mean, correlations, and covariances under MCAR. However, the estimates can be biased when the data is not MCAR.

The big selling point of pairwise deletion is that it tries to use all available data.

### 2.2.3 Mean imputation

This one is real simple. Replace every missing value by the mean. Let's try it:


```R
library(mice)

imputed <- mice(df, method = "mean", m = 1, maxit = 1)
```

    
     iter imp variable
      1   1  heights  weights



```R
completed <- complete(imputed, 1)
completed
```


<table>
<thead><tr><th scope="col">heights</th><th scope="col">weights</th></tr></thead>
<tbody>
	<tr><td>171.00</td><td>55.40 </td></tr>
	<tr><td>165.00</td><td>48.50 </td></tr>
	<tr><td>170.25</td><td>58.50 </td></tr>
	<tr><td>173.00</td><td>63.40 </td></tr>
	<tr><td>172.00</td><td>56.45 </td></tr>
</tbody>
</table>



Let's have another look at the initial data and the correlations:


```R
df
```


<table>
<thead><tr><th scope="col">heights</th><th scope="col">weights</th></tr></thead>
<tbody>
	<tr><td>171 </td><td>55.4</td></tr>
	<tr><td>165 </td><td>48.5</td></tr>
	<tr><td> NA </td><td>58.5</td></tr>
	<tr><td>173 </td><td>63.4</td></tr>
	<tr><td>172 </td><td>  NA</td></tr>
</tbody>
</table>




```R
cor(df, use = "pair")
```


<table>
<thead><tr><th></th><th scope="col">heights</th><th scope="col">weights</th></tr></thead>
<tbody>
	<tr><th scope="row">heights</th><td>1.0000000</td><td>0.9480866</td></tr>
	<tr><th scope="row">weights</th><td>0.9480866</td><td>1.0000000</td></tr>
</tbody>
</table>




```R
cor(completed, use = "pair")
```


<table>
<thead><tr><th></th><th scope="col">heights</th><th scope="col">weights</th></tr></thead>
<tbody>
	<tr><th scope="row">heights</th><td>1.0000000</td><td>0.8927452</td></tr>
	<tr><th scope="row">weights</th><td>0.8927452</td><td>1.0000000</td></tr>
</tbody>
</table>



The correlations are different. That's no good. Granted, this is a bit contrived example, but you can see how this can happen in the real world.

Mean imputation offers a simple and fast fix for missing data. However, it will bias any estimate other than the mean when data are not MCAR.

### 2.2.4 Stochastic regression imputation

This method creates regression model and uses it for completing missing values. Additionally, it adds noise to the predictions, thus reducing the correlation between the variables. Let's have a look:


```R
imputed <- mice(df, method = "norm.nob", m = 1, maxit = 1, seed = 42)
```

    
     iter imp variable
      1   1  heights  weights


`method = "norm.nob"` tells `mice` to use the stochastic regression method.


```R
completed <- complete(imputed, 1)
completed
```


<table>
<thead><tr><th scope="col">heights</th><th scope="col">weights</th></tr></thead>
<tbody>
	<tr><td>171.0000</td><td>55.40000</td></tr>
	<tr><td>165.0000</td><td>48.50000</td></tr>
	<tr><td>169.3624</td><td>58.50000</td></tr>
	<tr><td>173.0000</td><td>63.40000</td></tr>
	<tr><td>172.0000</td><td>61.86775</td></tr>
</tbody>
</table>




```R
cor(df, use = "pair")
```


<table>
<thead><tr><th></th><th scope="col">heights</th><th scope="col">weights</th></tr></thead>
<tbody>
	<tr><th scope="row">heights</th><td>1.0000000</td><td>0.9480866</td></tr>
	<tr><th scope="row">weights</th><td>0.9480866</td><td>1.0000000</td></tr>
</tbody>
</table>




```R
cor(completed, use = "pair")
```


<table>
    <thead>
        <tr><th></th><th scope="col">heights</th><th scope="col">weights</th></tr>
    </thead>
    <tbody>
        <tr><th scope="row">heights</th><td>1.0000000</td><td>0.9240839</td></tr>
        <tr><th scope="row">weights</th><td>0.9240839</td><td>1.0000000</td></tr>
    </tbody>
</table>

Now, this is better. Stochastic regression imputation is a nice step in the right direction, but it does not solve all our problems. However, the idea of using other variable values is fundamental for more advanced techniques.

# 3. Where to go from here?

A couple of big questions remains still open. How can we tell if the imputation is good? Can we improve on those simple methods? Let's try to answer those in the next post.

<!--bibtex

@inproceedings{orchard1972missing,
  title={A missing information principle: theory and applications},
  author={Orchard, Terence and Woodbury, Max A and others},
  booktitle={Proceedings of the 6th Berkeley Symposium on mathematical statistics and probability},
  volume={1},
  pages={697--715},
  year={1972},
  organization={University of California Press Berkeley, CA}
}

@book{allison2001missing,
  title={Missing data},
  author={Allison, Paul D},
  volume={136},
  year={2001},
  publisher={Sage publications}
}

@article{rubin1976inference,
  title={Inference and missing data},
  author={Rubin, Donald B},
  journal={Biometrika},
  volume={63},
  number={3},
  pages={581--592},
  year={1976},
  publisher={Biometrika Trust}
}

@article{schafer2002missing,
  title={Missing data: our view of the state of the art.},
  author={Schafer, Joseph L and Graham, John W},
  journal={Psychological methods},
  volume={7},
  number={2},
  pages={147},
  year={2002},
  publisher={American Psychological Association}
}

-->

# References

* <a name="cite-orchard1972missing"/>Orchard, Terence and Woodbury, Max A and others. 1972. _A missing information principle: theory and applications_.

* <a name="cite-allison2001missing"/>Allison, Paul D. 2001. _Missing data_.

* <a name="cite-rubin1976inference"/>Rubin, Donald B. 1976. _Inference and missing data_.

* <a name="cite-schafer2002missing"/>Schafer, Joseph L and Graham, John W. 2002. _Missing data: our view of the state of the art._.


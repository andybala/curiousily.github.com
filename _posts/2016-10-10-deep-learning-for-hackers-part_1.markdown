---
layout: post
title:  "Deep Learning for Hackers (Part I)"
date:   2016-10-10 9:37:28 +0300
categories: ["deep-learning"]
excerpt: "We will build your first Neural Network and train it using the Backpropagation algorithm. Your network will be tested on iris plant classification. All this awesomeness will be implemented in JavaScript."
---

You heard the hype, right? Machine learning and more specifically **Deep Learning (DL)** is the thing right now. But what is Deep Learning all about? Let's find out together, shall we?

# What is Deep Learning?

First, try to watch this:

<div class="center">
    <iframe width="640" height="360" src="https://www.youtube.com/embed/HJ58dbd5g8g" frameborder="0" allowfullscreen></iframe>
</div>

Ok, you got me! I don't really know and for that matter, nobody really does. But that is not really useful answer now, is it? Let's try that again: Deep Learning is a branch of Machine Learning that tries to move us closer to AI (or General AI). One can simplify things a bit and say: an Artificial Neural Network (ANN) with more than 1 hidden layer. Let's take a look at one ANN:

{:.center}
![image]({{site.url}}/assets/2.deep_learning_for_hackers_part_i_files/simple_neural_network.png)

A layer in ANN consists of a bunch of neurons. To make a network you stack several layers and connect them in some way (yes, there are different types of connections you can create). In this example, we have input layer with 4 neurons, 1 hidden layer with 3 neurosn and 1 neuron in the output layer.

### Why Deep Learning is powerful and popular right now?

One hypothesis would be that big companies like Google, Facebook, Microsoft are constantly talking/showing demos about Deep Learning, but more importantly:

* The end of feature engineering? Automatically tries to learn good features or representations instead of manually creating them.
* Automatic hierarchical representation? It tries to learn multiple levels of representation from raw data. Think of this as stacking more and more complex representation layers on top of each other.

### Where is the catch?

Large amounts of data (not necessarily labeled) are typically needed in order to really hit the sweet spot and get benefits from using Deep Learning. Furthermore, getting good performance from such large models still require careful tuning of many parameters.

### Who uses Deep Learning and for what?

Below list is by no means complete! Feel free to provide new interesting examples in the comments below.

#### **Automatic Machine Translation**

The task here is simple (or so it seems at first): given a word, phrase, sentence or whole document in let's say, French, translate it into English. How hard can it be? Some humans are pretty good at it, right? But what if you had an instant translator in your pocket, everywhere you go? Furthermore, what if it can translate from images? Well, that seems possible with Deep Learning:

<div class="center">
    <iframe width="640" height="360" src="https://www.youtube.com/embed/06olHmcJjS0" frameborder="0" allowfullscreen></iframe>
</div>

#### **Deep Reinforcement Learning is used for playing games**

Like playing Atari only from screen pixels and beating humans on Go

<div class="center">
    <iframe width="640" height="360" src="https://www.youtube.com/embed/TmPfTpjtdgg" frameborder="0" allowfullscreen></iframe>
</div>

# Building a Neural Network

```javascript
class NeuralNetwork {

    constructor(inputSize, outputSize, hiddenSize) {
        this.inputSize = inputSize
        this.outputSize = outputSize
        this.hiddenSize = hiddenSize

        this.weights1 = randnMatrix(inputSize, hiddenSize)
        this.weights2 = randnMatrix(hiddenSize, outputSize)
    }

    forward(x) {
        this.z2 = x.dot(this.weights1)
        this.a2 = sigmoidMatrix(this.z2)
        this.z3 = this.a2.dot(this.weights2)
        return sigmoidMatrix(this.z3)
    }
}
```

### Weights

```javascript
function randn() {
    var u = 1 - Math.random(); // Subtraction to flip [0, 1) to (0, 1].
    var v = 1 - Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) *
           Math.cos( 2.0 * Math.PI * v );
}

function randnMatrix(rows, columns) {

    var rands = []
    for(var i = 0; i < rows * columns; i++) {
        rands.push(randn())
    }
    return np.array(rands).reshape(rows, columns)
}
```

### Activation function

```javascript
function sigmoidMatrix(x) {

    x = x.clone()

    const sigmoidCwise = cwise({
        args: ['array'],
        body: function (z) {
            z = 1.0 / (1.0 + Math.exp(z))
        }
    })

    sigmoidCwise(x.selection)
    return x
}
```

# Evaluating our performance

```javascript
cost(x, y) {
    var yHat = this.forward(x)
    var J = 0.5 * y.subtract(yHat).pow(2).sum()
    return J
}
```

# Conclusion

We built a simple NN in Javacript. In the next part of the series we will train it using the Backpropagation algorithm.

# References

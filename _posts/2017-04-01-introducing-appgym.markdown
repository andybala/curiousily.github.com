---
layout: post
title: "AppGym - Android Apps as Reinforcement Learning Environment"
date: 2017-04-01 13:59:00 +0300
categories: ["data-science"]
excerpt: "Is it possible to use mobile apps as a training ground for your RL agents? You might need a tad bit more than sugar, spice and everything nice to do it."
---

The most important takeaway from this post is the fact that my new VIM config is pretty tight, I challenge you to find and use it yourself! Now that the fun part is over, let's begin by watching a video that was recently shown to me:

<div class="center">
    <iframe width="100%" height="360" src="https://www.youtube.com/embed/iFOus8ehxUc" frameborder="0" allowfullscreen></iframe>
</div>

Puzzled? You might've figured that the video had nothing to do with the topic of this post, again? Maybe, but isn't your immune system just epic? What if the Neutrophil was the agent with a task of finding and destroying the poor bacteria in the environment of your body. We have many algorithms/models to use as brains for our agents and tons of problems to solve. Proper environments, however, tend to be scarce. I guess, something between those lines might've sparked the idea of creating [Gym by OpenAI](https://gym.openai.com/). Like that wasn't enough the same team introduced the granddaddy of all - [Universe](https://universe.openai.com/).

More recently, even web browsers got some attention when [Mini World of Bits (Mini WOB)](http://alpha.openai.com/miniwob/index.html) was introduced. While the other environments were focused primarily on games and some physics simulations, WOB tries to present an environment that has more instantly applicable tasks. However, Mini WOB has tasks that are constrained within 210x160 pixels. Furthermore, web sites tend to contain lots of elements per page.

Are mobile apps the final frontier for RL agents? Highly doubtful. They might be an exciting next step, though. While the resolution of modern smartphones are close (or even greater) to those of desktop monitors, their physical size seems to hover around 5". Due to our humanly fat fingers (don't look at yours now!) the touchable elements per screen can't be that many. Some design guidelines even suggest that there should be one primary action per screen. So why not give it a go?

# Introducing AppGym

Let's create an environment that is as fun as games and probably more useful. How hard can it be? After all, Android development is so "easy" these days. No coding required (yeah, right). So, what do you need? 

## Requirements

* **State** - you need where you are within the app. A mobile app can have pretty substantial state space to search. How can we represent the current state **S** at time **t**? Just like in the Atari games we will use an image from the screen.
* **Actions** - you have the state and know where you at, what can you do? How can we use the elements on the screen? We can use [uiautomator](https://github.com/xiaocong/uiautomator) to create the action set.
* **Reward** - this one's easy. How much of the source code we've covered? The more, the merrier. Is it enough? No! Can you think of something more?
- **Done?** - if we accumulated enough of a reward during this training episode we can go to the next one.

## Implementing it

Let's try to stick the [**Env**](https://github.com/openai/gym/blob/master/gym/core.py#L13) interface (just in case we might integrate into it someday). Let's make an environment:

```python
env = AndroidEnv(app_package, dict(width=1080, height=1920))
```

Nothing magical so far. Just specifying the package of the app we want to run with and the resolution of the device (hopefully that will not be necessary in the future). Couple of additional things are happening under the hood:

- A connection to the device is established using the **uiautomator** lib
- Copy JaCoCo our preloaded [Nailgun](https://github.com/martylamb/nailgun) instance
- Forward a port from the Android device for a custom code coverage server
- Hope that the code coverage server was started during app initialization

Next up - resetting the state of the environment.

```python
state, actions = env.reset()
```

Here's what is done:

- Screenshot of the current screen is taken as start state
- Actions are extracted via the hierarchy provided by the **uiautomator**. Only clickable elements are considered (for now).

Now for the grand finale - the **step** function:

```python
action = choose_action(state, actions)
next_state, actions, reward, done = env.step(action)
```

- Get the current code coverage via the server that was started on the device
- Screenshot of the current screen
- Extract the possible actions from the screen using the view hierarchy

# Taking it for a spin

The code is at GitHub [here](https://github.com/appgym/appgym) and [here](https://github.com/curiousily/dissertation/tree/master/code). Feel free to poke around. It is not ready for the big league, yet. You can expect updated README and a *pip* package (hopefully around the corner). In some of the next posts we will explore how **AppGym** can be used for agents that are trying to take over QA people jobs! Is this even possible?

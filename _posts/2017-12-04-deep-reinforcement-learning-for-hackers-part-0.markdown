---
layout: post
title:  "Getting Your Feet Rewarded - Deep Reinforcement Learning for Hackers (Part 0)"
date:   2017-12-04 22:31:00 +0300
categories: ["Reinforcement Learning"]
excerpt: Reinforcement Learning might sound kind of complicated at first, but demystifying it can be a really fun way to learn more about Deep Learning (yes, Neural Networks), Markov Chains and Bayesian methods. Do you accept the challenge?
---

The best way to understand what Reinforcement Learning is to watch this video:

<div class="center">
    <iframe src="https://www.youtube.com/embed/0DQjCPBZaUk" frameborder="0" allowfullscreen></iframe>
</div>

Remember the first time you went behind the wheel of a car? Your dad, mom or driving instructor was next to you, waiting for you to mess something up. You had a clear goal - make a couple of turns and get to the supermarket for ice cream. The task was infinitely more fun if you had to learn to drive stick. Ah, good times. Too bad that your kids might never experience that. More on that later.

# What is Reinforcement Learning?

Reinforcement learning (RL) is learning what to do, given a situation and a set of possible actions to choose from, in order to maximize a reward. The learner, which we will call agent, is not told what to do, he must discover this by himself through interacting with the environment. The goal is to choose its actions in such a way that the cumulative reward is maximized. So, choosing the best reward now, might not be the best decision, in the long run. That is greedy approaches might not be optimal.

Back to you, behind the wheel with running engine, properly strapped seatbelt, adrenaline pumping and rerunning the latest Fast & Furious through your mind - you have a good feeling about this, the passenger next to you does not look that scared, after all...

How all of this relates to RL? Let's try to map your situation to an RL problem. Driving is really complex, so for your first lesson, your instructor will do everything, except turning the wheel. The environment is the nature itself and the agent is you. The state of the environment (situation) can be defined by the position of your car, surrounding cars, pedestrians, upcoming crossroads etc. You have 3 possible actions to choose from - turn left, keep straight and turn right. The reward is well defined - you will eat ice cream if you are able to reach the supermarket. Your instructor will give your intermediate rewards based on your performance. At each step (let's say once every second), you will have to make a decision - turn left, right or continue straight ahead. Whether or not the ice cream is happening is mostly up to you.

Let's summarize what we've learned so far. We have an agent and an environment. The environment gives the agent a state. The agent chooses an action and receives a reward from the environment along with the new state. This learning process continues until the goal is achieved or some other condition is met.

{:.center}
![png]({{site.url}}/assets/17.rl_for_hackers_part_0_files/reinforcement_learning.png)
*Source: https://phrasee.co/*

# Examples of Reinforcement Learning

Let's have a look at some example applications of RL:

## Cart-Pole Balancing

{:.center}
![png]({{site.url}}/assets/17.rl_for_hackers_part_0_files/cart_pole.png)

- **Goal** - Balance the pole on top of a moving cart <br/>

- **State** - angle, angular speed, position, horizontal velocity
- **Actions** - horizontal force to the cart
- **Reward** - 1 at each time step if the pole is upright

## Atari Games

{:.center}
![png]({{site.url}}/assets/17.rl_for_hackers_part_0_files/atari.png)

- **Goal** - Beat the game with the highest score <br/>

- **State** - Raw game pixels of the game
- **Actions** - Up, Down, Left, Right etc
- **Reward** - Score provided by the game

## DOOM

{:.center}
![png]({{site.url}}/assets/17.rl_for_hackers_part_0_files/doom.png)

- **Goal** - Eliminate all opponents <br/>

- **State** - Raw game pixels of the game
- **Actions** - Up, Down, Left, Right etc
- **Reward** - Positive when eliminating an opponent, negative when the agent is eliminated

## Training robots for Bin Packing

{:.center}
![png]({{site.url}}/assets/17.rl_for_hackers_part_0_files/robot_arm.png)
*Source: www.plasticsdist.com*

- **Goal** - Pick a device from a box and put it into a container <br/>

- **State** - Raw pixels of the real world
- **Actions** - Possible actions of the robot
- **Reward** - Positive when placing a device successfully, negative otherwise

You started thinking that all RL researchers are failed pro-gamers, didn't you? In practice, that doesn't seem to be the case. For example, somewhat "meta" applications include *["Designing Neural Network Architectures using Reinforcement Learning"](https://arxiv.org/abs/1611.02167)*.

# Formalizing the RL problem

Markov Decision Process (MDP) is mathematical formulations of the RL problem. They satisfy the Markov property:

**Markov property** - the current state completely represents the state of the environment (world). That is, the future depends only on the present.

An MDP can be defined by $$(S, A, R, P, \gamma)$$ where:

- $$S$$ - set of possible states
- $$A$$ - set of possible actions
- $$R$$ - probability distribution of reward given (state, action) pair
- $$P$$ - probability distribution over how likely any of the states is to be the new states, given (state, action) pair. Also known as transition probability.
- $$\gamma$$ - reward discount factor

## How MDPs work

At the initial time step $$t=0$$, the environment chooses initial state $$s_0 \sim p(s_0)$$. That state is used as a seed state for the following loop:

for $$t=0$$ until done:
 - The agent selects action $$a_t$$
 - The environment chooses reward $$r_t \sim R(. \vert\, s_t, a_t)$$ and next state $$s_{t + 1} \sim P(. \vert\, s_t, a_t)$$
 - The agent receives reward $$r_t$$ and next state $$s_{t + 1}$$
 
More formally, the environment does not choose, it samples from the reward and transition probability distributions.

What is the objective of all this? Find a function $$\pi^*$$, known as optimal policy, that maximizes the cumulative discounted reward: $$\sum_{t > 0}\gamma^t r_t$$.

A policy $$\pi$$ is a function that maps state $$s$$ to action $$a$$, that our agent believes is the best given that state.

# Your first MDP

Let's get back to you, cruising through the neighborhood, dreaming about that delicious ice cream. Here is one possible situation, described as an MDP:

{:.center}
![png]({{site.url}}/assets/17.rl_for_hackers_part_0_files/drive_grid_world.png)

Your objective is to get to the bitten ice cream on a stick, without meeting a zombie. The reasoning behind the new design is based on solid data science - people seem to give a crazy amount of cash for a bitten fruit and everybody knows that candy is much tastier. Putting it together you get "the all-new ice cream". And honestly, it wouldn't be cool to omit the zombies, so there you have it.

The state is fully described by the grid. At the first step you have the following actions:

{:.center}
![png]({{site.url}}/assets/17.rl_for_hackers_part_0_files/drive_grid_world_actions.png)

Crashing into a zombie (Carmageddon anyone?) gives a reward of -100 points, taking an action is -1 points and eating the ice cream gives you the crazy 1000 points. Why -1 points for taking an action? Well, the store might close anytime now, so you have to get there as soon as possible.

Congrats, you just created your first MDP. But how do we solve the problem? Stay tuned for that :)

Oops, almost forgot, your reward for reading so far:

<div class="center">
    <iframe src="https://www.youtube.com/embed/HjoEN6Ocfgs" frameborder="0" allowfullscreen></iframe>
</div>

# Want to learn more?

- [Dissecting Reinforcement Learning](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) <br/>
- [Reinforcement Learning: An Introduction 2nd edition draft](http://incompleteideas.net/book/bookdraft2017nov5.pdf)

---
layout: post
title:  "About Time - Part III"
date:   2016-12-20 20:08:00 +0300
categories: ["projects"]
excerpt: "Enough chit-chat. Let's define a model that can schedule your day!"
---

{:.center}
![png]({{site.url}}/assets/10.about_time_part_iii_files/hero.jpg)
*How adversarial training is NOT done! <br/> Image courtesy of [Ociacia](http://ociacia.deviantart.com/)*

Let's try to formalize our finding a bit. Can we make it all work?

Can we generate a schedule (rank a set of tasks) so that it maximizes the chances of achieving a set of predefined goals?

We have three types of tasks:

- completed
- scheduled - tasks that have defined start date and time
- unscheduled - tasks that might have an undefined start date, time or both

# Data definition

A task has the following properties:

- predicted duration
- category
- start time
- is it complete
- at what time was completed (available if it is complete)
- actual duration (available if it is complete)
- priority - number in the interval $$[1, 4]$$. 1 defines the minimum priority, 4 - the maximum. Can be undefined.

# How to make your schedule

Essentially, we are going to use [Adversarial Training](https://sites.google.com/site/nips2016adversarial/) to oppose the opinions of our minions.

- (1) Put scheduled and completed tasks at their appropriate time slots
- (2) Generate a schedule
- (3) If the obtained schedule is predicted to be completed above a certain threshold - done, else - go to 2.

# How to find the probability for each time slot $$s$$

- (1) Define prior discrete distribution using historical data for the interval $$[t_1, t_2]$$
- (2) Compute discrete likelihood distribution for task $$t$$ by creating a joint discrete distribution $$P(C, T)$$ where $$C$$ is using the task's category and start times
- (3) Find the posterior distribution using Bayes Rule and normalize
- (4) Compute the slot probability by adding all time units that are covered by $$s$$

# How to compute the priority for a task $$t$$

The importance $$I$$ of a task is defined as $$I = r ^ 2 * d$$ where $$r$$ is a priority as defined by the user and $$d$$ is the predicted duration of the task.
Note that $$r$$ can also be inherited from the priority of the higher level goal, if $$t$$ is associated with such.

# How to tell if a schedule is feasible

Here is the part of the pessimist. He tries to ruin our beautifully crafted schedule. Since how much of the schedule will be completed is highly user-specific, we're going to train a model that predicts how much of a given schedule will be completed using user's historical data. Furthermore, it would be useful to have uncertainty associated with each prediction. We are not required to provide a schedule if we're highly uncertain about its completion. Thus, we might not produce predictions during initial learning period for a user.

Considering we want our schedule to become more or less challenging every week, the model should obtain its posterior distribution conditioning on the performance from the previous week. Consequently, we should define some value by which to increase or decrease the initial threshold.

# How to generate a schedule

- order the tasks by their priority
- repeat until close to the threshold
    - Pick the first task from the queue
    - Assign probabilities for each slot during the predefined interval
    - Remove the slots that are occupied
    - Sample from the remaining distribution to rank/sort the slots
    - Assign the slot time for the current task and remove it from the task queue

# TBD: actual implementation of the algorithm above

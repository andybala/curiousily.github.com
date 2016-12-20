---
layout: post
title:  "About Time - Part III"
date:   2016-12-20 20:08:00 +0300
categories: ["projects"]
excerpt: "Enough chit-chat. Let's define a model that can schedule your day!"
---

Let's try to formalize our finding a bit. Can we make it all work?

# Data definition

A task has the following properties:

- predicted duration
- category
- start time
- is it complete
- at what time was completed (available if it is complete)
- actual duration (available if it is complete)

# How to make your schedule

- (1) Put scheduled tasks at their appropriate time slots
- (2) Generate a schedule
- (3) If the obtained schedule is predicted to be completed above a certain threshold - done, else - go to 2.

# How to find the probability for each time slot $$s$$

- (1) Define prior discrete distribution using historical data for the interval $$i_{t_1}$$ to $$i_{t_2}$$ 
- (2) Compute discrete likelihood distribution for task $$t$$ by creating a joint discrete distribution $$P(C, T)$$ where $$C$$ is using the task's category and start times
- (3) Find the posterior distribution using Bayes Rule and normalize
- (4) Compute the slot probability by adding all time units that are covered by $$s$$

# How to compute the priority for a task $$t$$

The importance $$I$$ of a task is defined as $$I = r ^ 2 * d$$ where $$r$$ is a priority as defined by the user and $$d$$ is the predicted duration of the task.
Note that $$r$$ is inherited from the priority of the higher level goal, if $$t$$ is associated with such.

# How to tell if a schedule is feasible

Here is the part of the pessimist. He tries to ruin our beautifully crafted schedule. Since how much of the schedule will be completed is highly user-specific, we're going to train a model that predicts how much of a given schedule will be completed using user's historical data. Furthermore, it would be useful to have uncertainty associated with each prediction. We are not required to provide a schedule if we're highly uncertain about its completion. Thus, we might not produce predictions during initial learning period for a user.

Considering we want our schedule to become more or less challenging every week, our model should condition its posterior distribution the performance from the previous week. Consequently, we should define some value by which to increase or decrease the initial threshold.

# How to generate a schedule

- order the tasks by their priority
- repeat until close to the threshold
    - Pick the first task from the queue
    - Assign probabilities for each slot during the predefined interval
    - Remove the slots that are occupied
    - Sample from the remaining distribution to rank/sort the slots
    - Assign the slot time for the current task and remove it from the task queue

# TBD: actual implementation of the algorithm above

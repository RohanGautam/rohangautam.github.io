---
title: MCMC - Adventures with the metropolis algorithm and bayesian linear regression
date: 2025-02-03
description: ""
tags:
  - mcmc
  - statistics
draft: true
---

- intro and motivation. Along with application examples. Why learn to implement it then use advanced tools
- what is mcmc, metropolis, etc, simply put?
- the math behind it
- code
- what's a good chain
- mcmc variants and developments.

I've recently grown more interested in bayesian statistics and inference, due to the direct use of [it in my work](https://www.sciencedirect.com/science/article/pii/S0169555X25000194?via%3Dihub). I've been slowly but surely working through Richard McElreath's fantastic book on bayesian statistics, [Statistical Rethinking](https://xcelab.net/rm/), and that inspired me to write about mcmc, namely the simplest mcmc algorithm - the Metropolis algorithm, along with some applied examples in python. The algorithm does feel like magic, and that's exactly why we need to peel the magical sounding layers away.

Let me first set up an example problem we'll be considering, vaguely. You're a scientist, and you have collected data on some observations from the world. Chances are you're studying the _phenomena_ which caused in the data you observed. You come up with a hypothesis, or mathematical formula, of how the phenomena works. This of course, requires you to make assumptions of the inner workings of the phenomenon, and these have variables which are hidden from you but affect the observed outcome. Simplistic as your model for the phenomenon may be, it's a start, and what you have is a simple _generative model_ for your data. Your model might not perfectly estimate the observation - there will be some noise, due to other factors that you dont consider.

Let's call the hidden variables the 'parameters', the generative model of your data the 'model'. What we want to do, is see what parameters best explain the observations you see. Since we're taking a bayesian approach (link: why bayesian?), we're not after the parameter that ensures the best fit (the point estimate). We're looking for the range of plausible values the parameters can take given the observation - their probability distribution.

This is where MCMC comes in. It helps us find these parameter probabilities (known as the posterior distribution) by giving us _samples_ from the parameter's probability distributions. From these samples, we can estimate the likely values of each parameter and understand their relationships with each other.

MCMC stands for "Markov Chain Monte Carlo" - let's break this down. The "Markov Chain"(link) part means the algorithm works like a trail of breadcrumbs - each step only depends on where we just were, not the entire path we took to get there. In our case, when we're searching for good parameter values, each new guess only depends on our current guess. The "Monte Carlo" part refers to how we use randomness to explore - like a gambler trying different strategies and seeing what sticks, we use random sampling to explore the possible parameter values and find the ones that work best.

## The math behind it

# References

https://blog.djnavarro.net/posts/2023-04-12_metropolis-hastings/#fnref1

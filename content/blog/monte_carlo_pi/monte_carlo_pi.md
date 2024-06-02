---
title: Estimating π by throwing needles
date: 2024-06-02
description: "Discover the fascinating Buffon's Needle experiment, a powerful Monte Carlo method to approximate π!"
tags:
  - simulation
  - montecarlo
---

## Buffon's needle 
> Discussion on [X](https://x.com/rohang_yall/status/1792705027217473707)

The  Buffon's Needle experiment is a cool Monte Carlo method to approximate π! 🪡

{% image "./ezgif-7-71edd53516.gif", "gif of simulation of needles and pi value being highlighted"%}


When you throw a needle of length $L$ onto a piece of ruled paper, it intersects the lines if $x <= L\sin(\theta)$, where 𝑥 is the distance from the bottom of the needle to the closest line above it.

{% image "./image.png", "Positions when a needle is thrown onto a piece of paper"%}

Assuming that 𝑥 and 𝜃 are uniformly distributed, for an intersection, we require $0<=\theta<=\pi$ and $0<=x<=L\sin(\theta)$. The probability of this is given by the integral:

$$
P(\text{intersection}) = \int_{0}^{\pi}\int_{0}^{L\sin\theta}\frac{1}{D\pi}dxd\theta = \frac{2L}{D\pi}
$$


We can estimate the probability of intersection via repeated random trials (needles intersecting line / needles thrown) and rearrange the equation to get a Monte Carlo approximation of π:

$$
\pi=\frac{2L}{D*P\text({intersection})}
$$

I've made a colab notebook where you can mess around with this simulation in an interactive way. Play with it here! => [Interactive simulation](https://colab.research.google.com/drive/1AdmkqeuU-1WQDG88-EB77Nlv1eL3wStO?usp=sharing)

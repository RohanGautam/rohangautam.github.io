---
title: (Road to KAN part 1) A gentle introduction to B-splines
date: 2024-06-18
description: "Let's talk about B-splines, a polular approximation and interpolation curve, and the workhorse of the Kolmogorov-Arnold network"
tags:
  - machine learning
  - splines
  - KAN
---

In this blog series, I'll be going through the technical details involved in understanding [Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2404.19756) - a new type of machine learning architecture which has gotten significant attention lately due to it's several advantages over MLPs - interpretability and avoiding catastrophic forgetting, to name a few.

In part one of this series, let's build our understanding up to B-splines, the workhorse of the Kolmogorov-Arnold Network (KAN). B splines are used in KANs to learn activation functions. Let's start with simple polynomial interpolation.

[[toc]]

# Polynomial interpolation

The goal of polynomial interpolation is to find the coefficients of a polynomial $a_0 +  a_1x + a_2x^2 + ... a_nx^n$ , such that it passes through your data points.
For example, if we had 4 data points $[(x_1,y_1), (x_2,y_2),(x_3,y_3),(x_4,y_4)]$, we could fit a cubic polynomial $a_0 +  a_1x + a_2x^2 + a_3x^3$ through it by solving for $a_i$ in the following system of equations:

$$
\begin{pmatrix}
1 & x_1 & x_1^2 & x_1^3 \\
1 & x_2 & x_2^2 & x_2^3 \\
1 & x_3 & x_3^2 & x_3^3 \\
1 & x_4 & x_4^2 & x_4^3
\end{pmatrix}
\begin{pmatrix}
a_0 \\
a_1 \\
a_2 \\
a_3
\end{pmatrix}
=
\begin{pmatrix}
y_1 \\
y_2 \\
y_3 \\
y_4
\end{pmatrix}
$$

Here's an example of the fit with some toy data:
{% image "Pasted image 20240616150014.png","-" %}
You'd need a polynomial of degree $n-1$, to pass through $n$ data points. A higher polynomial degree $n-1$ would imply non-unique solutions, and a lower polynomial degree would imply non-existence of a solution (unless in special cases). Let's see what a higher order polynomial fit looks like:
{% image "Pasted image 20240616151313.png","-" %}

This interpolation looks reasonable in the middle, but what's going on at the edges? Those jumps look a bit too extreme. Well, we've encountered the well-known [Runge's phenomenon](https://en.wikipedia.org/wiki/Runge%27s_phenomenon), where higher degree polynomials oscillate at the edges. This leads us to one of the many motivations behind using splines for interpolation.

# Splines

We can think of splines as combining many low-order (often cubic) polynomials in order to create a more complex, smooth curve. Splines have many varieties, mostly involving different ways these individual polynomial pieces can be joined, such that they smoothly flow from one to the other, enforcing certain _continuity_ measures. We'll take a look at continuity soon. But first, let's see how these can be better than polynomial interpolation:

| Feature                | Polynomial Interpolation                                           | Spline Interpolation                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Degree**             | Single polynomial of degree $n-1$ for $n$ points                   | Lower-degree polynomials (typically cubic) for each interval                                                                                                                      |
| **Continuity**         | Ensures smoothness over entire range                               | Ensures smoothness at the knots (joining points)                                                                                                                                  |
| **Local Control**      | No local control, changing one point affects the entire polynomial | Local control, changing one point affects only the nearby intervals. **This is a desirable feature for interpolation tasks**. B splines, among other splines, have this property. |
| **Runge's Phenomenon** | Susceptible to Runge's phenomenon (oscillations at the edge)       | Avoids Runge's phenomenon                                                                                                                                                         |
| **Efficiency**         | Computationally intensive for large datasets                       | More efficient for large datasets due to local computation                                                                                                                        |
| **Flexibility**        | Less flexible for complex shapes                                   | More flexible                                                                                                                                                                     |

{% image "Pasted image 20240618140850.png","-" %}
// image of a B spline, with piecewise components highlighted. Highlight the knots (where polynomials join) and the control points (which control the spline, and do not necessarily lie on the curve.)

## Continuity

So splines are connected at points called _knots_ or joins, and their overall shape is determined by user-defined _control points_. Continuity (or "smoothness") is a measure of how smoothly one polynomial connects to the other. When thinking of smoothness, one can naturally think of having a continuous rate of change, and this is indeed a form of continuity, called $C^1$ continuity, where the first derivative is continuous at the joins. There can be some physical intuition here, because if you imagine a tangent vector moving along the spline, $C^1$ continuity implies it moving at a continuous speed (the first derivative of distance with respect to time). Notably, there is no abrupt change of speed **at the join** in particular, because before the join, we're on a smooth polynomial anyway. $C^2$ continuity would imply no sudden change of acceleration at the join, and so on. The lovely video by Freya Holmér on [the continuity of splines](https://www.youtube.com/watch?v=jvPPXbo87ds) beautifully explains and animates continuity measures, including other ones, like geometric continuity which we won't get into here.

In general, $C^n$ continuity means that the $n^{th}$ derivative is continuous at the joins, and implies that the lower ( $(n-1 ... 0)^{th}$) derivatives also exist and are continuous. A higher $n$ gives us a smoother curve.
{% image "Pasted image 20240618150758.png","-" %}
// image of continuity order 0,1,2

## Basis functions

I want to touch on the intuition behind basis functions before we jump into B-splines. Like mentioned before, splines have a key property of _local control_, wherein moving a control point only effects a fixed region near itself instead of affecting the entire curve. This "influence" that a control point has is determined by it's basis function, and there are as many basis functions as there are control points. A widely spread out basis function for a specific control point would imply that moving the control point can affect a larger part of the curve. The animation below represents this quite well.
{% image "b_spline_influence.gif","-" %}
// an animation highlighting the basis function on one control points neighbourhood on the curve.

# B splines

B-splines are cool, as they are $C^2$ continuous (read: it's quite smooth) and have local control. Also, they are spline's that don't necessarily pass through their control points! In other words, they can be made to be both approximating splines(don't pass through all control points), or interpolating splines(pass through all control points, like with [scipy.interpolate.splrep](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html)). In this article, we'll focus on approximating splines, as our end goal involves making use of _approximating_ B-splines to learn activation functions in a KAN. generates _interpolating_ B splines.

In cubic B-splines, each piecewise polynomial is expressed as

$$
\mathbf{P}(t) =
\begin{bmatrix}
1 & t & t^2 & t^3
\end{bmatrix}
\frac{1}{6}
\begin{bmatrix}
1 & 4 & 1 & 0 \\
-3 & 0 & 3 & 0 \\
3 & -6 & 3 & 0 \\
-1 & 3 & -3 & 1
\end{bmatrix}
\begin{bmatrix}
\color{red}{\mathbf{P}_a} \\
\color{lightblue}{\mathbf{P}_b} \\
\color{green}{\mathbf{P}_c} \\
\color{yellow}{\mathbf{P}_d}
\end{bmatrix}
$$

where $\color{red}{\mathbf{P}_a}, \color{lightblue}{\mathbf{P}_b}, \color{green}{\mathbf{P}_c}, \color{yellow}{\mathbf{P}_d}$ represent the 4 control points that are needed to define a cubic polynomial. Each of the 4 consecutive points from a given set of control points form the polynomial pieces stitched together. Note that B splines represent parameterised curves, so $t$ goes between $0$ and $1$, giving us our final curve $P(t)$ as a vector, equating to $[x(t), y(t)]^T$.

Expanding, we get our basis functions, which determine the influence that each of the 4 control points have on any given point on our spline. The matrix of numbers in the equation from which we expand might look arbitrary, but these values can actually be solved for provided our constraints for the B spline, namely $C^2$ continuity and the requirement that the basis functions(influences) sum up to one (to avoid arbitrary scaling effects).

$$
\mathbf{P}(t) = \frac{1}{6} \left[ (1 - 3t + 3t^2 - t^3)\color{red}{\mathbf{P}_a} + (4 - 6t^2 + 3t^3)\color{lightblue}{\mathbf{P}_b} + (1 + 3t + 3t^2 - 3t^3)\color{green}{\mathbf{P}_c} + t^3\color{yellow}{\mathbf{P}_d} \right]
$$

Plotting the functions influencing each control point, we can see the cubic B spline basis functions look like this ([graph on desmos](https://www.desmos.com/calculator/ubtrhyjn0m)):
{% image "Pasted image 20240617111409.png","-" %}
These represent the influences of the 4 control points throughout one cubic piece of a spline. Alternatively, you can see the influence of the control points on the _whole_ spline in the animated figure in the basis functions section.

// todo: reconstruct this in python, another view where this is rearranged to see the influence of one control point. maybe highlighting what view you're considering would be nice? like an eye from a control point vs a point on the spline.
We can see from the alternative view of basis functions that one control point exerts maximum influence close to itself, and this influence diminishes away from it. In the first view, we see that at any given point _on the spline_, the different influences acting on it. ==does this belong in the basis function section?==

## Cox–de Boor recursion formula

The Cox-de Boor formula tells us how to calculate the basis functions of a B spline, allowing us to calculate the influence a control point will have on different parts of the final curve. It does this though an intermediate abstraction called 'knots', which allow us to specify _where and how_ the control points influence the curve. We'll look at a concrete example called _knot clamping_ in a bit.
Let $N_{\text{knot index},\text{degree}}$ be the basis function. As a reminder, our spline will be some linear combination of these:

$$
S_{n,t}(x) = \sum_i \alpha_i N_{i,k}(x)
$$

The Cox-de Boor algorithm is recursive, with a base case and a recursive step.The base case is for $k=0$

$$
N_{i,0}(u) = \begin{cases} 1 & \text{if } t_i \leq u < t_{i+1} \\ 0 & \text{otherwise} \end{cases}
$$

This is telling us that the base case is just a step function, which is 1 in the knot interval $[t_i,t_{i+1})$ and zero elsewhere.
The recursive step is:

$$
N_{i,\color{red}{k}}(u) = \frac{u - t_i}{t_{i+k} - t_i} N_{i,\color{red}{k-1}}(u) + \frac{t_{i+k+1} - u}{t_{i+k+1} - t_{i+1}} N_{i+1,\color{red}{k-1}}(u)
$$

The basis function for higher degrees can be expressed as a weighted combination of two lower-degree basis functions, which I've highlighted in red. Their scaling factors ensure a smooth transition from one to the other.
Things can be clearer and more concrete in code. Here's an (inefficient) implementation!

```python
def cox_de_boor(u, i, k, knots):
    """
    u : x value of the point to be evaluated in the input domain
    i : index of the basis function to compute
    k : degree of the spline
    knots : values in the input(x) domain that divide the spline into pieces

    returns -> a scalar value that calculates the influence of the i'th basis function on the point u in the input domain.
    """
    if k == 0:
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0
    left_term = 0.0
    right_term = 0.0
    if knots[i + k] != knots[i]:
        left_term = (
            (u - knots[i]) / (knots[i + k] - knots[i]) * cox_de_boor(u, i, k - 1, knots)
        )
    if knots[i + k + 1] != knots[i + 1]:
        right_term = (
            (knots[i + k + 1] - u)
            / (knots[i + k + 1] - knots[i + 1])
            * cox_de_boor(u, i + 1, k - 1, knots)
        )
    return left_term + right_term
```

## Fitting a B-spline with code

Let's fit a B spline to some data with this. The snippet below also explains knot clamping, a trick used to make an approximating B spline pass through the start and end points, if needed.

```python
import numpy as np
import matplotlib.pyplot as plt

control_points = np.array([[0, 0], [1, 2], [2, 0], [3, 2], [4, 0], [5, 0], [6, 1]])
## repeat the last knot value degree+1 times. degree+1 points are needed to define a degree-ordered polynomial, so if all are the same,
## the curve WILL pass through that point, as that point will be responsible for all the influence on the curve at that region.
##  This is known as clamping
knots = np.array([0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4])
degree = 3
# we evaluate the final spline only at the un-clamped (non repeated) points.
u_values = np.linspace(knots[degree], knots[-degree - 1], 100)
num_basis_functions = len(knots) - degree - 1

# Calculate the basis functions
basis_functions = np.zeros((len(u_values), num_basis_functions))
for i in range(num_basis_functions):
    basis_functions[:, i] = [cox_de_boor(u, i, degree, knots) for u in u_values]
basis_functions[-1, -1] = 1

# Construct the B-spline curve. It's a linear combination of the basis functions, weighted by the control points!
curve = np.zeros((len(u_values), 2))
for i in range(num_basis_functions):
    curve += basis_functions[:, i].reshape(-1, 1) * control_points[i]

## plotting basis functions
for i in range(num_basis_functions):
    plt.plot(u_values, basis_functions[:, i], label=f"B{i}")
plt.show()

## plotting the spline
plt.plot(curve[:, 0], curve[:, 1], label="B-Spline Curve")
plt.plot(control_points[:, 0], control_points[:, 1], "ro--", label="Control Points")
plt.show()

```

Because of clamping, note that the first and last functions are 1 at the extremes and all other influences are 0, implying the start and end points have full influence on the curve, making the curve have to pass through it.
{% image "Pasted image 20240618110327.png","-" %}
And here's our B-spline!
{% image "Pasted image 20240618110405.png","-" %}

# Closing

Hope you enjoyed this article about B splines! If you have any feedback, leave them with the blog title on [Issues · RohanGautam/rohangautam.github.io](https://github.com/RohanGautam/rohangautam.github.io/issues). We started from polynomial interpolation, talked about splines, and took a look at the construction of approximating B splines in this post. I plan on following this up with more on how they are used in KANs, touching on grid extension, stacking splines and more. Follow me on [Twitter/X](https://x.com/rohang_yall) to hear about it when it's done!

# References

1. [pg2455/KAN-Tutorial: Understanding Kolmogorov-Arnold Networks: A Tutorial Series on KAN using Toy Examples (github.com)](https://github.com/pg2455/KAN-Tutorial)
2. [Splines in 5 minutes: Part 1 -- cubic curves (youtube.com)](https://www.youtube.com/watch?v=YMl25iCCRew&t=5s) a three part series
3. [The Continuity of Splines (youtube.com)](https://www.youtube.com/watch?v=jvPPXbo87ds&t=62s)
4. [B-spline - Wikipedia](https://en.wikipedia.org/wiki/B-spline#Definition)

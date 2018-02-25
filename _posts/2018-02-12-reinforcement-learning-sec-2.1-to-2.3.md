---
layout: post
title: Reinforcement Learning (Sections 2.1-2.3)
mathjax: true
---

This marks the first post in a series devoted to my exploration of *Reinforcement Learning: An Introduction, 2nd Edition* by Sutton & Barto (2017), located [here](http://incompleteideas.net/book/the-book-2nd.html).
I'm going to be following the excellent course website for CMPUT 609 on Richard Sutton's website.
There's also awesome code samples for each chapter [here](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction), though I'll probably write some of my own up too.

Chapter 2 is all about multi-armed bandits. Onto the sections!

## Section 2.1: A $k$-armed bandit problem
A multi-armed bandit problem is one in which we are repeatedly presented with $k$ different actions $a_i$, each of which give a random reward after being chosen.
- The rewards are distributed according to some stationary probability distribution. For example, there might be two actions, $a_1 \sim N(0,2)$ and $a_2 \sim N(1,1)$.
- Our objective is to, over time, maximize the cumulative reward we gain from our actions.
- These problems play out on a discrete timeline. Following the book, we'll denote the action at time $t$ as $A_t$ and the reward at time $t$ as $R_t$.
- We call the true reward of an action $q_*(a) = E[R_t \mid A_t=a]$, and our estimate at time $t$ is $Q_t(a)$.

## Section 2.2: Action-value methods
A simple way to estimate $q_*(a)$ is to let $Q_t(a)$ be the mean reward we've received when we've taken action $a$. We'll initialize $Q_1(a)$ to zero for all actions.

Here's two methods to select actions for this bandit problem:
1. The greedy method! At each iteration, let's just that lets $A_t = argmax_a Q_t(a)$
2. An $\epsilon$-greedy method is a strategy that, at each iteration, chooses with probability $1 - \epsilon$ between (a) the action with the highest estimated reward, and (b) one of the actions with a smaller estimated reward. The point of using an $\epsilon$-greedy method instead of *just* a greedy method is to explore other actions we haven't seen yet that might have better mean rewards.

In most of these notes, I'll assume that we use what I'm calling a uniform $\epsilon$-greedy method. This method uses the following conventions:
- If we roll an $\epsilon$ or below on our random number generator, we choose uniformly randomly from among the suboptimal strategies.
- If there's multiple actions with the same best estimate, we choose uniformly randomly among them.

#### Exercise 2.1
Under $\epsilon$-greedy action selection, in the case of two actions and $\epsilon=0.5$, the lesser action is selected with probability $0.5$.

#### Exercise 2.4
Using the uniform eps-greedy method on a problem with 4 actions, we get the following feedback at each round:
1. a=1, r=1
2. a=2, r=1
3. a=2, r=2
4. a=2, r=2
5. a=3, r=0

And here are our
At round 1, we took action 1 and got a rewards of one. Therefore, we have the following estimates of the actions' rewards (in order from 1 to 4): $ 1, 0, 0, 0 $. Thus $a_1$ is the best action.

However, at round 2, we choose action 2, despite the fact that the best action is action 1! Therefore, we must have rolled an $\epsilon$. Our new estimates: $ 1, 1, 0, 0 $.

On round 3, we choose round 2. We mustn't've rolled an $\epsilon$, as $a_2$ is one of the best actions. New estimates: $ 1, 1.5, 0, 0 $. Notably, $a_2$ is now the best and only best action.

Round 4 sees another $a_2$! We rolled a $1-\epsilon$. Estimates: $ 1, 5/3, 0, 0 $.

On round 5, we must've rolled an eps again, as we chose action 3. Estimates: $ 1, 5/3, 0, 0 $.

## Section 2.3: The 10-armed Testbed
We get a testbed: the 10-armed bandit problem. Each $q_\*(a)$ is set once before the first round to a draw from $N(0,1)$; then at each round $R_t$ is drawn from $N(q_\*(a),1)$. Each bandit problem goes for 1000 iterations, and we perform 2000 of these runs.
We test against the greedy method, and two uniform eps-greedy methods with $\epsilon \in \\{ 0.1,0.01 \\}$.
The epsilon strategies beat the greedy one, hands down.

#### Exercise 2.3
So which strategy performs best in the long run? Here's the intuition: over a very long timeframe, the 0.01 strategy does better, because it selects the best action more often. The 0.1 strategy might identify the best action sooner, but at infinity, the 0.01 strategy has selected the best action far more often. It is possible to start with a higher epsilon and decrease it over time to get the best of both methods, but we'll think about that later.

Exactly how much better is the 0.01 strategy expected to do? Well, for the percent of time a method selects the best action, we've got 99% vs. 90%, which means the 0.01 strategy is about 10% better than the 0.1 strategy. As for the average reward, let's see...

Since the average reward is just within a linear factor (our timestep, n) of the cumulative (sum of) rewards, we'll calculate the expected value of the cumulative reward for each strategy assuming $t=\infty$, and take the difference. Since we're at infinity here, we can assume that the strategy has identified perfectly the mean rewards of each action (thanks to the Central Limit Theorem, our average estimates of rewards must have converged to their true values).

As we're drawing the rewards' means from a continuous distribution, there is zero probability that we choose the same mean reward for two different actions. Therefore we can assume that there's *one* best action. I'll denote the set of non-optimal actions as $A'$, and the optimal action as $a^\*$. Let's just calculate the expected cumulative reward ("ECR") for any $\epsilon$ to start with.

The ECR is the expected reward of the best action multiplied by the number of times we choose it, plus the expected reward of the suboptimal actions multiplied by the number of times we choose them.

$$
ECR_{\epsilon} = (1-\epsilon) \cdot E[R_t | A_t = \text{argmax}_a Q_t(a)] + \epsilon \cdot E[R_t | A_t \neq \text{argmax}_a Q_t(a)]
$$

Since these are uniform epsilon strategies, we have that the expected reward of the suboptimal actions is just their average:

$$
E[R_t | A_t \neq \text{argmax}_a Q_t(a)] := E[R_t | A_t \in A'] = \sum_{a \in A'} q_*(a) / 9.
$$

Each of these $q_\*(a)$ is a draw from a standard normal distribution, right? But wait! That isn't *quite* true. In fact, if you decided to believe this, you'd see that:

$$
\sum_{a \in A'} q_*(a) / 9 = \sum_{a \in A'} E[X \sim N(0,1)] / 9 = 0.
$$

Womp womp. We know that can't be true. And it isn't! The probability of selecting nine draws from $N(0,1)$ that add up to zero is pretty much zero. If we recall from statistics, each $q_\*(a)$ is actually an *order statistic*. If you've forgotten all about order statistics, you might enjoy the links in the references (specifically, refrences 1-5). What we need to do is to compute the average of the expectations of each of the first through ninth order statistics from a group of 10 standard normal variates (where "variates" means "draws from a standard normal distribution").

But that's going to be a little involved, so let's start with a more illustrative example: the best action's true mean reward. The best action is the maximum of 10 draws from a standard normal. This is a nice simple example of an order statistic--called the 10th order statistic, or $X_{(10)}$, if it's part of a group of 10 variates. Now, to get the expected value of any random variable, we need its distribution, right? So, what's the distribution of the max of ten stadard normal random variables ("RVs")? We'll denote these RVs $X_i$, $i=1,...,10$, and the max $Y$. Let's attack it from the CDF point of view. What's the probability that the max takes on some value $y$ or greater? Well, we're really asking if each $X_i$ is less than $y$ (thus making $y$ the max), so let's find the probability of this. By this reasoning, the CDF of $Y$ is, by iidness of the $X_i$s,

$$
P(Y < y) = P(X_1 < y, X_2 < y, ..., X_10 < y) = P(X < y)^n = \Phi(y)^n.
$$

where $\Phi(y)$ is the standard normal CDF. Now, to get the expected value, we need to take the integral over the product of $y$ and the pdf of $Y$, which is simply the derivative of the CDF:

$$
E[Y] = \int_{-\infty}^\infty y \frac{d}{dy} \Phi(y)^n dy.
$$

Numerically integrating shows us that $E[Y] \approx 1.5388$.

Now, we're back to the calculation of the other nine order statistics' expected values. These are, well, kind of a pain to calculate. I've realized that either (a) Sutton & Barto didn't expect people to be going this far, or (b) I'm going about this entirely the wrong way, and there's a simpler way to answer the question, so (forgive me!) I've decided to take the easy way out! This is something I really don't like to do, because there's a lot of fun stuff to learn here, but it's hard to wait while tantalizing subjects like gradient bandits lie around the corner (or page flip, if you prefer not mixing metaphors). So, from [1], we have that, for order statistics of $N(\mu, \sigma)$, we can approximate like so:

$$
E[X_{(r)/n}] \approx \mu + \Phi^{-1}(\frac{r-\alpha}{n-2\alpha+1})\sigma, \qquad \alpha=0.375
$$

where $r$ is the order of the statistic, and $n$ is the number of variates. Thanks to Chris Taylor (writer) and kjetil b halvorsen (editor) for that answer. The original formula is due to Blom (1958).

So, we now know that

$$
E[q_*(a) | a=a^*] = \frac{1}{9} \sum_{r=1}^9 E[X_{(r)/10}] \approx \frac{1}{9} \sum_{r=1}^9 \Phi^{-1}(\frac{r-\alpha}{10-2\alpha+1})
\approx -0.1718.
$$

I used the following Python code to calculate this, and you can too:
```python
from scipy.stats import norm
def blom(r, n):
    a = 0.375
    return (r-a)/(n-2*a+1)
exp_vals = 0
for i in range(1,10):
    exp_vals += norm.ppf(blom(i,10))
print(exp_vals / 9)
```

**Aside:** It's fascinating to note that we can actually derive not only the distribution of the maximum of a set of normal samples, but the distribution of the maximum of a set of random variables distributed according to *any* distribution (as long as the samples are iid, and the maximum converges--e.g. that means no infinite variances!). This is due to the Fisher–Tippett–Gnedenko theorem. So, theoretically, we could get the entire distribution even if the rewards were distributed according to an exponential distribution. Another interesting thing to note is that we can do more than just point estimates if we know the whole distribution. So this is interesting!

Putting this together with the earlier result, we have that

$$
ECR_{\epsilon_1} - ECR_{\epsilon_2} \approx (\epsilon_2-\epsilon_1) E[q_*(a) | a=a^*] + (\epsilon_1-\epsilon_2) E[q_*(a) | a \neq a^*]
$$

So,

$$
\begin{align}
ECR_{0.01} - ECR_{0.1} & \approx (0.1-0.01)(-0.1718) + (0.01 - 0.1)(1.5388) \\
& \approx -0.09(-0.1718) + 0.09(1.5388) = 0.15395
\end{align}
$$

As

$$
ECR_{0.1} \approx (0.1)(-0.1718) + (0.9)1.5388 \approx 1.3677
$$

we see that the 0.01 strategy does about 11.3% better in terms of cumulative expected reward. Nice!

Well, that's all for this post. Come back soon for future sections of Chapter 2!

## References
1. https://stats.stackexchange.com/questions/9001/approximate-order-statistics-for-normal-random-variables
2. https://stats.stackexchange.com/questions/83307/expected-lowest-value-of-10-normally-distributed-values?noredirect=1&lq=1
3. https://stats.stackexchange.com/questions/141376/finding-expected-order-statistics-from-a-normal-with-known-parameters?noredirect=1&lq=1
4. https://math.stackexchange.com/questions/473229/expected-value-for-maximum-of-n-normal-random-variable
5. https://stats.stackexchange.com/questions/18433/how-do-you-calculate-the-probability-density-function-of-the-maximum-of-a-sample

This project was created aiming to implement the exercise 2.5 described in the book: Reinforcement Learning: An introduction, written by Richard S. Sutton and Andrew G. Barto. 

The exercise as depicted in the book is transcribed as follow:

Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for non-stationary problems.
Use a modified version of the 10-armed testbed in which all the $q_*(a)$ start out equal and then take independent random walks 
(say by adding a normally distributed increment with mean zero and standard deviation $0.01$ to all the $q_*(a)$ on each step).
Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed,
and another action-value method using a constant step-size parameter, $\Alpha = 0.1$. Use $\Epsilon = 0.1$ and longer runs, say of $10,000$ steps.

This exercise is implemented in the file main.py where the original 10-armed testbed was recreated and modified to attend to the new requests.

- To install required dependencies run: `pip install -r lib.txt `
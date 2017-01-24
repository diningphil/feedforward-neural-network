# Feedforward Neural Network
An implementation of a Neural Network simulator based on the theory of the book Rumelhart, McClelland et al 1986 - Parallel Distributed Processing. 
### Authors: diningphil and korialis

## What to expect
The implementation is vectorized (a non-vectorized version exists in Octave folder, and strictly follows the results in the book).
Notice that this implementation is based on the batch method of learning.
The list of things to do is organized in this way:

Repository issues:

- Produce a good documentation

Technical issues:

- Training selects the "best" over a number of runs; best can be less error or max accuracy. So it needs to be able to discriminate, on the basis of the performance measure.


- Try to avoid saturation of the units limiting the response of a neuron; it should be a bit away from the max or min value of the activation function (its gradient becomes 0 and the weights tend to remain unchanged during next iterations)

## Not strictly necessary

- Implement more Cross Validation techniques

- The learning rate should be smaller in the last layers and higher in the first layers (the gradient decreases backwards). This way, the entire network should learn at the same speed

- Make a stochastic(online) version

- Shuffle the TR examples at each iteration (only for online version)

- Add other stopping criteria heuristics (in a configurable manner). It may depend on the problem taken into consideration

## What has been done up to now

- Implemented a for-loop version, with only 1 hidden layer. This may turn out to be helpful for those who wants to see how theory can be transferred into Octave code in the simplest way.

- Implemented a vectorized version =)

- Added random initialization of the weights in a range.

- Added momentum.

- Added regularization to the code.

- Now an arbitrary number of hidden layers can be set, each layer having the desired amount of units.

- Learning-rate annealing: eta(t) = eta(0)/(1 + iterations/#training_examples);

- Implemented Hold Out CV with grid search (the grid must be hardcoded in the file hold_out.m)

- Now you can try a number of random starting configuration (10 or more tr runs or trials) and take the "best" one (where best means less error on training --> hoping to find a global minimum)

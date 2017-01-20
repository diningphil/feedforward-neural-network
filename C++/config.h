#include <vector>

using namespace std;

#ifndef _CONFIG
#define _CONFIG

// Max number of iterations
static const int maxIter = 1000;

// Error threshold
static const float threshold = 0.1; 

// Weights initialization range
static const float init_range = 0.5;

// sigmoid constant
static const float a = 1;

// learning rate a bit higher since now is scaled by m and annealing too
// (with larger nets a much smaller eta is sufficient it seems)
static const float eta = 0.5;

// momentum constant
static const float alpha = 0.5; 

// regularization constant
static const float lambda = 0.00; 

// hidden_units, each value stands for an hidden layer's number of units
// (so its dimension is the number of hidden layers)
//layers_dims = [500 30 10]; 
static const vector<int> layers_dims = { 2 };

// Number of runs
static const int runs = 10;

#endif

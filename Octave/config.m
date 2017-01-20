% Max number of iterations
maxIter = 15000;

% Error threshold
threshold = 0.1; 

% Weights initialization range
## WE ARE USING FANIN, we don't care
useFanIn = true;
init_range = 0.2;

% learning rate a bit higher since now is scaled by m and annealing too
% (with larger nets a much smaller eta is sufficient it seems)
eta = 0.05;
useAnnealing = false;

% momentum constant
alpha = 0.1;

% regularization constant
lambda = 0.000;
 
% hidden_units, each value stands for an hidden layer's number of units
% (so its dimension is the number of hidden layers)
%layers_dims = [500 30 10]; 
layers_dims = [2]; # Per il fan in è meglio più unità e meno layer, e il grad si propaga meglio

lossType = "lms";
outputActivation = "linear";
hiddenActivation = "sigmoid";

% Number of runs
runs = 5;

usePlot = true;


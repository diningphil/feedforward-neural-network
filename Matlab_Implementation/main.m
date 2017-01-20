%
clear ; close all; clc
addpath(genpath('./Training'));
addpath(genpath('./ModelSelection_Assessment'));
addpath(genpath('./Estimates'));
addpath(genpath('./Activations'));
addpath(genpath('./Utils'));
addpath(genpath('./logging'));

config;
logConfig;

%######## MONK LOADING PHASE ########
%TR = load("../Monk_Dataset/monks2prep.train");
%TS = load("../Monk_Dataset/monks2prep.test");
%X = TR(:,2:end);
%Xtest = TS(:,2:end);
%Y = TR(:,1);
%Ytest = TS(:,1);  
%####################################

%######## CUP 2016 LOADING PHASE ####
DATA = csvread('../2016CUP/cupTRshuffled.csv');
%## DATA HAS BEEN SHUFFLED
X = DATA(:,2:11);
Y = DATA(:,12:13);
%#####################################

% Number of training examples
m = size(X, 1); 

% Total error on training for each iteration
TR_Err = 1:maxIter;

estimateMeasure = @(O,Y)( mee(O,Y) );

Xtrain = X(1:floor(70*m/100), :);
Ytrain = Y(1:floor(70*m/100), :);
Xval = X(floor(70*m/100 + 1):floor(85*m/100), :);
Yval = Y(floor(70*m/100 + 1):floor(85*m/100), :);
Xtest = X(floor(85*m/100 + 1):end, :);
Ytest = Y(floor(85*m/100 + 1):end, :);

mtrain = size(Xtrain,1);
[newX, newY] = shuffle([Xtrain; Xval], [Ytrain; Yval]);
Xtrain = newX(1:mtrain, :);
Ytrain = newY(1:mtrain, :);
Xval = newX(mtrain+1:end, :);
Yval = newY(mtrain+1:end, :);

%#### SIMPLE TRAINING (NO MODEL SELECTION METHODS) ####
%logDir = startLogging(SIMPLE_T);
%[TR_Err, TR_Acc, nIter, Wres, W_O_res, output_hidden_units] = train(X, Y, eta, lambda, alpha, outputActivation, hiddenActivation, lossType, estimateMeasure, threshold, init_range, maxIter, layers_dims, runs, useAnnealing, useFanIn, usePlot);
%myPlot(TR_Err, TR_Acc, nIter, lossType, alpha, eta, lambda, layers_dims, size(Y,2), logDir, testName);
%stopLogging();
%######################################################

%#### SIMPLE TRAINING (NO MODEL SELECTION METHODS) WITH 'TEST' COMPARING ####
logDir = startLogging(SIMPLE_T);
[TR_Err, TR_Acc, nIter, Wres, W_O_res, output_hidden_units, TS_a, TS_b] = trainWithValidation(Xtrain, Ytrain, Xval, Yval, eta, lambda, alpha, outputActivation, hiddenActivation, lossType, estimateMeasure, threshold, init_range, maxIter, layers_dims, runs, useAnnealing, useFanIn, usePlot);
logPlotWithTest(TR_Err, TR_Acc, nIter, TS_a, TS_b, lossType, alpha, eta, lambda, layers_dims, size(Y,2), logDir);
stopLogging();
%############################################################################

% learning rate
eta_range = [0.09];%[0.035 0.02];
% momentum constant
alpha_range = [0.015 0.05  0.01];
% regularization constant
lambda_range = [0.00015 0.0005 0.0007];


%#### HOLD OUT MODEL SELECTION ####
%[estimation, accuracy, W, W_O, eta_val, alpha_val, lambda_val] = hold_out(X, Y, 80, 10, eta_range, alpha_range, lambda_range, outputActivation, hiddenActivation, lossType, estimateMeasure, threshold, init_range, maxIter, layers_dims, runs, useAnnealing, useFanIn, true);
%fprintf("Chosen parameters: eta = %f, alpha = %f and lambda = %f. Assessment value on TS:\n Cost = %f \t Accuracy:%f\n", eta_val, alpha_val, lambda_val, estimation, accuracy);
%##################################

%#### CROSS VALIDATION + GRID SEARCH MODEL SELECTION ####
% Cross validation for model selection

%[W, W_O, eta_val, alpha_val, lambda_val] = modelSelection_CV(4, X, Y, eta_range, alpha_range, lambda_range, outputActivation, hiddenActivation, lossType, estimateMeasure, threshold, init_range, maxIter, layers_dims, runs, useAnnealing, useFanIn, false, true);
%########################################################

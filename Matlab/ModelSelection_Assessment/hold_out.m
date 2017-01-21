function [estimation, accuracy, W, W_O, eta_val, alpha_val, lambda_val] = hold_out(X, Y, trainPerc, valPerc, h_eta, h_alpha, h_lambda, outputActivationType, hiddenActivationType, lossType, estimateFunction, threshold, init_range, maxIter, layers_dims, runs, useAnnealing, useFanIn logging)

  if nargin < 20
    logging = true;
  end
  
  logConfig;
  
  if(trainPerc + valPerc >= 100)
    fprintf('train + perc needs to be < 100, taking 75% 25% 25%\n');
    trainPerc = 75; valPerc = 25;
  end
  
  valPerc = trainPerc + valPerc; % 80, 10 --> 80% train 90% validation 10 test, valPerc needs to be 90
  
  lossFunction = chooseLoss(lossType);
  hiddenActivation = chooseHidden(hiddenActivationType);
  outputActivation = chooseOutput(outputActivationType);
  
  [X,Y] = shuffle(X,Y);
  
  % Perform hold out
  m = size(X,1);
  Xtrain = X(1:floor(trainPerc*m/100), :);
  Ytrain = Y(1:floor(trainPerc*m/100), :);
  Xval = X(floor(trainPerc*m/100 + 1):floor(valPerc*m/100), :);
  Yval = Y(floor(trainPerc*m/100 + 1):floor(valPerc*m/100), :);
  Xtest = X(floor((valPerc*m/100) + 1):end, :);
  Ytest = Y(floor((valPerc*m/100) + 1):end, :);
  
  size(Xval)
  
  % Exhaustive grid search  
  minVal = Inf;
  eta_val = h_eta(1);
  alpha_val = h_alpha(1);
  lambda_val = h_lambda(1);     

  if (logging)
    logDir = startLogging(HOLD_OUT);
  end
  
  fprintf('##### HO BEGINS #####\n\n');
  
  
  % Model Selection
  for e_h = h_eta
    for a_h = h_alpha
      for l_h = h_lambda
          fprintf('##### (Eta=%f, Alpha=%f, Lambda=%f) BEGINS #####\n', e_h, a_h, l_h);
          % Select the function that minimize error on TR for these hyperparam. --> set threshold to 0
          [TR_Err, TR_Acc, nIter, Wres, W_O_res, output_hidden_units, TS_a, TS_b] = trainWithValidation(Xtrain, Ytrain, Xval, Yval, e_h, l_h, a_h, outputActivationType, hiddenActivationType, lossType, estimateFunction, threshold, init_range, maxIter, layers_dims, runs, useAnnealing, useFanIn);
          
           if(logging)
              logPlotWithTest(TR_Err, TR_Acc, nIter, TS_a, TS_b, lossType, a_h, e_h, l_h, layers_dims, size(Y,2), logDir);
           end
          
          size(Xval)
          [E,A] = modelAssessment(Wres, W_O_res, Xval, Yval, outputActivation, hiddenActivation, lossFunction, estimateFunction);
          
          Rval = E;
          Aval = A;
                  
          fprintf('VL error: %f\t VL Accuracy: %f\n', Rval, Aval);
     
          if minVal > Rval
            %fprintf('New best for eta = %f, alpha = %f and lambda = %f \n', e_h, a_h, l_h);
            eta_val = e_h;
            alpha_val = a_h;
            lambda_val = l_h;
            minVal = Rval;
          end 
          
          fprintf('##### (Eta=%f, Alpha=%f, Lambda=%f) ENDS #####\n\n', e_h, a_h, l_h);
      end
    end
  end
  
  fprintf('##### SELECTED MODEL #####\n\n');
  fprintf('Training the selected model (eta=%f, alpha=%f, lambda=%f): \n', eta_val, alpha_val, lambda_val);
  % Model Assessment
  [TR_Err, TR_Acc, nIter, W, W_O] = train([Xtrain; Xval],[Ytrain; Yval], eta_val, lambda_val, alpha_val, outputActivationType, hiddenActivationType, lossType, estimateFunction, threshold, init_range, maxIter, layers_dims, runs, useAnnealing, useFanIn);
  
  [estimation,accuracy] = modelAssessment(W, W_O, Xtest, Ytest, outputActivation, hiddenActivation, lossFunction, estimateFunction);
  
  fprintf('##### HO ENDS #####\n');
  if (logging)
    stopLogging();
  end
    
end

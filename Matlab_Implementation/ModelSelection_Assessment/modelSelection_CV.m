function [W, W_O, eta_val, alpha_val, lambda_val] = modelSelection_CV(k, X, Y, eta_range, alpha_range, lambda_range, outputActivationType, hiddenActivationType, lossType, estimateFunction, threshold, init_range, maxIter, layers_dims, runs, useAnnealing, useFanIn, showPlots, logging)
  
  if nargin < 19
    logging = true;
  end

  logConfig;
  
  %# Exhaustive grid search  
  minEst = Inf;
  minErr = Inf;
  eta_val = eta_range(1);
  alpha_val = alpha_range(1);
  lambda_val = lambda_range(1);
  
  % Variance stats for best model
  std_val  = 0;

  lossFunction = chooseLoss(lossType);
  hiddenActivation = chooseHidden(hiddenActivationType);
  outputActivation = chooseOutput(outputActivationType);
  
  n = size(X,2);
  m = size(X,1);
  
  [X,Y] = shuffle(X,Y);
  
  if (logging)
    logDir = startLogging(CROSS_VL);
  end
  
  fprintf('##### CV BEGINS #####\n\n');
  
  %# Model Selection with Cross Validation
  for e_h = eta_range
    for a_h = alpha_range
      for l_h = lambda_range
  
          err_estimates = zeros(k, 1);
          acc_estimates = zeros(k, 1);
          
          fprintf('##### (Eta=%f, Alpha=%f, Lambda=%f) BEGINS #####\n', e_h, a_h, l_h);
          
          for f=1:k
            %# Build the folders
            Dk = X((f-1)*floor(m/k)+1 : f*floor(m/k), :);
            Yk = Y((f-1)*floor(m/k)+1 : f*floor(m/k), :);
            if f == 1
              Dtr = X(f*floor(m/k) + 1:end, :);
              Ytr = Y(f*floor(m/k) + 1:end, :);
            elseif f == k
              Dtr = X(1:(f-1)*floor(m/k), :);
              Ytr = Y(1:(f-1)*floor(m/k), :);
            else
             Dtr = [X(1:(f-1)*floor(m/k), :); X(f*floor(m/k)+1:end, :)];
             Ytr = [Y(1:(f-1)*floor(m/k), :); Y(f*floor(m/k)+1:end, :)];
            end
            
            fprintf('##### FOLD %f #####\n', f);
            
            %# Select the function that minimize error on TR for these hyperparam. AND for these folders (--> set threshold to 0)
            [TR_Err, TR_Acc, nIter, W, W_O, out_hidden, TS_Err, TS_Acc] = trainWithValidation(Dtr, Ytr, Dk, Yk, e_h, l_h, a_h, outputActivationType, hiddenActivationType, lossType, estimateFunction, threshold, init_range, maxIter, layers_dims, runs, useAnnealing, useFanIn, showPlots);
            
            if(logging)
              logPlotWithTest(TR_Err, TR_Acc, nIter, TS_Err, TS_Acc, lossType, a_h, e_h, l_h, layers_dims, size(Y,2), logDir, f);
            end
                      
            %# Estimate on Dk (VL set for this fold iteration)
            [E,A] = modelAssessment(W, W_O, Dk, Yk, outputActivation, hiddenActivation, lossFunction, estimateFunction);
                    
            err_estimates(f) = E;    
            acc_estimates(f) = A;
          end

          %## HERE WE CAN COMPUTE MEAN AND ALSO STD FOR THIS CONF, ON THE FOLDS!##
          fprintf('Mean and standard deviation for the model, computed across different folds\n');
          mean_err_estimate = mean(err_estimates)
          mean_acc_estimate = mean(acc_estimates)
          %
          std_err_estimate = std(err_estimates,1)
          std_acc_estimate = std(acc_estimates,1)  
          
          % Show results
          %fprintf('TR error for eta = %f, alpha = %f,lambda = %f for k-fold \n Cost: %f \t Accuracy: %f \n', e_h, a_h, l_h, mean_err_estimate, _acc_estimate);
         

          % Keep the values for which the best (mean) estimate on k folders has been obtained
          if minEst > mean_err_estimate
              %fprintf('New best for eta = %f, alpha = %f and lambda = %f \n', e_h, a_h, l_h);
              eta_val = e_h
              alpha_val = a_h
              lambda_val = l_h
              
              minErr  = mean_err_estimate
              minEst  = mean_acc_estimate
              std_val = std_acc_estimate
              
              
          end
          fprintf('##### (Eta=%f, Alpha=%f, Lambda=%f) ENDS #####\n\n', e_h, a_h, l_h);
      end
    end
  end
  
  fprintf('##### SELECTED MODEL #####\n\n');
  fprintf('Model conf:\n');
  eta_val
  alpha_val
  lambda_val
  fprintf('Model accuracy variance:\n');
  std_val
  fprintf('Training the selected model (eta=%f, alpha=%f, lambda=%f): \n', eta_val, alpha_val, lambda_val);
  % Return the selected model
  [TR_Err, TR_Acc, nIter, W, W_O] = train(X, Y, eta_val, lambda_val, alpha_val, outputActivationType, hiddenActivationType, lossType, estimateFunction, threshold, init_range, maxIter, layers_dims, runs, useAnnealing, useFanIn, showPlots);
  
  fprintf('##### CV ENDS #####\n');
  if (logging)
    stopLogging();
  end

end

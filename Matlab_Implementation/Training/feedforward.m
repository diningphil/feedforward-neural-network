 function [prediction, nets_of_output_units, nets_of_hidden_units, outs_of_hidden_units] = feedforward(X, W, W_O, outputActivation, hiddenActivation)

  nets_of_hidden_units = {};
  outs_of_hidden_units = {};
  
  % Assumes X has already been enriched with the 1's column

  A = X'; % All the examples, transposed to obtain the same effects as with single columns
  
  % prec_out is the output of the precedent layer, at the beginning is the input matrix
  prec_out = A; 
  
  for w = W
  
    netNext = cell2mat(w)*prec_out;
    
    %compute out of next layer
    prec_out = hiddenActivation(netNext);
    
    % and add bias for this layer
    prec_out = [ones(1,size(prec_out,2)); prec_out];
    
    nets_of_hidden_units{1,end+1} = netNext';
    
    outs_of_hidden_units{1,end+1} = prec_out';
  
  end
  
  netOut = W_O * prec_out;
  
  nets_of_output_units = netOut'; % the net of the OUT units in each row (for each pattern)
  
  prediction = outputActivation(netOut)'; % transpose to return a prediction in each row (for each pattern)
  
end
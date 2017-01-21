function [weights, output_weights] = initLayers (input_dim, layers, out_dim, range, useFanIn)

  % handle base case for input
  prec_layer = input_dim;
  weights = {};
  
  for l = layers
  
    %% FAN IN %%
    if(useFanIn)
      range = 1/sqrt(prec_layer);
    end
    %%%%%%%%%%%%
    weights{1, end+1} = rem(randn(l, prec_layer+1), range); % append a matrix to this "list of matrices"
    prec_layer = l;
  end

    %% FAN IN %%
    if(useFanIn)
      range = 1/sqrt(prec_layer);
    end
    %%%%%%%%%%%%
  
    % handle last iteration for output
    output_weights = rem(randn(out_dim, prec_layer+1), range);
    
end
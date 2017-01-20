function g = sigmoidGradient(z)
    
  g = sigmoid(z).*(1 - sigmoid(z)); % can be expensive, try also with an approximation
  
end


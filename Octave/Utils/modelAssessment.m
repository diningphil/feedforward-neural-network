function [error, accuracy, O] = modelAssessment(W, W_O, X, Y, outputActivation, hiddenActivation, errorFunction, estimateMeasure)
  O = predict(X, W, W_O, outputActivation, hiddenActivation);
  error = errorFunction(O, Y);
  accuracy = estimateMeasure(O, Y);
end

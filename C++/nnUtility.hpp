#include <armadillo>
#include <iostream>
#include <vector>
#include "weights.hpp"
#include <tuple>
#include <functional>

using namespace std;
using namespace arma;

mat sigmoid(float a, mat z) {
  return 1.0 / (1.0 + exp(-a*z)); 
}

mat sigmoidGradient(float a, mat z) {
  return sigmoid(a,z)%(1 - sigmoid(a,z))*a; // % in Armadillo stands for .* in Octave
}
  
float leastMeanSquare(mat O, mat Y)  {
  mat Delta = Y - O;
  int m = Y.n_rows;
  
  // LMS
  mat square = Delta*trans(Delta); 
  return sum( square.diag() )/(2*m);
  
}

float crossEntropy(mat O, mat Y)  {
  mat Delta = Y - O;
  int m = Y.n_rows;
  
  // Cross Entropy
  return -1; //sum( sum( -Y%log(O) - (1-Y)%log(1-O) ), 1)/m; // sum on the cols, return a row vector, and sum its elements 
  // do not confuse that 1 argument, corresponds to 2 in matlab
}

float estimateAccuracy(mat O, mat Y)  {
  mat Delta = Y - O;
  int m = Y.n_rows;
  float A = 0.0;
  
  mat yS = sign(Y) - 0.5;
  mat oS = sign(O) - 0.5;

  for (int i = 0; i < m; i++)
    if( yS[i] == oS[i] )
      A += 1.0;

  return 100*A/m;
  
}

float errorFunctionWithLambda(mat O, mat Y, float lambda, Weights w, function<float (mat, mat)> errorFun) {
  mat Delta = Y - O;
  float norms = 0;
  int m = size(Y, 1);
  
  for (int l = 0; l < w.weights.size(); l++) {
    mat layerMat = w.weights[l];
    norms += pow(norm(layerMat.cols(1, layerMat.n_cols - 1), "fro"), 2);
  }
  
  norms += pow(norm(w.weights_output.cols(1, w.weights_output.n_cols - 1), "fro"), 2);

  float E = errorFun(O,Y) + lambda*(norms); // Is there something better?
  
  return E;
}

tuple<mat, mat, vector<mat>, vector<mat>> feedforward(mat X, Weights w, function< mat (mat)> activationFunction) {
    vector<mat> nets_of_hidden_units;
    vector<mat> outs_of_hidden_units;
    
    // Assumes X has already been enriched with the 1's column

    mat A = trans(X); // All the examples, transposed to obtain the same effects as with single columns
    
    // prec_out is the output of the precedent layer, at the beginning is the input matrix
    mat prec_out = A; 
    
    for (int l = 0; l < w.weights.size(); l++) {
      
      mat netNext = w.weights[l]*prec_out; // RUNTIME ERROR
      
      // compute out of next layer
      prec_out = activationFunction(netNext);
      
      // and add bias for this layer
      prec_out = join_vert(ones(1, prec_out.n_cols), prec_out);
      
      nets_of_hidden_units.push_back(trans(netNext));
      
      outs_of_hidden_units.push_back(trans(prec_out));
    
    }
    
    mat netOut = w.weights_output * prec_out;
    
    mat nets_of_output_units = trans(netOut); // the net of the OUT units in each row (for each pattern)
    
    mat prediction = trans(activationFunction(netOut)); // transpose to return a prediction in each row (for each pattern)

    // Compute the result struct
    tuple<mat, mat, vector<mat>, vector<mat>> result{ prediction, nets_of_output_units, nets_of_hidden_units, outs_of_hidden_units };
    
    return result;
}
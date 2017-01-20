 //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 //  @Input()
 //  X: input examples vector
 //  Y: input targets vector
 //  Weights: represent the layes of the NN, included the output one, separately
 //  a: sigmoid constant
 //  eta: learning rate
 //  alpha: momentum constant
 //  lambda: regularization constant
 //  epochs: max #epochs
 //  threshold: error threshold
 //
 //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#include <armadillo>
#include <iostream>
#include <vector>
#include "weights.hpp"
#include "nnUtility.hpp"
#include <tuple>
#include "config.h"
#include <cfloat>
#include <functional>

using namespace arma;
using namespace std;

tuple<vector<float>, vector<float>, int, Weights> train(mat X, mat Y,
                                                        float eta, float lambda, float alpha,
                                                        function<mat (mat)> activationFunction,
                                                        function<mat (mat)> activationGradient,
                                                        function<float (mat, mat)> errorFunction,
                                                        function<float (mat, mat)> estimateMeasure,
                                                        float threshold, int maxIter, const vector<int> layers_dims) {
  // No of training examples
  int m = X.n_rows;
  
  // Add column of ones to X;
  mat tmp; tmp.ones(m,1); 
  X = join_horiz(tmp, X);
  
  // No of input features 
  int noOfFeats = X.n_cols;
  
  // Number of output units
  int out_dim = Y.n_cols;

  Weights W_best = Weights(noOfFeats-1, layers_dims, out_dim, init_range); // Useless initialization
  float tr_best = FLT_MAX;
  vector<float> TR_Err_best;
  vector<float> TR_estimateMeasure_best;
  int nIter_best = 0;
  
  //------------------- Train different runs in order to find a better minimum ------------------//

  for (int r = 0; r < runs; r++) { 
  
    vector<float> TR_Err;
    vector<float> TR_estimateMeasure;
    Weights W = Weights(noOfFeats-1, layers_dims, out_dim, init_range); // n-1 to exclude additional 1s column
  
    // Compute initial error
    tuple<mat, mat, vector<mat>, vector<mat>> ffresult;
    
    ffresult = feedforward(X, W, activationFunction);

    mat O = get<0>(ffresult);
    mat nets_of_output_units = get<1>(ffresult);
    vector<mat> nets_of_hidden_units = get<2>(ffresult);
    vector<mat> outs_of_hidden_units = get<3>(ffresult);

    float Err_tot = errorFunction(O, Y);
    TR_Err.push_back(Err_tot);
    TR_estimateMeasure.push_back(estimateMeasure(O, Y));

    // number of training examples
    int m = X.n_rows; 
    int nIter = 1;

    // prepare the cell vector of old hidden units deltas for momentum implementation
    vector<mat> D_old;
    for(int l = 0; l < W.weights.size(); l++) { 
      D_old.push_back(zeros(W.weights[l].n_rows, W.weights[l].n_cols));
    }
    
    mat D_O_old = zeros(W.weights_output.n_rows, W.weights_output.n_cols);

    //------------------------- Backpropagation ------------------------//
    
    while ( ((Err_tot) > threshold) && (nIter < maxIter) ) {
      
      // Simple Learning-Rate Annealing
      //float eta_t = eta/(1 + nIter/m);
      
      // Simple eta
      float eta_t = eta;
      
      nIter++;
      
      // Delicate step (do not reverse the order :] )
      mat delta = (Y - O);   
      
      mat delta_out = (delta) % activationGradient(nets_of_output_units); // a row for each pattern
      // Compute weights adjustment for the least hidden layer and the output units
      
      mat D_O_grad = (eta_t/m)*(trans(delta_out)*(outs_of_hidden_units.back()) );
      mat D_O_reg  = -2*lambda*W.weights_output; 
      mat D_O_mom  = alpha*D_O_old; 
      mat D_O = D_O_grad + D_O_reg + D_O_mom;
      
      D_O_old = D_O_grad;
        
      int d = layers_dims.size()-1 ; // extract the number of deltas (=hidden layers) to compute

      mat W_O_nobias = W.weights_output.cols(1, W.weights_output.n_cols - 1);

      mat upper_layer_nobias = W_O_nobias; // matrix of the next layer (initially W_O) without bias
      mat upper_layer_delta = delta_out; // delta matrix of the upper layer (initially out)
    
      // Consider layers in reverse order
      for (int n = nets_of_hidden_units.size() - 1; n >= 0; n--) {
        
        mat cur_layer_nets = nets_of_hidden_units[n];
        mat lower_layer_out;
        if (d > 1) { 
          lower_layer_out = outs_of_hidden_units[d-1];      
        } else { // d == 1, consider the input matrix
          lower_layer_out = X;
        }
        
        mat delta_hidden = (trans(upper_layer_nobias)*trans(upper_layer_delta)) % activationGradient(trans(cur_layer_nets));
        
        //// temp variable to avoid calling cell2mat many times 
        mat W_d = W.weights[d];
        
        //// no regularization on bias (check with/wo)
        mat D_d_grad = (eta_t/m)*(delta_hidden * lower_layer_out);
        mat D_d_reg  = -2*lambda*join_horiz(zeros(W_d.n_rows, 1), W_d.cols(1, W_d.n_cols - 1));
        mat D_d_mom  = alpha*D_old[d];
        mat D_d = D_d_grad + D_d_reg + D_d_mom;
        
        D_old[d] = D_d_grad;
        upper_layer_nobias = W_d.cols(1, W_d.n_cols - 1);
        upper_layer_delta = trans(delta_hidden);
        
        // Now update weights
        // NOTE: this line MUST stay here, to avoid interference with next for-loop iterations
        W.weights[d] = W_d + D_d;

        d--;

      }
      
      // Update weights for the "rightmost" matrix
      // NOTE: this line MUST stay here, to avoid interference of D_O during the algorithm
      W.weights_output += D_O;
          
      // compute new output error
      float Err_tot = 0;
      tuple<mat, mat, vector<mat>, vector<mat>> ffresult = feedforward(X, W, activationFunction);
      mat O = get<0>(ffresult);
      mat nets_of_output_units = get<1>(ffresult);
      vector<mat> nets_of_hidden_units = get<2>(ffresult);
      vector<mat> outs_of_hidden_units = get<3>(ffresult);
      
      Err_tot = errorFunction(O, Y);
      float A = estimateMeasure(O, Y);
      TR_estimateMeasure.push_back(A);
      TR_Err.push_back(Err_tot);
      
    }
    printf("Err_tot is %.6f \n", Err_tot);
    
    if(Err_tot < tr_best) {
      W_best = W;
      tr_best = Err_tot;
      TR_Err_best = TR_Err;
      TR_estimateMeasure_best = TR_estimateMeasure;
      nIter_best = nIter;
      printf("Found new best for run %d\n", r);
    }
  } // runs loop end
  
  // Build result
  tuple<vector<float>, vector<float>, int, Weights> result{TR_Err_best, TR_estimateMeasure_best, nIter_best, W_best};
  return result;
}

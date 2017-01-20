#include <iostream>
#include <vector>
#include <math.h>
#include <tuple>
#include <armadillo>
#include "config.h"
#include "weights.hpp"
#include "train.hpp"

using namespace arma;
using namespace std;


int main() {
    mat TR;
    TR.load("../Monk_Dataset/monks2prep.train", raw_ascii);
    
    mat X = TR.cols(1, TR.n_cols-1);

    // Number of training examples
    int m = X.n_rows; 

    // Add column of ones to X;
    X = join_horiz(ones(m,1), X);

    // Training targets: MUST BE A TARGET FOR EACH ROW
   mat Y = TR.cols(0,0);  

    // Number of input units, also considering the bias
    int n = X.n_cols; 

    // Number of output units
    int out_dim = 1;

    // Total error and accuracy on training for each iteration
    vector<float> TR_Err;
    vector<float> TR_Acc;

    auto estimateMeasure = [](mat O, mat Y){ return estimateAccuracy(O,Y); };
    auto errorFunction = [](mat O, mat Y){ return leastMeanSquare(O,Y); };
    auto activ = [](mat z){ return sigmoid(a, z); };
    auto grad = [](mat z){ return sigmoidGradient(a, z); };
    
    Weights W = Weights(n-1, layers_dims, out_dim, init_range); 

    mat Xtrain = X.rows(0, floor(8*m/10) - 1);
    mat Ytrain = Y.rows(0, floor(8*m/10) - 1);
    mat Xval = X.rows(floor(8*m/10), floor(9*m/10) - 1);
    mat Yval = Y.rows(floor(8*m/10), floor(9*m/10) - 1);
    mat Xtest = X.rows(floor((9*m/10)), X.n_rows - 1);
    mat Ytest = Y.rows(floor((9*m/10)), X.n_rows - 1);


    /*----------------- ATTENZIONE!!! ------------------
        Check d linea 112 train.. forse il -1 non ci vuole
        Fixare crossEntropy

    ------------------------------------------------------*/    

    tuple<vector<float>, vector<float>, int, Weights> trRes = train(X, Y, eta, lambda, alpha, activ, grad, errorFunction, estimateMeasure, threshold, maxIter, layers_dims);
    TR_Err = get<0>(trRes);
    TR_Acc = get<1>(trRes);
    int nIter = get<2>(trRes);
    W = get<3>(trRes);

    cout << "Everything went fine, hopefully... need to plot results" << endl;

    /* Show results
    set(0, 'defaultaxesfontsize', 18);

    subplot(2,1,1);
    plot(1:nIter, TR_Acc(1:nIter));
    title("Accuracy over training data / Epochs", "fontsize", 24);
    xlabel ("Epochs");
    ylabel ("Accuracy");

    subplot(2,1,2);
    plot(1:nIter, TR_Err(1:nIter));
    title("MSE over training data / Epochs", "fontsize", 24);
    xlabel ("Epochs");
    ylabel ("MSE");

    printf("Training error on the whole dataset: %f \n", TR_Err[nIter]);
    */

    //// Perform HOLD OUT + ASSESSMENT
    // [estimation, accuracy, W, W_O, eta_val, alpha_val, lambda_val] = hold_out(W, W_O, Xtrain, Xval, Xtest, Ytrain, Yval, Ytest);
    // printf("Chosen parameters: eta = %f, alpha = %f and lambda = %f. Assessment value on TS:\n Cost = %f \t Accuracy:%f\n", eta_val, alpha_val, lambda_val, estimation, accuracy);


    //// Perform CROSS VALIDATION + ASSESSMENT
    //eta_range = [0.01 0.1 0.2 0.6 0.9];
    //alpha_range = [0.1 0.2 0.4];
    //lambda_range = [0.0001 0.0005 0.001];
    
    //[Wsel, W_Osel, eta_val, alpha_val, lambda_val] = modelSelection_CV(4, W, W_O, [Xtrain; Xval], [Ytrain; Yval], eta_range, alpha_range, lambda_range);
    //[estimation, accuracy] = modelAssessment(Wsel, W_Osel, Xtest, Ytest)
    //printf("Chosen parameters: eta = %f, alpha = %f and lambda = %f. Assessment value on TS:\n Cost = %f \t Accuracy:%f\n", eta_val, alpha_val, lambda_val, estimation, accuracy);
    return 0;
}

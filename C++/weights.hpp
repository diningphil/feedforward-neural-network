#include <armadillo>
#include <iostream> 
#include <vector>

using namespace std;
using namespace arma;

#ifndef _WEIGHTS
#define _WEIGHTS

class Weights {
   public:
      vector<mat> weights;
      mat weights_output;      
      int input_dim, output_dim;
	Weights(int in_dim, vector<int> layers, int out_dim, float range) {
		input_dim = in_dim;
		output_dim = out_dim;

		int prec_layer = input_dim;
		
		for (int l = 0; l < layers.size(); l++) {
			//cout << "Layer " << l << " rows " << layers[l] << " cols " << (prec_layer + 1) << endl;
			weights.push_back(randu(layers[l], prec_layer+1));
			prec_layer = layers[l];
		
		}
		weights_output = randu(out_dim, prec_layer+1);
	}
};
#endif

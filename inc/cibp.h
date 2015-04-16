#ifndef NPNET_CIBP
#define NPNET_CIBP

#include "npnet.h"
#include "Metroplis.h"

namespace NPnet {

    class CIBPnet {
    public:
        CIBPnet(const param& p);

        int depth;                         // depth of network.
        vec<int> num_node;         // number of nodes per layer. 
        vec<mat> weights;            // weights of the neural network.
        vec<mat> edges;                // binary edge indicators. 
        vec<colvec> biases;        // bias vectors.
        vec<colvec> stds;            // standard deviations of the prior for hidden units.

        struct Hidden {                // hidden variables.
            Hidden(const colvec& x, const CIBPnet& net) 
            :x(x), net(net) {
                this->u.resize(1);
                u[0] = x;
                this->y.resize(1);
                y[0] = x;
            }
            const CIBPnet& net;    // model.
            const colvec& x;         // input.
            vec<colvec> u;             // latent representation.
            vec<colvec> y;             // responses from previous layer.

            inline int num_layer() const {
                if(u.size() != y.size()) {
                    throw "size inconsistent between u and y.";
                }
                return u.size();
            }

            void resize(int num_layer, vec<int> num_node) {
                u.resize(num_layer);
                y.resize(num_layer);
                for(size_t i = this->num_layer(); i >= 1; i--) {
                    int old_size = u[i].size(), 
                            new_size = num_node[i];
                    if(new_size > old_size) {
                         // sample from prior.
                        if(i == this->num_layer()) {
                            y[i] = net.gm[i];
                        }else{
                            y[i] = net.W[i] * y[i+1] + net.gm[i];
                        }
                        u[i] = randn<colvec>(new_size) * net.stds[i] + y[i]; 
                        u[i] = sigmoid(u[i]);
                        // TODO: incremental computation.
                    }
                }
            }
        };

        /* train the network with given training set */
        bool train(const vec<colvec>& input, int num_iter = 10) {
            size_t num_train = input.size();
            vec<Hidden> hidden;
            for(size_t ni = 0; ni < num_train; ni++) {
                hidden.push_back(Hidden(input[ni], *this));
            }

            // Gibbs sampling.
            for(int it = 0; it < num_iter; it++) {
                for(auto& h : hidden) {
                    sample_hidden(h);
                }
                sample_weights(hidden);
                sample_biases(hidden);
                sample_types(hidden);
                sample_structure(hidden);
            }
            return true;
        }

        /* sampling */
        void sample_hidden(Hidden& hidden) {
            // resize the hidden units based on network structure.
            hidden.resize(this->depth, this->num_node);
            // sample hidden units. 
            // TODO: order of sampling.

        }

        void sample_weights(const vec<Hidden>& hidden) {
                
        }

        void sample_biases(const vec<Hidden>& hidden) {

        }

        void sample_types(const vec<Hidden>& hidden) {

        }

        void sample_structure(const vec<Hidden>& hidden) {

        }

    private:
        /* short notations for variables */
        int& M = this->depth;
        vec<int>& K = this->num_node;
        vec<mat>& W = this->weights;
        vec<mat>& Z = this->edges;
        vec<colvec>& gm = this->biases;

        /* Metroplis Hasting sampler */
        
    };
}

#endif
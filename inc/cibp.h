#ifndef NPNET_CIBP
#define NPNET_CIBP

#include "npnet.h"
#include "Metroplis.h"

namespace NPnet {

  class CIBPnet {
  public:
    /* Initialize the CIBP network with 
      input:
        p (param)
        arch (vec<int>), initial architecture, the first layer should be number of visible nodes.
      effect: 
        sets the num_node and depth.
     */
    CIBPnet(const param& p, const vec<int>& arch);

    /* disable copy and move constructors */
    CIBPnet(const CIBPnet& net);
    CIBPnet(CIBPnet&& net);

    int depth;             // depth of network, number of hidden layers.
    vec<int> num_node;     // number of nodes per layer. 
    vec<double> a, b;    // gamma prior for precision.
    vec<double> mu_w, rho_w, mu_b, rho_b;   // Guassian prior for weights and biases.
    vec<mat> weights;      // weights of the neural network.
    vec<mat> edges;        // sparse binary edge lists. 
    vec<mat> wz;         // effective weights = W .* Z.
    vec<colvec> biases;    // bias vectors.
    vec<colvec> stds;      // standard deviations of the prior for hidden units.

    struct Hidden; 
    struct Layer;

    /* train the network with given training set */
    bool train(const vec<colvec>& input, double pass = 1);

    /* sampling */
    void sample_hidden(Hidden& hidden);

    void sample_weights(const vec<Hidden>& hidden);

    void sample_biases(const vec<Hidden>& hidden);

    void sample_types(const vec<Hidden>& hidden);

    void sample_structure(const vec<Hidden>& hidden);


  private:
    /* short notations for variables */
    int& M = this->depth;
    vec<int>& K = this->num_node;
    vec<mat>& W = this->weights;
    vec<mat>& Z = this->edges;
    vec<colvec>& gm = this->biases;
  };

  /* Warning: do not move / copy Layer objects 
   * as Layer contains reference to its own member variables 
   * using a copy of the object might lead to unexpected behavior
   */
  struct CIBPnet::Layer {     // one hidden layer.

    Layer(const size_t size, size_t id, const Hidden& hidden);
    Layer(const colvec& x, const Hidden& hidden);

    // copy and move constructors disabled.
    Layer(const Layer& another_layer);
    Layer(Layer&& another_layer);

    // get size of layer.
    inline size_t size() const {
      if(u.size() != y.size()) {
        std::cout << u << std::endl;
        std::cout << v << std::endl;
        std::cout << y << std::endl;
        throw "size inconsistent between u, y or v";
      }
      return u.size();
    }

    // get std.
    inline colvec get_std() const {
      colvec std = sqrt(1/this->v);
      return std;
    }

    // activate and compute y.
    void activate();

    // sample latent representation. 
    void sample(size_t iter);

    // sample from prior. 
    void sample_prior();

    // resize layer.
    void resize(size_t num_node);

    colvec u;     // latent representation.
    colvec y;     // response from previous layer.
    const colvec& v;     // precision of nodes.
    size_t id;     // pointer previous layer.

    const Hidden& hidden;
    const CIBPnet& net;

    LikelihoodFunc lhood;
    ProposalFunc propose;
    ProposalLikelihoodFunc proposal_lhood;
    ptr<MetroplisHasting> mh;
  };

  struct CIBPnet::Hidden {        // hidden variables.
    Hidden(const colvec& x, const CIBPnet& net);

    const CIBPnet& net;  // model.
    const colvec& x;     // input.
    vec<ptr<Layer>> layer;    // hidden layers.

    inline int num_layer() const;

    void resize(vec<int> num_node);
  };

}   


#endif
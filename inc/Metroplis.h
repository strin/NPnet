#ifdef NPNET_METROPLIS
#define NPNET_METROPLIS

#include "npnet.h"

namespace NPnet {

  typedef std::function<double(colvec)> LikelihoodFunc;
  typedef std::function<colvec(colvec)> ProposalFunc;

  /* The multi-try metroplis hasting sampler */
  class MultitryMH {
  public:
    MultitryMH(LikelihoodFunc lhood, ProposalFunc propose, int sample_k) 
    :lhood(lhood), propse(propose), sample_k(sample_k) {
      num_sample = 0;
      num_acc = 0;
    }

    inline float accept_ratio() const {
      if(num_sample == 0) {
        return nan("");
      }else{
        return float(num_acc) / float(num_sample);
      }
    }

    colvec sample(const colvec& old_x) {
      vec<colvec> y(sample_k);
      vec<double> weight_xy(sample_k);
      double sumw_x = 0, sumw_y = 0;
      for(int k = 0; k < sample_k; k++) {
        y[k] = propose(old_x);
        weight_xy[k] = lhood(y[k]);
        sumw_x += weight_xy[k];
      }
      brand::discrete_distribution<vec::iterator> 
        dist(weight_xy.begin(), weight_xy.end());
      vec::iterator it = dist(global_rng);
      colvec& one_y = y[it-weight_xy.begin()];
      vec<colvec> x(sample_k);
      x[0] = old_x;
      sumw_y += lhood(x[0]);
      for(int k = 1; k < sample_k; k++) {
        x[k] = propose(one_y)
        sumw_y += lhood(x[k]);
      }
      double acc = fmin(1, sumw_x / sumw_y);
      if(acc >= brand::uniform_real<>(0, 1)(global_rng)) {
        return one_y;
      }else{
        return old_x;
      }
    }

  private:
    LikelihoodFunc lhood;
    ProposalFunc propse;
    int sample_k;
    int num_sample;
    int num_acc;
  };
}

#endif
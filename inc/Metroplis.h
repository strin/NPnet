#pragma once

#include "npnet.h"

namespace NPnet {
    /* sampling primitives for p(x) */
    // log likelihood L(x).
    typedef std::function<double(colvec)> LikelihoodFunc;
    // proposal distribution q(x | x').
    typedef std::function<colvec(colvec)> ProposalFunc;
    // proposal log likelihood L(x | x').
    typedef std::function<double(colvec, colvec)> ProposalLikelihoodFunc;
    ProposalLikelihoodFunc symmetric_proposal = [] (colvec x, colvec old_x) -> double {
      
    }

    /* Vanilla metroplis-hasting sampler */
    class BasicMH {
    public:
        BasicMH(LikelihoodFunc lhood, ProposalFunc propose, 
                                        ProposalLikelihoodFunc proposal_lhood) 
        :lhood(lhood), proposal_lhood(proposal_lhood), 
         propose(propose), num_sample(0), num_acc(0) {

        }

        inline float accept_ratio() const {
            if(num_sample == 0) {
                return nan("");
            }else{
                return float(num_acc) / float(num_sample);
            }
        }

        virtual colvec sample(const colvec& old_x) {
            colvec x = propose(old_x);
            // cout << "lhood(x) " << lhood(x) << " , lhood(old_x) " << lhood(old_x) << endl;
            double acc = fmin(0, lhood(x) - lhood(old_x));
            double coin = boost::uniform_real<>(0, 1)(global_rng);
            num_sample++;
            if(acc >= log(coin)) {
                num_acc++;
                return x;
            }else{
                return old_x;
            }
        }

    protected:
        LikelihoodFunc lhood;
        ProposalLikelihoodFunc proposal_lhood;
        ProposalFunc propose;

        /* stats */
        int num_sample;
        int num_acc;
    };

    /* The multi-try metroplis-hasting sampler */
    class MultitryMH : public BasicMH {
    public:
        MultitryMH(LikelihoodFunc lhood, ProposalFunc propose, 
                                             int sample_k, ProposalLikelihoodFunc proposal_lhood) 
        :BasicMH(lhood, propose, proposal_lhood), sample_k(sample_k) {
        }

        virtual colvec sample(const colvec& old_x) {
            vec<colvec> y(sample_k);
            vec<double> weight_xy(sample_k);
            double sumw_x = 0, sumw_y = 0, max_wxy = -DBL_MAX;

            for(int k = 0; k < sample_k; k++) {
                y[k] = propose(old_x);
                weight_xy[k] = lhood(y[k]);
                sumw_x += weight_xy[k];
                max_wxy = fmax(sumw_x, max_wxy);
            }

            for(auto& w : weight_xy) w = exp(w-max_wxy);

            brand::discrete_distribution<int, double> 
                dist(weight_xy);
            int it = dist(global_rng);
            colvec& one_y = y[it];
            vec<colvec> x(sample_k);
            x[0] = old_x;
            sumw_y += lhood(x[0]);

            for(int k = 1; k < sample_k; k++) {
                x[k] = propose(one_y);
                sumw_y += lhood(x[k]);
            }

            double acc = fmin(0, sumw_x - sumw_y);
            double coin = boost::uniform_real<>(0, 1)(global_rng);
            num_sample++;
            if(acc >= log(coin)) {
                num_acc++;
                return one_y;
            }else{
                return old_x;
            }
        }

    private:
        int sample_k;
    };
}


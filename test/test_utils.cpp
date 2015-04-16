#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test_framework.hpp>
#include "npnet.h"

#include "Metroplis.h"

using namespace std;
using namespace NPnet;

BOOST_AUTO_TEST_CASE(test_sigmoid) 
{
  colvec x({-2,-1,0,1,2});
  colvec y = sigmoid(x);
  colvec ty({0.1192,0.2689,0.5000,0.7311,0.8808});
  BOOST_CHECK(all(ty-y <= 1e-4));
}

BOOST_AUTO_TEST_CASE(test_metroplis_gaussian2d) {
    auto lhood = [] (colvec x) -> double {
        mat iSigma;
        iSigma << 2/3.0 << -1/3.0 << endr 
               << -1/3.0 << 2/3.0 << endr;
        return as_scalar(-x.t() * iSigma * x);
    };

    auto proposal = [] (colvec x) -> colvec {
        colvec noise = randn<colvec>(x.n_rows);
        return x + noise;
    };
    
    auto proposal_lhood = [](colvec x, colvec old_x) -> double {
        return as_scalar(-(x-old_x).t() * (x-old_x));
    };

    BasicMH mh(lhood, proposal, proposal_lhood);
    size_t num_sample = 10000;
    vector<colvec> samples;
    colvec x0 = randn<colvec>(2);
    colvec x = x0;

    for(size_t ni = 0; ni < num_sample; ni++) {
        x = mh.sample(x);
        samples.push_back(x);
        cout << x.t();
    }

    cout << "acc ratio = " << mh.accept_ratio() << endl;   
}

BOOST_AUTO_TEST_CASE(test_multi_try_metroplis_gaussian2d)
{
    auto lhood = [] (colvec x) -> double {
        mat iSigma;
        iSigma << 2/3.0 << -1/3.0 << endr 
               << -1/3.0 << 2/3.0 << endr;
        return as_scalar(-x.t() * iSigma * x);
    };

    auto proposal = [] (colvec x) -> colvec {
        colvec noise = randn<colvec>(x.n_rows);
        return x + noise;
    };

    auto proposal_lhood = [](colvec x, colvec old_x) -> double {
        return as_scalar(-(x-old_x).t() * (x-old_x));
    };

    MultitryMH mh(lhood, proposal, 10, proposal_lhood);
    size_t num_sample = 10000;
    vector<colvec> samples;
    colvec x0 = randn<colvec>(2);
    colvec x = x0;

    for(size_t ni = 0; ni < num_sample; ni++) {
        x = mh.sample(x);
        samples.push_back(x);
        cout << x.t();
    }

    cout << "acc ratio = " << mh.accept_ratio() << endl;
}
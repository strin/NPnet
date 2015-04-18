#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test_framework.hpp>
#include "npnet.h"
#include "cibp.h"

#include "Metroplis.h"

using namespace std;
using namespace NPnet;

BOOST_AUTO_TEST_CASE(test_sigmoid) 
{
  colvec x({-2,-1,0,1,2});
  colvec y = sigmoid(x);
  colvec ty({0.1192,0.2689,0.5000,0.7311,0.8808});
  BOOST_CHECK(all(abs(ty-y) <= 1e-4));
}

BOOST_AUTO_TEST_CASE(test_inv_sigmoid) 
{
  colvec x({-2,-1,0,1,2});
  colvec y = sigmoid(x);
  colvec z = inv_sigmoid(y);
  BOOST_CHECK(all(abs(x-z) <= 1e-4));
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
  size_t num_sample = 1000;
  vector<colvec> samples;
  colvec x0 = randn<colvec>(2);
  colvec x = x0;

  for(size_t ni = 0; ni < num_sample; ni++) {
    x = mh.sample(x);
    samples.push_back(x);
//    cout << x.t();
  }

//  cout << "acc ratio = " << mh.accept_ratio() << endl;   
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

  MultitryMH mh(lhood, proposal, proposal_lhood, 10);
  size_t num_sample = 1000;
  vector<colvec> samples;
  colvec x0 = randn<colvec>(2);
  colvec x = x0;

  for(size_t ni = 0; ni < num_sample; ni++) {
    x = mh.sample(x);
    samples.push_back(x);
//    cout << x.t();
  }

//  cout << "acc ratio = " << mh.accept_ratio() << endl;
}

BOOST_AUTO_TEST_CASE(test_cibp_net_two_pattern) 
{  
  const int dim = 2;
  CIBPnet net(param(), {dim, 2});
  colvec p1 = randu<colvec>(dim), 
       p2 = randu<colvec>(dim);
  const int n = 1000;
  vec<colvec> data;
  for(int ni = 0; ni < n; ni++) {
    if(randint(0, 1) == 0) {
      data.push_back(randber(p1));
    }else{
      data.push_back(randber(p2));
    }
//    cout << data.back() << endl;
  }
  net.train(data, 1);
}

BOOST_AUTO_TEST_CASE(test_cibp_net_sample_hidden_toy)
{
  CIBPnet net(param(), {1, 1});
  net.weights[1] = 1;
  vec<colvec> data;
  data.push_back(ones<colvec>(1) * 0.5);
  CIBPnet::Hidden hidden(data[0], net);
  double mean = 0;
  const size_t num_iter = 1000;
  for(size_t i = 0; i < num_iter; i++)
  {
    net.sample_hidden(hidden);
    auto sample = hidden.layer[1]->u;
    mean += as_scalar(sample);
  }
  mean /= double(num_iter);
  cout << "mean " << mean << endl;
  BOOST_CHECK(fabs(mean-0.551222) < 1e-4);
}

BOOST_AUTO_TEST_CASE(test_arma_load_mat) 
{
  auto res = loadmat("data/test_mat_load.mat");
  for(auto item : res) {
    cout << "key: " << item.first << endl;
    if(item.first == "data") {
      cout << boost::any_cast<cube>(item.second) << endl;
    }else{
      cout << boost::any_cast<mat>(item.second) << endl;
    }
  }
}
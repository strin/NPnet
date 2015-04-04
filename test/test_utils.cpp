#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test_framework.hpp>
#include "npnet.h"

using namespace std;
using namespace NPnet;

BOOST_AUTO_TEST_CASE(test_sigmoid) 
{
  colvec x({-2,-1,0,1,2});
  colvec y = sigmoid(x);
  colvec ty({0.1192,0.2689,0.5000,0.7311,0.8808});
  cout << y << endl;
  BOOST_CHECK(all(ty-y <= 1e-4));
}

#pragma once
#include "cibp.h"

namespace NPnet {

  /* python wrapper for CIBPnet */
  class pyCIBPnet {
  public:
    pyCIBPnet(const bpy::dict& p) 
    {
      param new_p;
      bpy::list keys = p.keys();
      for(int i = 0; i < bpy::len(keys); i++) {
        bpy::object key = keys[i];
        bpy::object value = p[key];
        new_p[key] = bpy::extract<any>(value);
      }
      this->net = make_shared<CIBPnet>(new_p);
    }

    bool train(const bnp::list& input) {
      for(int i = 0; i < bpy::len(input); i++) {
        bnp::array& x = input[i];

      }
      net->train();
    }

    ptr<CIBPnet> net;
  };
}

BOOST_PYTHON_MODULE(libcibpnet)
{
  class_<pyCIBPnet>("CIBPnet",init<const boost::python::dict&>())
    .def("train", &pyCIBPnet::train)
    ;
};

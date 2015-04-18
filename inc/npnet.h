#ifndef NPNET_MAIN
#define NPNET_MAIN

#include <boost/python.hpp>
#include <boost/any.hpp>
#include <boost/random.hpp>
// #include <boost/random/mersenne_twister.hpp>
// #include <boost/random/discrete_distribution.hpp>
#include <functional>
#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include <armadillo>
#include "matio.h"

namespace arma {

    /* extension on randomness */
    template<>
    double randn<double>() 
    {
      return as_scalar(randn<colvec>(1));
    }

    double randg(distr_param dist) 
    {
      return as_scalar(randg<colvec>(1, dist));
    }

    /* boost randomness */
    namespace brand = boost::random;
    using randgen = boost::mt19937;
    static randgen global_rng;
    
    /* std randomness */
    std::default_random_engine global_std_rng;
    
    template<class V>
    V randber(const V& p) 
    {
        V res = p;
        typename V::const_iterator it = p.begin();
        typename V::iterator rt = res.begin();
        for(;it != p.end() and rt != res.end(); it++, rt++)
        {
            std::bernoulli_distribution rg(*it);
            *rt = rg(global_std_rng);
        }
        return res;
    }

    /* io with mat files */
    std::unordered_map<std::string, boost::any> 
    loadmat(std::string path) 
    {
        mat_t* matfp;
        matvar_t* matvar;
        matfp = Mat_Open(path.c_str(),MAT_ACC_RDONLY);
        std::unordered_map<std::string, boost::any> res;

        if(matfp == nullptr) 
            return res;

         while((matvar = Mat_VarReadNextInfo(matfp)) != nullptr) 
         {
            std::string name(matvar->name);
            if(matvar->data_type == MAT_T_CELL) 
            {
                throw "cell type not supported";
            }
            else
            {
                if(matvar->class_type == MAT_C_DOUBLE 
                   and (matvar->rank == 2))
                {
                    int start[2] = {0, 0};
                    int stride[2] = {1, 1};
                    const int& n0 = matvar->dims[0];
                    const int& n1 = matvar->dims[1];
                    int edge[2] = {n0, n1};
                    static std::vector<double> x(n0 * n1);
                    Mat_VarReadData(matfp, matvar, &x[0], start, stride, edge);
                    mat x2(x);
                    x2.reshape(n0, n1);
                    res[name] = x2;
                }
                else if(matvar->class_type == MAT_C_DOUBLE
                        and matvar->rank == 3) 
                {
                    int start[3] = {0, 0, 0};
                    int stride[3] = {1, 1, 1};
                    const int& n0 = matvar->dims[0];
                    const int& n1 = matvar->dims[1];
                    const int& n2 = matvar->dims[2];
                    int edge[3] = {n0, n1, n2};
                    static std::vector<double> x(n0 * n1 * n2);
                    Mat_VarReadData(matfp, matvar, &x[0], start, stride, edge);
                    cube x3(n1, n2, n0);
                    for(int i = 0; i < n0; i++) {
                        for(int j = 0; j < n1; j++) {
                            for(int k = 0; k < n2; k++) {
                                x3(j, k, i) = x[k * n0 * n1 + j * n0 + i];
                            }
                        }
                    }
                    res[name] = x3;
                }
                else
                {
                    throw "unknow type in mat"; 
                }
            }
            Mat_VarFree(matvar);
         }
         return res;
    }
}

namespace NPnet 
{
    /* namespace resolution */
    using str = std::string;

    template<class T>
    using vec = std::vector<T>;

    template<class T>
    using vec2d = vec<vec<T>>;

    template<class T>
    using ptr = std::shared_ptr<T>;

    using any = boost::any;

    template<class T, class... Args>
    ptr<T> make_shared(Args&&... args) 
    {
        return ptr<T>(new T(std::forward<Args>(args)...));
    }

    template<class K, class T>
    class map : public std::unordered_map<K, T> 
    {
    public:
        bool contains(K key) {
            return this->find(key) != this->end();
        }
    };

    using param = map<str, any>;

    template<class T, class... Args>
    T extract(Args&&... args) {
        return boost::python::extract<T>(args...);
    }

    using namespace arma;

    /* boost python */
    namespace bpy = boost::python;
    namespace bnp = boost::python::numeric;

    template<>
    mat extract<mat>(const bnp::array& arr) {
        bpy::tuple shape = static_cast<bpy::tuple>(arr.getshape());
        if(bpy::len(shape) != 2) {
            throw "unable to extract: array    shape size is not 2.";
        }
        int nrows = bpy::extract<int>(shape[0]), 
        ncols = bpy::extract<int>(shape[1]);
        mat res = zeros(nrows, ncols);
        for(int i = 0; i < nrows; i++) {
            for(int j = 0; j < ncols; j++) {
                res(i,j) = bpy::extract<double>(arr[bpy::make_tuple(i, j)]);
            }
        }
        return res;
    }


    /* activation function */
    auto sigmoid = [] (const colvec& x) {
        colvec y = x;
        for(auto& elem : y) {
            elem = 1/(1+exp(-elem));
        }
        return y;
    };

    auto inv_sigmoid = [] (const colvec& y) {
        colvec x = y;
        for(auto& elem : x) {
            x = log(elem) - log(1-elem);
        }
        return x;
    };

    

    /* arma randomness */
    static inline double randint(int a, int b) {
        return as_scalar(randi(1, distr_param(a,b)));
    }

}

#endif
#include "cibp.h"

using namespace NPnet;

///////////////////////////////// CIBPnet ///////////////////////////////////////////////
CIBPnet::CIBPnet(const param& p, const vec<int>& arch) 
  :num_node(arch), depth(arch.size()-1) 
{
  a.resize(depth+1);
  b.resize(depth+1);
  mu_w.resize(depth+1);
  rho_w.resize(depth+1);
  biases.resize(depth+1);
  stds.resize(depth+1);
  weights.resize(depth+1);
  edges.resize(depth+1);
  wz.resize(depth+1);
  
  for(size_t i = 0; i <= depth; i++) {
    a[i] = 1;
    b[i] = 1;
    mu_w[i] = 0;
    rho_w[i] = 1;
//    biases[i] = randu<colvec>(num_node[i]);
    biases[i] = zeros<colvec>(num_node[i]);
    // biases /= norm(biases, 2);
    stds[i] = randg<colvec>(num_node[i], distr_param(a[i], b[i]));
    if(i > 0) {
      size_t k = num_node[i-1];
      size_t kp = num_node[i];
      weights[i] = randn<mat>(k, kp) * sqrt(1/rho_w[i]) + mu_w[i];
      // weights[i] /= norm(weights[i], 2);
      edges[i] = ones<mat>(k, kp);
      wz[i] = weights[i] % edges[i];
    }
  }
}

CIBPnet::CIBPnet(const CIBPnet& net) 
  :num_node(net.num_node), depth(net.depth)
{
  throw "CIBPnet cannot be copied";
}

CIBPnet::CIBPnet(CIBPnet&& net)
  :num_node(net.num_node), depth(net.depth)
{
  throw "CIBPnet cannot be copied";
}

bool CIBPnet::train(const vec<colvec>& input, double pass) {
  size_t num_train = input.size();
  vec<Hidden> hidden;
  for(size_t ni = 0; ni < num_train; ni++) {
    hidden.push_back(Hidden(input[ni], *this));
  }

  // Gibbs sampling.
  for(int it = 0; it < int(pass * input.size()); it++) {
    int ni = it % input.size();
    auto& h = hidden[ni];
    sample_hidden(h);
  //   sample_weights(hidden);
  //   sample_biases(hidden);
  //   sample_types(hidden);
  //   sample_structure(hidden);
  }
  return true;
}

void CIBPnet::sample_hidden(Hidden& hidden) 
{
  // resize the hidden units based on network structure.
  hidden.resize(this->num_node);

  // sample hidden units. 
  // TODO: order of sampling.
  assert(hidden.num_layer() == this->depth);
  for(size_t li = 1; li <= hidden.num_layer(); li++)
  {
    hidden.layer[li]->activate();
    hidden.layer[li]->sample(1);
  }

}

void CIBPnet::sample_weights(const vec<Hidden>& hidden) 
{
  for(size_t li = 1; li <= depth; li++) 
  {
    // sample weights.
    const auto& np = num_node[li-1];
    const auto& n = num_node[li];
    rowvec B = zeros<rowvec>(n);
    colvec prec = 1 / (stds[li-1] * stds[li-1]);
    mat A = zeros<mat>(np, n);
    for(size_t kp = 0; kp < num_node[li]; kp++) 
    {
      for(auto& h : hidden) 
      {
        auto& up = h.layer[li-1]->u;
        auto& u = h.layer[li]->u;
        auto& yp = h.layer[li-1]->y;
        A.col(kp) += u[kp] * 
            (inv_sigmoid(up) - yp + wz[li].col(kp) * u[kp]);
        B[kp] += u[kp] * u[kp];
      }
      A.col(kp) %= prec;
    }
    auto rho_w_post = rho_w[li] + prec * B;
    auto mu_w_post = (mu_w[li] * rho_w[li] + A) / rho_w_post;
    this->weights[li] = randn(np, n) * sqrt(1/rho_w_post) + mu_w_post;

    // sample biases.
    colvec C = zeros<colvec>(n);
    for(auto& h : hidden) 
    {
      auto& u = h.layer[li]->u;
      auto& y = h.layer[li]->y;
      C += inv_sigmoid(u) - y + biases[li];
    }
    auto rho_b_post = rho_b[li] + double(hidden.size()) * prec;
    auto mu_b_post = (mu_b[li] * rho_b[li] + prec * C) / rho_b_post;
    this->biases[li] = randn<colvec>(n) * sqrt(1 / rho_b_post) + mu_b_post;
  }
}

void CIBPnet::sample_types(const vec<Hidden>& hiddens) 
{
  const double N = double(hiddens.size());
  for(size_t li = 0; li <= depth; li++) 
  {
    const size_t& n = num_node[li];
    colvec a_post = ones<colvec>(n) * (a[li] + N/2.0);
    colvec mse = zeros<colvec>(n);
    for(auto& h : hiddens) 
    {
      mse += pow(inv_sigmoid(h.layer[li]->u) - h.layer[li]->y, 2);
    }
    colvec b_post = b[li] + .5 * mse;
    colvec prec = zeros<colvec>(n);
    for(size_t i = 0; i < n; i++) 
    {
      prec[i] = randg(distr_param(a_post[i], b_post[i]));
    }
    stds[li] = sqrt(1/prec);
  }
}


///////////////////////////////// Layer ///////////////////////////////////////////////
CIBPnet::Layer::Layer(const size_t size, size_t id, const Hidden& hidden) 
:hidden(hidden), net(hidden.net), id(id), v(hidden.net.stds[id]) {

  lhood = [&] (colvec x) -> double
  {
    size_t id = this->id;
    auto y = (net.W[id] % net.Z[id]) * x + net.gm[id-1];
    colvec inv_next_u = inv_sigmoid(hidden.layer[id-1]->u);
    const colvec& next_v = hidden.layer[id-1]->v;
    cout << "x " << x << " , " << " y " << y << endl;
    return as_scalar(-sum(next_v % (y - inv_next_u) % (y - inv_next_u) / 2));
  };

  propose = [&] (colvec x)  -> colvec
  {
    const size_t layer_size = this->size();
    colvec new_x = randn<colvec>(layer_size);
    return sigmoid(this->get_std() % new_x + this->y);
  };

  proposal_lhood = [&] (colvec x, colvec old_x) -> double {
    auto inv_x = inv_sigmoid(x);
    return as_scalar(-sum(v % (y - inv_x) % (y - inv_x)) / 2);
  };
  
  mh = std::make_shared<BasicMH>(lhood, propose, proposal_lhood);

}

CIBPnet::Layer::Layer(const colvec& x, const Hidden& hidden)
:hidden(hidden), net(hidden.net), id(0), v(hidden.net.stds[id]) {
  this->u = x;
}

CIBPnet::Layer::Layer(const Layer& another_layer)
  :hidden(another_layer.hidden), net(hidden.net), v(another_layer.v)
{
    throw "Layer objects should not be copied";
}

CIBPnet::Layer::Layer(Layer&& another_layer) 
  :hidden(another_layer.hidden), net(hidden.net), v(another_layer.v)
{
    throw "Layer objects should not be moved";
}

void CIBPnet::Layer::activate() {
  if(id == hidden.num_layer()) { // top layer.
    this->y = net.gm[id];
  }else{
    this->y = (net.W[id] % net.Z[id]) * hidden.layer[id+1]->u + net.gm[id];
  }
}

// sample latent representation. 
void CIBPnet::Layer::sample(size_t iter) 
{
  if(this->id == 0) throw "cannot sample visible units";
  for(size_t it = 0; it < iter; it++) {
    this->u = mh->sample(this->u);
  }
}

// sample from prior. 
void CIBPnet::Layer::sample_prior() {
  const size_t layer_size = this->size();
  colvec new_x = randn<colvec>(layer_size);
  this->u = this->get_std() % new_x + this->y;
}

// resize layer.
void CIBPnet::Layer::resize(size_t num_node) 
{
  size_t old_size = this->size();
  this->u.resize(num_node);
  this->y.resize(num_node);
  if(num_node > old_size) 
  {  // add nodes.
    if(id == hidden.num_layer()) {
      y.rows(old_size, num_node-1) = net.gm[id].rows(old_size, num_node-1);
    }else{
      mat w = net.W[id] % net.Z[id];
      y.rows(old_size, num_node-1) = w.rows(old_size, num_node-1) * y[id+1];
    }
    // v.rows(old_size, num_node-1) = randg<colvec>(num_node-old_size, 
    //                 distr_param(net.a[id], net.b[id]));
    colvec prec = sqrt(1/v.rows(old_size, num_node-1));
    u.rows(old_size, num_node-1) = prec % randn<colvec>(num_node-old_size) 
                               + y.rows(old_size, num_node-1);
  }
}

///////////////////////////////// TITLE: ///////////////////////////////////////////////
CIBPnet::Hidden::Hidden(const colvec& x, const CIBPnet& net) 
  :x(x), net(net) 
{
  // initialize with one hidden layer.
  this->layer.push_back(std::make_shared<Layer>(x, *this));
  this->resize(net.num_node);
}

inline int CIBPnet::Hidden::num_layer() const {
  return layer.size() - 1;
}

void CIBPnet::Hidden::resize(vec<int> num_node) {
  assert(num_node.size() > 0 and num_node[0] == x.size());
  for(size_t i = this->layer.size(); i < num_node.size(); i++) {
    layer.push_back(std::make_shared<Layer>(num_node[i], i, *this));
  }
  for(size_t i = this->num_layer(); i >= 1; i--) {
    layer[i]->resize(num_node[i]);
  }
}

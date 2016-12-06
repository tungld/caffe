#ifndef CAFFE_PARALLEL_HPP_
#define CAFFE_PARALLEL_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/net.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.
template<typename Dtype>
class Params {
 public:
  explicit Params(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~Params() {
  }

  inline size_t size() const {
    return size_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }

 protected:
  const size_t size_;           // Size of buffers
  Dtype* data_;                 // Network parameters
  Dtype* diff_;                 // Gradient

DISABLE_COPY_AND_ASSIGN(Params);
};

// Params stored in GPU memory.
template<typename Dtype>
class GPUParams : public Params<Dtype> {
 public:
  GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device);
  virtual ~GPUParams();

  void configure(Solver<Dtype>* solver) const;

 protected:
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

class DevicePair {
 public:
  DevicePair(int parent, int device)
      : parent_(parent),
        device_(device) {
  }
  inline int parent() {
    return parent_;
  }
  inline int device() {
    return device_;
  }

  // Group GPUs in pairs, by proximity depending on machine's topology
  static void compute(const vector<int> devices, vector<DevicePair>* pairs);

 protected:
  int parent_;
  int device_;
};

// Synchronous data parallelism using map-reduce between local GPUs.
template<typename Dtype>
class P2PSync : public GPUParams<Dtype>, public Solver<Dtype>::Callback,
    public InternalThread {
 public:
  explicit P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                   P2PSync<Dtype>* parent, const SolverParameter& param);
  virtual ~P2PSync();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run(const vector<int>& gpus);
  void Prepare(const vector<int>& gpus,
               vector<shared_ptr<P2PSync<Dtype> > >* syncs);
  inline const int initial_iter() const { return initial_iter_; }

 protected:
  void on_start();
  void on_inner_iteration(int inner_iter){}
  void on_gradients_ready();

  void InternalThreadEntry();

  P2PSync<Dtype>* parent_;
  vector<P2PSync<Dtype>*> children_;
  BlockingQueue<P2PSync<Dtype>*> queue_;
  const int initial_iter_;
  Dtype* parent_grads_;
  shared_ptr<Solver<Dtype> > solver_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

/**
 * Data parallelism using overlap of backpropagation and communication
 */
template<typename Dtype>
class OverlapSync : public GPUParams<Dtype>, public Solver<Dtype>::Callback,
		    public Net<Dtype>::Callback, public Solver<Dtype>::ICallback,
		    public InternalThread {
 public:
  explicit OverlapSync(shared_ptr<Solver<Dtype> > root_solver,
		       OverlapSync<Dtype>* parent, const SolverParameter& param,
		       Dtype* grads, vector<BlockingQueue<int>* >* criticals_free,
		       int chunk, int threshold);
  virtual ~OverlapSync();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run(const vector<int>& gpus);
  void Prepare(const vector<int>& gpus,
               vector<shared_ptr<OverlapSync<Dtype> > >* syncs);
  inline const int initial_iter() const { return initial_iter_; }
  #ifndef CPU_ONLY
  static void CUDART_CB callback_grads(cudaStream_t stream,
				       cudaError_t status,
				       void* tp){
    OverlapSync<Dtype>* sync = (OverlapSync<Dtype>*)tp;
    sync->accumulate_gradients();
  }
  static void CUDART_CB callback_reset_variables(cudaStream_t stream,
						 cudaError_t status,
						 void* tp){
    OverlapSync<Dtype>* sync = (OverlapSync<Dtype>*)tp;
    sync->reset_variables();
  }
  #endif
 protected:
  void on_init();
  void on_start();
  void on_inner_iteration(int inner_iter);
  void on_gradients_ready();
  void on_gradients_layers_ready(int l);
  void accumulate_gradients();
  void reset_variables();
  void InternalThreadEntry();
  
  OverlapSync<Dtype>* parent_;
  vector<OverlapSync<Dtype>*> children_;
  BlockingQueue<OverlapSync<Dtype>*> queue_;
  const int initial_iter_;
  shared_ptr<Solver<Dtype> > solver_;

  // a shared array on host to store the summation of gradients
  Dtype* grads_;
  // gradients on cpu of a solver
  Dtype* cpu_diff_;
  // blobs of learnable parameters that the solver computed
  // during the backward
  BlockingQueue<vector<int> > ready_blobs_;
  // queues to sync callbacks for each layer
  vector<BlockingQueue<int>* >* criticals_free_;
  // mapping the id of a blob in the blobs of learnable parameters to 
  // its id in the shared array grads_ and diff_
  vector<int> pid_aid_;
  vector<int> pid_size_;
  // the number of blobs of learnable parameters
  int blobs_num_;
  // the number of solvers/gpus
  int solvers_num_;
  // the layer that has had the updated accumulation on host
  // this layer is ready to send the accum. on host to GPU
  int updated_layer_;
  // these are used to transfer data between host and devices
  #ifndef CPU_ONLY
  cudaStream_t d2h_h_stream_;
  cudaStream_t h2d_stream_;
  #endif
  // iteration index if iter_size is set
  int inner_iter_;
  
  // command line arguments
  int chunk_;
  int threshold_;
  
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif

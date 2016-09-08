#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "boost/thread/barrier.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"

#ifdef USE_NVTX
#include "caffe/util/mark_profile.hpp"
#endif

#include "caffe/sgd_solvers.hpp"

namespace caffe {

shared_ptr<boost::barrier> barrier_;
  
enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
    : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
      data_(),
      diff_() {
}

template<typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  // Allocate device buffers
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(Dtype)));

  // Copy blob values
  const vector<Blob<Dtype>*>& net =
      root_solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);

  CUDA_CHECK(cudaMalloc(&diff_, size_ * sizeof(Dtype)));
  caffe_gpu_set(size_, Dtype(0), diff_);

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaFree(data_));
  CUDA_CHECK(cudaFree(diff_));
#endif
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
  apply_buffers(net, data_, size_, replace_gpu);
  apply_buffers(net, diff_, size_, replace_gpu_diff);
}

void DevicePair::compute(const vector<int> devices, vector<DevicePair>* pairs) {
#ifndef CPU_ONLY
  vector<int> remaining(devices);

  // Depth for reduction tree
  int remaining_depth = static_cast<int>(ceil(log2(remaining.size())));

  // Group GPUs by board
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        cudaDeviceProp a, b;
        CUDA_CHECK(cudaGetDeviceProperties(&a, remaining[i]));
        CUDA_CHECK(cudaGetDeviceProperties(&b, remaining[j]));
        if (a.isMultiGpuBoard && b.isMultiGpuBoard) {
          if (a.multiGpuBoardGroupID == b.multiGpuBoardGroupID) {
            pairs->push_back(DevicePair(remaining[i], remaining[j]));
            DLOG(INFO) << "GPU board: " << remaining[i] << ":" << remaining[j];
            remaining.erase(remaining.begin() + j);
            break;
          }
        }
      }
    }
  }
  ostringstream s;
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by boards, remaining: " << s.str();

  // Group by P2P accessibility
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        int access;
        CUDA_CHECK(
            cudaDeviceCanAccessPeer(&access, remaining[i], remaining[j]));
        if (access) {
          pairs->push_back(DevicePair(remaining[i], remaining[j]));
          DLOG(INFO) << "P2P pair: " << remaining[i] << ":" << remaining[j];
          remaining.erase(remaining.begin() + j);
          break;
        }
      }
    }
  }
  s.str("");
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by P2P access, remaining: " << s.str();

  // Group remaining
  while (remaining.size() > 1) {
    for (int i = 0; i+1 < remaining.size(); ++i) {
      pairs->push_back(DevicePair(remaining[i], remaining[i + 1]));
      DLOG(INFO) << "Remaining pair: " << remaining[i] << ":"
                 << remaining[i + 1];
      remaining.erase(remaining.begin() + i + 1);
    }
  }

  // Should only be the parent node remaining
  CHECK_EQ(remaining.size(), 1);

  pairs->insert(pairs->begin(), DevicePair(-1, remaining[0]));

  CHECK(pairs->size() == devices.size());
  for (int i = 0; i < pairs->size(); ++i) {
    CHECK((*pairs)[i].parent() != (*pairs)[i].device());
    for (int j = i + 1; j < pairs->size(); ++j) {
      CHECK((*pairs)[i].device() != (*pairs)[j].device());
    }
  }
#else
  NO_GPU;
#endif
}

//

template<typename Dtype>
P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                        P2PSync<Dtype>* parent, const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      parent_(parent),
      children_(),
      queue_(),
      initial_iter_(root_solver->iter()),
      solver_() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  if (parent) {
    // Enable p2p access between devices
    const int peer = parent->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      LOG(INFO)<< "GPU " << self << " does not have p2p access to GPU " << peer;
    }
    // Allocate receiving buffer on parent
    CUDA_CHECK(cudaSetDevice(peer));
    CUDA_CHECK(cudaMalloc(&parent_grads_, size_ * sizeof(Dtype)));
    CUDA_CHECK(cudaSetDevice(self));
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent_) {
    CUDA_CHECK(cudaFree(parent_grads_));
    const int peer = parent_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void P2PSync<Dtype>::on_start() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#else
//  CHECK(false);
#endif

  // Wait for update from parent
  if (parent_) {
    P2PSync<Dtype> *parent = queue_.pop();
    CHECK(parent == parent_);
  }

  // Update children
  for (int i = children_.size() - 1; i >= 0; --i) {
    Dtype* src = data_;
    Dtype* dst = children_[i]->data_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == children_[i]->solver_->param().device_id());
#endif

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    children_[i]->queue_.push(this);
  }
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::on_gradients_ready() {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif

  // Sum children gradients as they appear in the queue
  for (int i = 0; i < children_.size(); ++i) {
    P2PSync<Dtype> *child = queue_.pop();
    Dtype* src = child->parent_grads_;
    Dtype* dst = diff_;

#ifdef DEBUG
    bool ok = false;
    for (int j = 0; j < children_.size(); ++j) {
      if (child == children_[j]) {
        ok = true;
      }
    }
    CHECK(ok);
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == device);
#endif

    caffe_gpu_add(size_, src, dst, dst);
  }

  // Send gradients to parent
  if (parent_) {
    Dtype* src = diff_;
    Dtype* dst = parent_grads_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == parent_->solver_->param().device_id());
#endif

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),  //
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    parent_->queue_.push(this);
  } else {
    // Loss functions divide gradients by the batch size, so to compensate
    // for split batch, the root solver divides by number of solvers.
    caffe_gpu_scal(size_, Dtype(1.0 / Caffe::solver_count()), diff_);
  }
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::Prepare(const vector<int>& gpus,
            vector<shared_ptr<P2PSync<Dtype> > >* syncs) {
  // Pair devices for map-reduce synchronization
  vector<DevicePair> pairs;
  DevicePair::compute(gpus, &pairs);
  ostringstream s;
  for (int i = 1; i < pairs.size(); ++i) {
    s << (i == 1 ? "" : ", ") << pairs[i].parent() << ":" << pairs[i].device();
  }
  LOG(INFO)<< "GPUs pairs " << s.str();

  SolverParameter param(solver_->param());

  // Build the GPU tree by finding the parent for each solver
  for (int attempts = 0; attempts < pairs.size(); ++attempts) {
    for (int i = 1; i < pairs.size(); ++i) {
      if (!syncs->at(i).get()) {
        P2PSync<Dtype>* parent = NULL;
        for (int j = 0; j < syncs->size(); ++j) {
          P2PSync<Dtype>* sync = j == 0 ? this : syncs->at(j).get();
          if (sync) {
            const SolverParameter& p = sync->solver()->param();
            if (p.device_id() == pairs[i].parent()) {
              parent = sync;
            }
          }
        }
        if (parent) {
          param.set_device_id(pairs[i].device());
          syncs->at(i).reset(new P2PSync<Dtype>(solver_, parent, param));
          parent->children_.push_back((P2PSync<Dtype>*) syncs->at(i).get());
        }
      }
    }
  }
}

template<typename Dtype>
void P2PSync<Dtype>::Run(const vector<int>& gpus) {
  vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
  Prepare(gpus, &syncs);

  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  solver_->Solve();

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}

//  OverlapSync
template<typename Dtype>
OverlapSync<Dtype>::OverlapSync(shared_ptr<Solver<Dtype> > root_solver,
			OverlapSync<Dtype>* parent, const SolverParameter& param,
				Dtype* grads, vector<BlockingQueue<int>* >* criticals_free,
			int chunk, int threshold)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      parent_(parent),
      children_(),
      queue_(),
      initial_iter_(root_solver->iter()),
      solver_(),
      grads_(grads),
      criticals_free_(criticals_free),
      chunk_(chunk), threshold_(threshold){
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new WorkerSGDSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->net()->MapLayerLearnableParams();
  solver_->add_callback(this);
  solver_->net()->add_callback(this);

  if (parent) {
    // Enable p2p access between devices
    const int peer = parent->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      LOG(INFO)<< "GPU " << self << " does not have p2p access to GPU " << peer;
    }
  }

  CUDA_CHECK(cudaMallocHost((void**)&cpu_diff_, size_ * sizeof(Dtype)));

  // Map layer indices to 1D array indices
  const vector<Blob<Dtype>*>& net =
      solver_->net()->learnable_params();
  blobs_num_ = net.size();
  solvers_num_ = Caffe::solver_count();
  
  int idx = 0;
  for (int i = 0; i < net.size(); ++i) {
    int size = net[i]->count();
    pid_aid_.push_back(idx);
    pid_size_.push_back(size);
    idx += size;
  }

  CUDA_CHECK(cudaStreamCreateWithFlags(&d2h_h_stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&h2d_stream_, cudaStreamNonBlocking));
  
  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
OverlapSync<Dtype>::~OverlapSync() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent_) {
    const int peer = parent_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }
  
  cudaFreeHost(cpu_diff_);

  CUDA_CHECK(cudaStreamDestroy(d2h_h_stream_));
  CUDA_CHECK(cudaStreamDestroy(h2d_stream_));
  
  CUDA_CHECK(cudaSetDevice(initial_device));
#endif
}

template<typename Dtype>
void OverlapSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }

  on_init();
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void OverlapSync<Dtype>::on_init() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#else
//  CHECK(false);
#endif

  // Wait for update from parent
  if (parent_) {
    OverlapSync<Dtype> *parent = queue_.pop();
    CHECK(parent == parent_);
  }

  // Update children
  for (int i = children_.size() - 1; i >= 0; --i) {
    Dtype* src = data_;
    Dtype* dst = children_[i]->data_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == children_[i]->solver_->param().device_id());
#endif

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    children_[i]->queue_.push(this);
  }
#endif
}

template<typename Dtype>
void OverlapSync<Dtype>::on_start() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#else
//  CHECK(false);
#endif

  CUDA_CHECK(cudaDeviceSynchronize());
  // Wait for other solvers
  barrier_->wait();
  
  // the root solver clears gradients on the host
  if (parent_ == NULL) {
    cudaStreamAddCallback(d2h_h_stream_, OverlapSync<Dtype>::callback_reset_variables, 
  			  (void*)this, 0);
  }
  updated_layer_ = blobs_num_ / (chunk_ * 2) - 1;

#endif
}

template<typename Dtype>
void OverlapSync<Dtype>::reset_variables() {
  // CPUTimer timer;
  // timer.Start();

  // reset the global gradients
  memset(grads_, 0, size_ * sizeof(Dtype));
  // reset queues

  for (int i = 0; i < criticals_free_->size(); ++i){
    criticals_free_->at(i)->pop();
    criticals_free_->at(i)->push(0);
  }

  // timer.Stop();
  // LOG(INFO) << "reset time " << timer.MicroSeconds() << " us";
}
  
template<typename Dtype>
void OverlapSync<Dtype>::on_gradients_layers_ready(int l) {
#ifndef CPU_ONLY
  // send previous layer's gradients to gpu
  if (updated_layer_ >= 0){
    int updated_solvers = 0;
    if (criticals_free_->at(updated_layer_)->try_peek(&updated_solvers)){
      if (updated_solvers == solvers_num_){
	int lid = updated_layer_ * (chunk_ * 2);
	int offset = pid_aid_.at(lid);
	int size = 0;
    
	for (int i = lid; i < lid + (chunk_ * 2); ++i){
	  size += pid_size_.at(i);
	}

	CUDA_CHECK(cudaMemcpyAsync(diff_ + offset, grads_ + offset, sizeof(Dtype) * size,
				   cudaMemcpyHostToDevice,
				   h2d_stream_));
	--updated_layer_;
      }
    }
  }

  // send current gradients to host and do accumulation
  const vector<int> learnable_params_id_vecs = solver_->net()
    ->learnable_params_id_vecs(l);

  if (learnable_params_id_vecs.size() > 0 && (learnable_params_id_vecs[0] % (chunk_ * 2) == 0)){
    int lid = -1;
    for (int i = learnable_params_id_vecs.size() - 1; i >= 0; --i){
      if (learnable_params_id_vecs[i] % (chunk_ * 2) == 0) {
	lid = learnable_params_id_vecs[i];
	break;
      }
    }

    if (lid >= 0){
      int glid = lid / (chunk_ * 2);
#ifdef USE_NVTX
      ostringstream msg;
      msg << "postlayer " << glid;
      if (glid % 2) {
	push_nvmark_range(msg.str(), 5);
      } else {
	push_nvmark_range(msg.str(), 6);
      }
#endif
      
      int offset = pid_aid_.at(lid);
      int size = 0;
    
      for (int i = lid; i < lid + (chunk_ * 2); ++i){
	size += pid_size_.at(i);
      }

      if (size > 0){ // Copy blob values and do accumulation
	vector<int> vt;
	vt.push_back(offset);
	vt.push_back(size);
	vt.push_back(glid);
	ready_blobs_.push(vt);

	CUDA_CHECK(cudaMemcpyAsync(cpu_diff_ + offset, diff_ + offset,
				   sizeof(Dtype) * size, cudaMemcpyDeviceToHost,
				   d2h_h_stream_));
	cudaStreamAddCallback(d2h_h_stream_, OverlapSync<Dtype>::callback_grads, 
			      (void*)this, 0);
      }
#ifdef USE_NVTX
      pop_nvmark_range();
#endif
    }
  }
#endif
}

template<typename Dtype>
void OverlapSync<Dtype>::accumulate_gradients() {
  vector<int> vt = ready_blobs_.pop();
  int offset = vt[0];
  int size = vt[1];
  int glid = vt[2];

  // Add up local gradients (on GPU) to the global gradients (on CPU)
  Dtype* acc = grads_ + offset;
  Dtype* src = cpu_diff_ + offset;

  // CPUTimer timer;
  // timer.Start();
  int idx = criticals_free_->at(glid)->pop();
  if (idx == solvers_num_) { idx = 0; }
#ifdef USE_NVTX
  ostringstream msg;
  msg << "[" << solver_->param().device_id() << "] [CPU] Accum. for " << (float)(size * sizeof(Dtype) / (float)1000 / (float)1000) << " MB";
  push_nvmark_range(msg.str(), 0);
#endif
  
  if (size < threshold_) {
    for(int i = 0; i < size; ++i) {
      acc[i] += src[i];
    }
  } else {
#pragma omp parallel for
    for(int i = 0; i < size; ++i) {
      acc[i] += src[i];
    }      
  }
  criticals_free_->at(glid)->push(++idx);
  // timer.Stop();
  // LOG(INFO) << size << " parameters, accumulation time " << timer.MicroSeconds() << " us";

#ifdef USE_NVTX
  pop_nvmark_range();
#endif
}

template<typename Dtype>
void OverlapSync<Dtype>::on_gradients_ready() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif
  CUDA_CHECK(cudaStreamSynchronize(d2h_h_stream_));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  // send remaining gradients on host to devices
  while (updated_layer_ >= 0) {
    int updated_solvers = 0;
    if (criticals_free_->at(updated_layer_)->try_peek(&updated_solvers)){
      if (updated_solvers == solvers_num_){
	int lid = updated_layer_ * (chunk_ * 2);
	int offset = pid_aid_.at(lid);
	int size = 0;
    
	for (int i = lid; i < lid + (chunk_ * 2); ++i){
	  size += pid_size_.at(i);
	}

	CUDA_CHECK(cudaMemcpyAsync(diff_ + offset, grads_ + offset, sizeof(Dtype) * size,
				   cudaMemcpyHostToDevice,
				   h2d_stream_));
	--updated_layer_;
      } else {
	continue;
      }
    } else {
      break;
    }
  }
 
  // Wait for the last stream finished
  CUDA_CHECK(cudaStreamSynchronize(h2d_stream_));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));

  // Loss functions divide gradients by the batch size, so to compensate
  // for split batch, the root solver divides by number of solvers.
  caffe_gpu_scal(size_, Dtype(1.0 / Caffe::solver_count()), diff_);
#endif
}

template<typename Dtype>
void OverlapSync<Dtype>::Prepare(const vector<int>& gpus,
            vector<shared_ptr<OverlapSync<Dtype> > >* syncs) {
  // Pair devices for map-reduce synchronization
  vector<DevicePair> pairs;
  DevicePair::compute(gpus, &pairs);
  ostringstream s;
  for (int i = 1; i < pairs.size(); ++i) {
    s << (i == 1 ? "" : ", ") << pairs[i].parent() << ":" << pairs[i].device();
  }
  LOG(INFO)<< "GPUs pairs " << s.str();

  SolverParameter param(solver_->param());

  // Build the GPU tree by finding the parent for each solver
  for (int attempts = 0; attempts < pairs.size(); ++attempts) {
    for (int i = 1; i < pairs.size(); ++i) {
      if (!syncs->at(i).get()) {
        OverlapSync<Dtype>* parent = NULL;
        for (int j = 0; j < syncs->size(); ++j) {
          OverlapSync<Dtype>* sync = j == 0 ? this : syncs->at(j).get();
          if (sync) {
            const SolverParameter& p = sync->solver()->param();
            if (p.device_id() == pairs[i].parent()) {
              parent = sync;
            }
          }
        }
        if (parent) {
          param.set_device_id(pairs[i].device());
          syncs->at(i).reset(new OverlapSync<Dtype>(solver_, parent, param, grads_, criticals_free_, chunk_, threshold_));
          parent->children_.push_back((OverlapSync<Dtype>*) syncs->at(i).get());
        }
      }
    }
  }
}

template<typename Dtype>
void OverlapSync<Dtype>::Run(const vector<int>& gpus) {
  vector<shared_ptr<OverlapSync<Dtype> > > syncs(gpus.size());
  Prepare(gpus, &syncs);
  barrier_.reset(new boost::barrier(gpus.size()));
  
  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  on_init();
  solver_->Solve();
  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(GPUParams);
INSTANTIATE_CLASS(P2PSync);
INSTANTIATE_CLASS(OverlapSync);

}  // namespace caffe

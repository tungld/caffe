#ifndef CAFFE_UTIL_MARK_PROFILE_H_
#define CAFFE_UTIL_MARK_PROFILE_H_

// colors and defines are from  https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/

#include <string>
#include "nvToolsExt.h"

namespace caffe {

  void push_nvmark_range(const std::string& name, int cid);
  
  void pop_nvmark_range();

  void nvmark_event(const std::string& name);
} // namespace caffe

#endif   // CAFFE_UTIL_MARK_PROFILE_H_

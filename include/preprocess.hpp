#pragma once

#include <pcl/point_cloud.h>
#include <Eigen/Dense>

namespace m9::perception::chromaloom {

template <typename PointCloudT>
void preprocess_reflect(typename PointCloudT::Ptr cloud_ptr,
                        const Eigen::Vector3d &camera_location,
                        double radius,
                        typename PointCloudT::Ptr out_cloud_ptr);

} // namespace m9::perception::chromaloom

// include impl
// Instead of including the .cu file, declare that the implementation will be provided elsewhere
#ifdef __CUDACC__
#include "preprocess.cu"
#endif

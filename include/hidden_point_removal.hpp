#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>

namespace m9::perception::chromaloom {

template <typename PointCloudT>
std::vector<size_t> hidden_point_removal_inliers(typename PointCloudT::Ptr cloud_ptr,
                                                 const Eigen::Vector3d &camera_location, double radius = 100.);

template <typename PointCloudT>
std::vector<size_t> hidden_point_removal_inliers(typename PointCloudT::Ptr cloud_ptr, const cv::Mat &extrinsics,
                                                 double radius = 100.);

template <typename PointCloudT>
typename PointCloudT::Ptr hidden_point_removal(typename PointCloudT::Ptr cloud_ptr,
                                               const Eigen::Vector3d &camera_location, double radius = 100.);

template <typename PointCloudT>
typename PointCloudT::Ptr hidden_point_removal(typename PointCloudT::Ptr cloud_ptr, const cv::Mat &extrinsics,
                                               double radius = 100.);

}  // namespace m9::perception::chromaloom

#include "hidden_point_removal_impl.hpp"

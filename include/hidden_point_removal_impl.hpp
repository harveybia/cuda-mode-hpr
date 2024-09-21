#pragma once

#include "hidden_point_removal.hpp"

#include <opencv2/opencv.hpp>

#undef USE_UNORDERED_MAP
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/convex_hull.h>

namespace m9::perception::chromaloom {

template <typename PointCloudT>
std::vector<size_t> hidden_point_removal_inliers(typename PointCloudT::Ptr cloud_ptr,
                                                 const Eigen::Vector3d &camera_location, double radius) {
    using InPointCloudType = typename PointCloudT::Ptr::element_type;
    using InPointType = typename InPointCloudType::PointType;
    using LabelPointType = pcl::PointXYZL;

    std::vector<Eigen::Vector3d> spherical_projection;
    pcl::PointCloud<LabelPointType>::Ptr new_cloud_ptr(new pcl::PointCloud<LabelPointType>);

    // Step 1: perform spherical projection
    for (size_t idx = 0; idx < cloud_ptr->points.size(); ++idx) {
        const InPointType current_point = cloud_ptr->points[idx];
        Eigen::Vector3d current_vector(current_point.x, current_point.y, current_point.z);
        Eigen::Vector3d projected_point = current_vector - camera_location;
        double norm = std::max(projected_point.norm(), 0.0001);

        spherical_projection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
    }

    // add origin to spherical projection
    size_t origin_idx = spherical_projection.size();
    spherical_projection.push_back(Eigen::Vector3d(0, 0, 0));
    assert(spherical_projection.size() == cloud_ptr->points.size() + 1);

    // convert to pointcloud for convex hull
    for (size_t idx = 0; idx < spherical_projection.size(); ++idx) {
        Eigen::Vector3d current_vector = spherical_projection.at(idx);
        LabelPointType current_point;
        current_point.x = current_vector.x();
        current_point.y = current_vector.y();
        current_point.z = current_vector.z();
        current_point.label = idx;
        new_cloud_ptr->push_back(current_point);
    }

    // Step 2: perform convex hull of origin + spherical projection
    // all the points that are in the convex hull are visible from origin.
    pcl::PointCloud<LabelPointType>::Ptr cloud_hull(new pcl::PointCloud<LabelPointType>);
    pcl::ConvexHull<LabelPointType> chull;
    chull.setInputCloud(new_cloud_ptr);
    chull.reconstruct(*cloud_hull);

    // Step 3: use original indices to construct visible pointcloud
    std::vector<size_t> visible_indices;
    for (size_t idx = 0; idx < cloud_hull->points.size(); ++idx) {
        LabelPointType current_point = cloud_hull->points[idx];
        uint32_t point_index = current_point.label;
        if (point_index >= origin_idx) {
            continue;
        }

        visible_indices.push_back(point_index);
    }

    return visible_indices;
}

template <typename PointCloudT>
std::vector<size_t> hidden_point_removal_inliers(typename PointCloudT::Ptr cloud_ptr, const cv::Mat &extrinsics,
                                                 double radius) {
    return hidden_point_removal_inliers<PointCloudT>(cloud_ptr,
                                                     Eigen::Vector3d{
                                                         extrinsics.at<double>(0, 3),
                                                         extrinsics.at<double>(1, 3),
                                                         extrinsics.at<double>(2, 3),
                                                     },
                                                     radius);
}

template <typename PointCloudT>
typename PointCloudT::Ptr hidden_point_removal(typename PointCloudT::Ptr cloud_ptr,
                                               const Eigen::Vector3d &camera_location, double radius) {
    using InPointCloudType = typename PointCloudT::Ptr::element_type;
    using InPointType = typename InPointCloudType::PointType;
    auto inliers = hidden_point_removal_inliers<PointCloudT>(cloud_ptr, camera_location, radius);

    typename PointCloudT::Ptr visible_cloud_ptr(new InPointCloudType);

    for (auto inlier : inliers) {
        InPointType current_point = cloud_ptr->points[inlier];
        visible_cloud_ptr->push_back(current_point);
    }

    return visible_cloud_ptr;
}

template <typename PointCloudT>
typename PointCloudT::Ptr hidden_point_removal(typename PointCloudT::Ptr cloud_ptr, const cv::Mat &extrinsics,
                                               double radius) {
    return hidden_point_removal<PointCloudT>(cloud_ptr,
                                             Eigen::Vector3d{
                                                 extrinsics.at<double>(0, 3),
                                                 extrinsics.at<double>(1, 3),
                                                 extrinsics.at<double>(2, 3),
                                             },
                                             radius);
}

}  // namespace m9::perception::chromaloom

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

namespace cuda
{

    template <typename PointT>
    class ConvexHull
    {
    public:
        ConvexHull();
        ~ConvexHull();

        void setInputCloud(const typename pcl::PointCloud<PointT>::Ptr &cloud);
        void setDimension(int dimension);
        void setComputeAreaVolume(bool compute);

        void reconstruct(pcl::PointCloud<PointT> &output);

        double getTotalArea() const;
        double getTotalVolume() const;

    private:
        typename pcl::PointCloud<PointT>::ConstPtr input_cloud_;
        int dimension_;
        bool compute_area_;
        double total_area_;
        double total_volume_;

        // CUDA-specific members
        PointT *d_input_cloud_;
        PointT *d_output_cloud_;
        int *d_hull_indices_;
        size_t input_size_;
        size_t output_size_;

        void allocateDeviceMemory();
        void freeDeviceMemory();
        void copyInputToDevice();
        void copyOutputFromDevice(pcl::PointCloud<PointT> &output);
    };

} // namespace cuda

// Instead of including the .cu file, declare that the implementation will be provided elsewhere
#ifdef __CUDACC__
#include "cuda_convex_hull.cu"
#endif
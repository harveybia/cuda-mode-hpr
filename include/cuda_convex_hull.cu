#include "cuda_convex_hull.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace cuda
{

    template <typename PointT>
    __global__ void convexHullKernel(const PointT *input, PointT *output, int *hull_indices, size_t input_size, size_t *output_size)
    {
        // Implement your CUDA kernel for convex hull algorithm here
    }

    template <typename PointT>
    ConvexHull<PointT>::ConvexHull()
        : dimension_(3), compute_area_(false), total_area_(0), total_volume_(0),
          d_input_cloud_(nullptr), d_output_cloud_(nullptr), d_hull_indices_(nullptr),
          input_size_(0), output_size_(0) {}

    template <typename PointT>
    ConvexHull<PointT>::~ConvexHull()
    {
        freeDeviceMemory();
    }

    template <typename PointT>
    void ConvexHull<PointT>::setInputCloud(const typename pcl::PointCloud<PointT>::Ptr &cloud)
    {
        input_cloud_ = cloud;
        input_size_ = cloud->size();
        allocateDeviceMemory();
        copyInputToDevice();
    }

    template <typename PointT>
    void ConvexHull<PointT>::setDimension(int dimension)
    {
        dimension_ = dimension;
    }

    template <typename PointT>
    void ConvexHull<PointT>::setComputeAreaVolume(bool compute)
    {
        compute_area_ = compute;
    }

    template <typename PointT>
    void ConvexHull<PointT>::reconstruct(pcl::PointCloud<PointT> &output)
    {
        // Implement the main convex hull algorithm using CUDA
        // Call the CUDA kernel here

        // Example kernel launch (you'll need to adjust the parameters):
        // convexHullKernel<<<gridSize, blockSize>>>(d_input_cloud_, d_output_cloud_, d_hull_indices_, input_size_, &output_size_);

        // Copy the results back to the host
        copyOutputFromDevice(output);

        // Compute area and volume if requested
        if (compute_area_)
        {
            // Implement area and volume computation
        }
    }

    template <typename PointT>
    void ConvexHull<PointT>::reconstruct(pcl::PointCloud<PointT> &output, std::vector<pcl::Vertices> &polygons)
    {
        // Implement reconstruction with polygons
        // This might require additional CUDA kernels or host-side post-processing
    }

    template <typename PointT>
    double ConvexHull<PointT>::getTotalArea() const
    {
        return total_area_;
    }

    template <typename PointT>
    double ConvexHull<PointT>::getTotalVolume() const
    {
        return total_volume_;
    }

    template <typename PointT>
    void ConvexHull<PointT>::allocateDeviceMemory()
    {
        cudaMalloc(&d_input_cloud_, input_size_ * sizeof(PointT));
        cudaMalloc(&d_output_cloud_, input_size_ * sizeof(PointT)); // Worst case: all points are on the hull
        cudaMalloc(&d_hull_indices_, input_size_ * sizeof(int));
    }

    template <typename PointT>
    void ConvexHull<PointT>::freeDeviceMemory()
    {
        if (d_input_cloud_)
            cudaFree(d_input_cloud_);
        if (d_output_cloud_)
            cudaFree(d_output_cloud_);
        if (d_hull_indices_)
            cudaFree(d_hull_indices_);
    }

    template <typename PointT>
    void ConvexHull<PointT>::copyInputToDevice()
    {
        cudaMemcpy(d_input_cloud_, input_cloud_->points.data(), input_size_ * sizeof(PointT), cudaMemcpyHostToDevice);
    }

    template <typename PointT>
    void ConvexHull<PointT>::copyOutputFromDevice(pcl::PointCloud<PointT> &output)
    {
        output.resize(output_size_);
        cudaMemcpy(output.points.data(), d_output_cloud_, output_size_ * sizeof(PointT), cudaMemcpyDeviceToHost);
    }

    // Explicit instantiation for common point types
    template class ConvexHull<pcl::PointXYZ>;
    template class ConvexHull<pcl::PointXYZI>;
    template class ConvexHull<pcl::PointXYZRGB>;

} // namespace cuda
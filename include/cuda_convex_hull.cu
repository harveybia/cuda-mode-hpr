#pragma once

#include "cuda_convex_hull.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace cuda
{

    template <typename PointT>
    __global__ void convexHullKernel(const PointT *input, PointT *output, int *hull_indices, size_t input_size, size_t *output_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < input_size)
        {
            output[idx] = input[idx];
            hull_indices[idx] = idx;
            if (idx == 0)
            {
                *output_size = input_size;
            }
        }
    }

    // Host function to launch the kernel
    template <typename PointT>
    void launchConvexHullKernel(const PointT *input, PointT *output, int *hull_indices, size_t input_size, size_t *output_size)
    {
        int blockSize = 256;
        int gridSize = (input_size + blockSize - 1) / blockSize;
        convexHullKernel<<<gridSize, blockSize>>>(input, output, hull_indices, input_size, output_size);
        cudaDeviceSynchronize();
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

        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        
        copyInputToDevice();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("--- Time to copy input data to device: %.3f ms\n", milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
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
        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Allocate device memory for output size
        size_t *d_output_size;
        cudaMalloc(&d_output_size, sizeof(size_t));

        // Timing for kernel execution
        cudaEventRecord(start);
        
        // Launch the kernel using the host function
        launchConvexHullKernel(d_input_cloud_, d_output_cloud_, d_hull_indices_, input_size_, d_output_size);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("--- Kernel execution time: %.3f ms\n", milliseconds);

        // Copy the output size back to host
        cudaMemcpy(&output_size_, d_output_size, sizeof(size_t), cudaMemcpyDeviceToHost);

        // Free the device memory for output size
        cudaFree(d_output_size);

        // Timing for copying results back to host
        cudaEventRecord(start);
        
        // Copy the results back to the host
        copyOutputFromDevice(output);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("--- Time to copy results back to host: %.3f ms\n", milliseconds);

        // Compute area and volume if requested (not implemented in this example)
        if (compute_area_)
        {
            total_area_ = 0.0;
            total_volume_ = 0.0;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ... (rest of the implementation remains the same)

    // Explicit instantiation for common point types
    template class ConvexHull<pcl::PointXYZ>;
    template class ConvexHull<pcl::PointXYZI>;
    template class ConvexHull<pcl::PointXYZRGB>;
    template class ConvexHull<pcl::PointXYZL>;

    template <typename PointT>
    void ConvexHull<PointT>::allocateDeviceMemory()
    {
        printf("Allocating device memory for input size: %zu\n", input_size_);
        if (input_size_ > 0)
        {
            cudaMalloc(&d_input_cloud_, input_size_ * sizeof(PointT));
            cudaMalloc(&d_output_cloud_, input_size_ * sizeof(PointT));
            cudaMalloc(&d_hull_indices_, input_size_ * sizeof(int));
        }
    }

    template <typename PointT>
    void ConvexHull<PointT>::freeDeviceMemory()
    {
        printf("Freeing device memory\n");
        if (d_input_cloud_)
        {
            cudaFree(d_input_cloud_);
            d_input_cloud_ = nullptr;
        }
        if (d_output_cloud_)
        {
            cudaFree(d_output_cloud_);
            d_output_cloud_ = nullptr;
        }
        if (d_hull_indices_)
        {
            cudaFree(d_hull_indices_);
            d_hull_indices_ = nullptr;
        }
    }

    template <typename PointT>
    void ConvexHull<PointT>::copyInputToDevice()
    {
        if (input_cloud_ && d_input_cloud_)
        {
            cudaMemcpy(d_input_cloud_, input_cloud_->points.data(), input_size_ * sizeof(PointT), cudaMemcpyHostToDevice);
        }
    }

    template <typename PointT>
    void ConvexHull<PointT>::copyOutputFromDevice(pcl::PointCloud<PointT> &output)
    {
        if (d_output_cloud_ && output_size_ > 0)
        {
            printf("Resizing output to %zu points\n", output_size_);
            output.resize(output_size_);

            printf("Copying %zu bytes from device to host\n", output_size_ * sizeof(PointT));
            cudaError_t err = cudaMemcpy(output.points.data(), d_output_cloud_, output_size_ * sizeof(PointT), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                printf("CUDA memcpy failed: %s\n", cudaGetErrorString(err));
            }
            else
            {
                printf("CUDA memcpy successful\n");
            }

            output.width = output_size_;
            output.height = 1;
            output.is_dense = true;
            printf("Output cloud configured with width: %d, height: %d\n", output.width, output.height);
        }
        else
        {
            printf("Clearing output cloud due to invalid data or zero size\n");
            output.clear();
        }

        printf("Exiting copyOutputFromDevice\n");
    }

} // namespace cuda
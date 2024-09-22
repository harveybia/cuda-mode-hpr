#include <Eigen/Dense>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace m9::perception::chromaloom {

// Preprocess CUDA Kernel
struct CudaPointXYZ {
  float x;
  float y;
  float z;
  float padding;
}; // struct CudaPointXYZ

struct CudaPointXYZL {
  float x;
  float y;
  float z;
  float label;
}; // struct CudaPointXYZL

/*  ------ Kernel Functions ------ */
// preprocess points to perform spherical reflection
__global__ void kernel_preprocess_reflect(const CudaPointXYZ *input_points,
                                          const size_t* num_points,
                                          const double* radius,
                                          const CudaPointXYZ* camera_center,
                                          CudaPointXYZ *output_points) {
  // get point index
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > *num_points) return;
  const auto point = input_points[idx];

  // convert to camera-center coordinate
  const auto point_in_cam_x = point.x - camera_center->x;
  const auto point_in_cam_y = point.y - camera_center->y;
  const auto point_in_cam_z = point.z - camera_center->z;

  // get point norm
  const auto norm = sqrt(point_in_cam_x * point_in_cam_x +
                         point_in_cam_y * point_in_cam_y +
                         point_in_cam_z * point_in_cam_z);
  
  // get reflected point
  CudaPointXYZ out_point;
  out_point.x = point_in_cam_x + 2 * (*radius - norm) * point_in_cam_x / norm;
  out_point.y = point_in_cam_y + 2 * (*radius - norm) * point_in_cam_y / norm;
  out_point.z = point_in_cam_z + 2 * (*radius - norm) * point_in_cam_z / norm;

  output_points[idx] = out_point;
}


/*  ------ Interface Impl ------ */
template <typename PointCloudT>
void preprocess_reflect(typename PointCloudT::Ptr cloud_ptr,
                        const Eigen::Vector3d &camera_location,
                        double radius,
                        typename PointCloudT::Ptr out_cloud_ptr) {

  using InPointCloudType = typename PointCloudT::Ptr::element_type;
  using InPointType = typename InPointCloudType::PointType;

  const size_t num_points = cloud_ptr->size();

  CudaPointXYZ camera_center;
  camera_center.x = camera_location[0];
  camera_center.y = camera_location[1];
  camera_center.z = camera_location[2];

  // alloc GPU memory blocks
  void* gpu_in_cloud_ptr;
  void* gpu_num_points;
  void* gpu_radius;
  void* gpu_camera_center;
  void* gpu_out_cloud_ptr;
  cudaMalloc(&gpu_in_cloud_ptr, sizeof(CudaPointXYZ) * num_points);
  cudaMalloc(&gpu_num_points, sizeof(size_t));
  cudaMalloc(&gpu_radius, sizeof(double));
  cudaMalloc(&gpu_camera_center, sizeof(CudaPointXYZ));
  cudaMalloc(&gpu_out_cloud_ptr, sizeof(CudaPointXYZ) * num_points);

  // assign value to GPU memory blocks
  cudaMemcpy(gpu_in_cloud_ptr, cloud_ptr->points.data(), num_points * sizeof(CudaPointXYZ), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_num_points, &num_points, sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_radius, &radius, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_camera_center, &camera_center, sizeof(CudaPointXYZ), cudaMemcpyHostToDevice);

  // assuming point cloud type PointXYZ
  const dim3 threads_per_block(1024);
  const dim3 num_blocks((num_points / threads_per_block.x) + 1);
  kernel_preprocess_reflect<<<num_blocks, threads_per_block>>>(reinterpret_cast<const CudaPointXYZ*>(gpu_in_cloud_ptr),
                                                               reinterpret_cast<const size_t*>(gpu_num_points),
                                                               reinterpret_cast<const double*>(gpu_radius),
                                                               reinterpret_cast<const CudaPointXYZ*>(gpu_camera_center),
                                                               reinterpret_cast<CudaPointXYZ*>(gpu_out_cloud_ptr));
  cudaDeviceSynchronize();

  // copy result back to host
  cudaMemcpy(out_cloud_ptr->points.data(), gpu_out_cloud_ptr, sizeof(CudaPointXYZ) * num_points, cudaMemcpyDeviceToHost);

  // free cuda mem
  cudaFree(gpu_in_cloud_ptr);
  cudaFree(gpu_num_points);
  cudaFree(gpu_radius);
  cudaFree(gpu_camera_center);
  cudaFree(gpu_out_cloud_ptr);

  return;
}

// Explicit instantiation for common point types
template void preprocess_reflect<pcl::PointCloud<pcl::PointXYZ>>(
    pcl::PointCloud<pcl::PointXYZ>::Ptr,
    const Eigen::Vector3d&,
    double,
    pcl::PointCloud<pcl::PointXYZ>::Ptr);

}  // namespace m9::perception::chromaloom

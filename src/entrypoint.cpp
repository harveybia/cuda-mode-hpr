#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "hidden_point_removal.hpp"

Eigen::Vector3d read_camera_location(const std::string &file_path)
{
    std::ifstream file(file_path);
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    std::vector<double> values;
    std::string value;
    while (std::getline(iss, value, ','))
    {
        values.push_back(std::stod(value));
    }
    if (values.size() != 3)
    {
        throw std::runtime_error("Invalid camera location file format");
    }
    return Eigen::Vector3d(values[0], values[1], values[2]);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_pointcloud.pcd> <camera_location.txt> [--radius <value>] [--output <output_pointcloud.pcd>]" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string camera_location_file = argv[2];
    double radius = 100.0;
    std::string output_file;

    for (int i = 3; i < argc; i += 2)
    {
        std::string arg = argv[i];
        if (arg == "--radius" && i + 1 < argc)
        {
            radius = std::stod(argv[i + 1]);
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            output_file = argv[i + 1];
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, *cloud) == -1)
    {
        std::cerr << "Couldn't read file " << input_file << std::endl;
        return 1;
    }

    Eigen::Vector3d camera_location = read_camera_location(camera_location_file);

    auto result = m9::perception::chromaloom::hidden_point_removal<pcl::PointCloud<pcl::PointXYZ>>(cloud, camera_location, radius);

    if (!output_file.empty())
    {
        pcl::io::savePCDFileBinary(output_file, *result);
        std::cout << "Saved result to " << output_file << std::endl;
    }

    return 0;
}
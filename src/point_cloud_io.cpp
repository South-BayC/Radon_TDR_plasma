#include "point_cloud_io.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

// ============================================================
// PCL 3D Visualization
// ============================================================

void PointCloudIO::visualizePCL(const std::vector<Point3D>& points,
                                 const std::string& windowTitle) {
    using CloudT = pcl::PointCloud<pcl::PointXYZI>;
    CloudT::Ptr cloud(new CloudT());
    cloud->points.reserve(points.size());

    for (const auto& p : points) {
        pcl::PointXYZI q;
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        q.intensity = p.intensity;
        cloud->points.push_back(q);
    }
    cloud->width = static_cast<std::uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;

    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer(windowTitle));
    viewer->setBackgroundColor(0.05, 0.05, 0.06);
    viewer->addPointCloud<pcl::PointXYZI>(cloud, "tdr_cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "tdr_cloud");
    viewer->addCoordinateSystem(50.0);
    viewer->initCameraParameters();

    std::cout << "PCL visualization window opened (program continues after closing window)." << std::endl;
    while (!viewer->wasStopped()) {
        viewer->spinOnce(16);
    }
}

// ============================================================
// PLY File Export
// ============================================================

bool PointCloudIO::savePLY(const std::vector<Point3D>& points,
                            const std::string& filePath) {
    if (points.empty()) return false;

    // Find min/max intensity
    float minIntensity = points[0].intensity;
    float maxIntensity = points[0].intensity;
    for (const auto& p : points) {
        minIntensity = std::min(minIntensity, p.intensity);
        maxIntensity = std::max(maxIntensity, p.intensity);
    }
    float intensityRange = maxIntensity - minIntensity;

    std::ofstream ofs(filePath);
    if (!ofs.is_open()) {
        std::cerr << "Error: Cannot open PLY file for writing: " << filePath << std::endl;
        return false;
    }

    ofs << "ply\nformat ascii 1.0\nelement vertex " << points.size()
        << "\nproperty float x\nproperty float y\nproperty float z"
        << "\nproperty float intensity"
        << "\nproperty uchar red\nproperty uchar green\nproperty uchar blue"
        << "\nend_header\n";

    for (const auto& p : points) {
        float normalized = (intensityRange > 0)
            ? (p.intensity - minIntensity) / intensityRange
            : 0.5f;
        normalized = std::max(0.0f, std::min(1.0f, normalized));
        int gray = static_cast<int>(normalized * 255.0f);
        ofs << p.x << " " << p.y << " " << p.z << " "
            << p.intensity << " "
            << gray << " " << gray << " " << gray << "\n";
    }

    std::cout << "PLY point cloud saved: " << filePath
              << " (" << points.size() << " points)" << std::endl;
    return true;
}

// ============================================================
// Side View (XZ Projection) Image
// ============================================================

void PointCloudIO::generateSideView(const std::vector<Point3D>& points,
                                     const std::string& outputPath) {
    if (points.empty()) return;

    // Find intensity range
    float minIntensity = points[0].intensity;
    float maxIntensity = points[0].intensity;
    for (const auto& p : points) {
        minIntensity = std::min(minIntensity, p.intensity);
        maxIntensity = std::max(maxIntensity, p.intensity);
    }

    const float theoreticalRadius = TARGET_IMAGE_SIZE / 2.0f;
    const float theoreticalZMax = static_cast<float>(TARGET_IMAGE_SIZE - 1);

    const int imgSize = TARGET_IMAGE_SIZE;
    cv::Mat xzView(imgSize, imgSize, CV_8UC3, cv::Scalar(0, 0, 0));

    float intensityRange = maxIntensity - minIntensity;
    if (intensityRange <= 0.0f) intensityRange = 1.0f;

    for (const auto& p : points) {
        float normalized = (p.intensity - minIntensity) / intensityRange;
        normalized = std::max(0.0f, std::min(1.0f, normalized));

        int grayValue = static_cast<int>(normalized * 255.0f);
        grayValue = std::max(0, std::min(255, grayValue));
        cv::Vec3b color(grayValue, grayValue, grayValue);

        // XZ projection: X→image column, Z→image row (inverted for Y-up)
        float normX = p.x / theoreticalZMax;
        int x_pixel = static_cast<int>(normX * (imgSize - 1));

        float vMin = -theoreticalRadius;
        float vMax = theoreticalRadius;
        int z_pixel = static_cast<int>((vMax - p.z) / (vMax - vMin) * (imgSize - 1));

        if (x_pixel >= 0 && x_pixel < imgSize && z_pixel >= 0 && z_pixel < imgSize) {
            xzView.at<cv::Vec3b>(z_pixel, x_pixel) = color;
        }
    }

    std::string outFile = outputPath + "point_cloud_side_view.png";
    cv::imwrite(outFile, xzView);
    std::cout << "Side view saved: " << outFile << std::endl;
}

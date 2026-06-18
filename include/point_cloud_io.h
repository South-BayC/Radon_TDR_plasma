#ifndef POINT_CLOUD_IO_H
#define POINT_CLOUD_IO_H

#include "config.h"
#include <vector>
#include <string>

// 3D point with intensity value
struct Point3D {
    float x;
    float y;
    float z;
    float intensity;
};

// Utility class for point cloud output operations
class PointCloudIO {
public:
    // Export point cloud as ASCII PLY file
    static bool savePLY(const std::vector<Point3D>& points,
                        const std::string& filePath);

    // Render side-view (XZ) projection image from point cloud
    static void generateSideView(const std::vector<Point3D>& points,
                                  const std::string& outputPath);

    // Launch interactive PCL 3D visualization
    static void visualizePCL(const std::vector<Point3D>& points,
                              const std::string& windowTitle);
};

#endif // POINT_CLOUD_IO_H

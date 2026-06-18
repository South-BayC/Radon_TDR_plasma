#ifndef RADON_RECONSTRUCTION_H
#define RADON_RECONSTRUCTION_H

#include "config.h"
#include "point_cloud_io.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class RadonReconstructor {
private:
    std::string inputPath;
    std::string outputPath;
    std::string intermediatePath;

    cv::Mat processedImage;
    cv::Mat reconstructedImage;
    int imageWidth;
    int imageHeight;

    enum FilterType {
        RamLak = 0,
        HannRamp = 1,
        SheppLogan = 2
    };

    FilterType filterType;
    int kernelRadius;
    float radialMaskFactor;
    int edgeTaperPixels;

    cv::Mat filterProjection(const cv::Mat& projection) const;

    cv::Mat reconstructRadonFromColumn(int columnIndex, bool saveIntermediate = false, float smoothSigma = 2.0f) const;

public:
    RadonReconstructor();
    RadonReconstructor(const std::string& inputDir, const std::string& outputDir,
        const std::string& intermediateDir);

    void setInputPath(const std::string& path);
    void setOutputPath(const std::string& path);
    void setIntermediatePath(const std::string& path);

    bool loadProcessedImage(const std::string& filename);

    bool reconstructSingleColumn(const std::string& inputImage, int columnIndex, float smoothSigma = 2.0f);

    bool perform3DReconstruction(const std::string& inputImage, bool saveIntermediate = true, float smoothSigma = 2.0f);

    bool saveResults(const std::string& baseName);

    void setFilterType(int type);
    void setKernelRadius(int radius);
    void setRadialMaskFactor(float factor);
    void setEdgeTaperPixels(int pixels);
};

#endif

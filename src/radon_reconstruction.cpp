#include "radon_reconstruction.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <direct.h>

// PCL includes for visualization
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

// Thresholds
#ifndef POINT_CLOUD_THRESHOLD_RATIO
#define POINT_CLOUD_THRESHOLD_RATIO 0.01f
#endif

// Real-time visualization using PCL
static void visualizePointCloudRealtimePCL(
	const std::vector<Point3D>& points,
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

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(windowTitle));
	viewer->setBackgroundColor(0.05, 0.05, 0.06);
	viewer->addPointCloud<pcl::PointXYZI>(cloud, "tdr_cloud");
	viewer->setPointCloudRenderingProperties(
		pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "tdr_cloud");
	viewer->addCoordinateSystem(50.0);
	viewer->initCameraParameters();

	std::cout << "PCL visualization window opened (program continues/ends after closing window)." << std::endl;
	while (!viewer->wasStopped()) {
		viewer->spinOnce(16);
	}
}

RadonReconstructor::RadonReconstructor()
    : reconstructedImage(), imageWidth(0), imageHeight(0) {
    inputPath = "data/proceed/";
    outputPath = "data/output/";
    intermediatePath = "data/proceed/";
    filterType = HannRamp;
    kernelRadius = 63;
    radialMaskFactor = 0.9f;
    edgeTaperPixels = 8;
}

RadonReconstructor::RadonReconstructor(const std::string& inputDir, const std::string& outputDir,
    const std::string& intermediateDir)
    : inputPath(inputDir), outputPath(outputDir), intermediatePath(intermediateDir),
    reconstructedImage(), imageWidth(0), imageHeight(0) {
    filterType = HannRamp;
    kernelRadius = 63;
    radialMaskFactor = 0.9f;
    edgeTaperPixels = 8;
}

void RadonReconstructor::setInputPath(const std::string& path) {
    inputPath = path;
    if (!inputPath.empty() && inputPath.back() != '/' && inputPath.back() != '\\') {
        inputPath += '/';
    }
}

void RadonReconstructor::setOutputPath(const std::string& path) {
    outputPath = path;
    if (!outputPath.empty() && outputPath.back() != '/' && outputPath.back() != '\\') {
        outputPath += '/';
    }
}

void RadonReconstructor::setIntermediatePath(const std::string& path) {
    intermediatePath = path;
    if (!intermediatePath.empty() && intermediatePath.back() != '/' && intermediatePath.back() != '\\') {
        intermediatePath += '/';
    }
}

bool RadonReconstructor::loadProcessedImage(const std::string& filename) {
    std::string fullPath = inputPath + filename;
    processedImage = cv::imread(fullPath, cv::IMREAD_GRAYSCALE);

    if (processedImage.empty()) {
        std::cerr << "Error: Failed to load processed image - " << fullPath << std::endl;
        return false;
    }

    imageWidth = processedImage.cols;
    imageHeight = processedImage.rows;

    std::cout << "Loaded image: " << filename << std::endl;
    std::cout << "  Size: " << imageWidth << "x" << imageHeight << std::endl;

    return true;
}

bool RadonReconstructor::saveMatTxt(const cv::Mat& mat, const std::string& filename) const {
    if (mat.empty()) return false;
    std::string fullPath = intermediatePath + filename;
    std::ofstream file(fullPath);
    if (!file.is_open()) return false;

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            file << mat.at<float>(i, j) << (j < mat.cols - 1 ? " " : "");
        }
        file << "\n";
    }
    return true;
}

cv::Mat RadonReconstructor::filterProjection(const cv::Mat& projection) const {
    int n = projection.rows;
    cv::Mat filtered = cv::Mat::zeros(n, 1, CV_32F);

    int kr = std::min(kernelRadius, std::min(255, n - 1));
    if (kr < 1) {
        projection.copyTo(filtered);
        return filtered;
    }
    std::vector<float> kernel(2 * kr + 1);
    for (int k = -kr; k <= kr; ++k) {
        float val;
        if (k == 0) {
            val = 0.25f;
        } else if (k % 2 != 0) {
            val = -1.0f / (CV_PI * CV_PI * k * k);
        } else {
            val = 0.0f;
        }
        float w = 1.0f;
        if (filterType == HannRamp) {
            float x = static_cast<float>(k) / static_cast<float>(kr);
            w = 0.5f * (1.0f + std::cos(CV_PI * x));
        } else if (filterType == SheppLogan) {
            if (k != 0) {
                float x = static_cast<float>(k) / static_cast<float>(kr + 1);
                float s = std::sin(CV_PI * x) / (CV_PI * x);
                w = s;
            } else {
                w = 1.0f;
            }
        }
        kernel[kr + k] = val * w;
    }
    cv::Mat kernelMat(2 * kr + 1, 1, CV_32F, kernel.data());
    cv::filter2D(projection, filtered, -1, kernelMat, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);

    return filtered;
}

#ifndef FFT_BACKPROJECTION_SWITCH
#define FFT_BACKPROJECTION_SWITCH 0
#endif

#if FFT_BACKPROJECTION_SWITCH

static int nextPowerOf2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static cv::Mat createRampFilterFFT(int n) {
    int nf = nextPowerOf2(2 * n);
    cv::Mat filter(nf, 1, CV_32F);
    
    for (int i = 0; i < nf; ++i) {
        int k = i - nf / 2;
        float val = 0.0f;
        if (k == 0) {
            val = 0.25f;
        } else if (k % 2 != 0) {
            val = -1.0f / (CV_PI * CV_PI * k * k);
        }
        float window = 1.0f;
        float x = static_cast<float>(k) / static_cast<float>(n);
        if (std::abs(x) < 1.0f) {
            window = 0.5f * (1.0f + std::cos(CV_PI * x));
        }
        filter.at<float>(i, 0) = val * window;
    }
    return filter;
}

cv::Mat RadonReconstructor::filterProjectionFFT(const cv::Mat& projection) const {
    int n = projection.rows;
    int nf = nextPowerOf2(2 * n);
    
    cv::Mat padded;
    cv::copyMakeBorder(projection, padded, 0, nf - n, 0, 0, cv::BORDER_CONSTANT, 0);
    
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat fftProj;
    cv::merge(planes, 2, fftProj);
    cv::dft(fftProj, fftProj, cv::DFT_COMPLEX_OUTPUT);
    
    cv::Mat filter = createRampFilterFFT(n);
    cv::Mat paddedFilter;
    cv::copyMakeBorder(filter, paddedFilter, 0, nf - filter.rows, 0, 0, cv::BORDER_CONSTANT, 0);
    
    cv::Mat planesF[] = { cv::Mat_<float>(paddedFilter), cv::Mat::zeros(paddedFilter.size(), CV_32F) };
    cv::Mat fftFilter;
    cv::merge(planesF, 2, fftFilter);
    cv::dft(fftFilter, fftFilter, cv::DFT_COMPLEX_OUTPUT);
    
    cv::Mat fftFiltered;
    cv::mulSpectrums(fftProj, fftFiltered, fftFiltered, 0);
    
    cv::dft(fftFiltered, fftFiltered, cv::DFT_INVERSE | cv::DFT_SCALE);
    
    cv::Mat result;
    cv::split(fftFiltered, planes);
    result = planes[0](cv::Rect(0, 0, n, 1));
    
    return result;
}

static cv::Mat fftBackprojectSlice(const cv::Mat& filteredProj, int size, int numAngles) {
    cv::Mat reconstruction = cv::Mat::zeros(size, size, CV_32F);
    float center = size / 2.0f;
    
    int nf = nextPowerOf2(size * 2);
    cv::Mat filter = createRampFilterFFT(size);
    cv::Mat paddedFilter;
    cv::copyMakeBorder(filter, paddedFilter, 0, nf - filter.rows, 0, 0, cv::BORDER_CONSTANT, 0);
    
    cv::Mat planesF[] = { cv::Mat_<float>(paddedFilter), cv::Mat::zeros(paddedFilter.size(), CV_32F) };
    cv::Mat fftFilter;
    cv::merge(planesF, 2, fftFilter);
    cv::dft(fftFilter, fftFilter, cv::DFT_COMPLEX_OUTPUT);
    
    for (int theta = 0; theta < numAngles; ++theta) {
        float angle = theta * CV_PI / 180.0f;
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        
        cv::Mat projLine(size, 1, CV_32F, cv::Scalar(0));
        for (int i = 0; i < size; ++i) {
            float t = static_cast<float>(i - static_cast<int>(center));
            int srcIdx = static_cast<int>(t * cosA + center);
            if (srcIdx >= 0 && srcIdx < filteredProj.rows) {
                projLine.at<float>(i, 0) = filteredProj.at<float>(srcIdx, 0);
            }
        }
        
        cv::Mat padded;
        cv::copyMakeBorder(projLine, padded, 0, nf - size, 0, 0, cv::BORDER_CONSTANT, 0);
        
        cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
        cv::Mat fftProj;
        cv::merge(planes, 2, fftProj);
        cv::dft(fftProj, fftProj, cv::DFT_COMPLEX_OUTPUT);
        
        cv::Mat fftFiltered;
        cv::mulSpectrums(fftProj, fftFilter, fftFiltered, 0);
        
        cv::dft(fftFiltered, fftFiltered, cv::DFT_INVERSE | cv::DFT_SCALE);
        
        cv::split(fftFiltered, planes);
        cv::Mat filteredLine = planes[0](cv::Rect(0, 0, size, 1));
        
        for (int y = 0; y < size; ++y) {
            float yCoord = y - center;
            for (int x = 0; x < size; ++x) {
                float xCoord = x - center;
                float s = xCoord * (-sinA) + yCoord * cosA;
                int idx = static_cast<int>(s + center);
                if (idx >= 0 && idx < size) {
                    reconstruction.at<float>(y, x) += filteredLine.at<float>(idx, 0);
                }
            }
        }
    }
    
    reconstruction *= (CV_PI / numAngles);
    return reconstruction;
}

#endif

cv::Mat RadonReconstructor::reconstructRadonFromColumn(int columnIndex, bool saveIntermediate, float smoothSigma) const {
    cv::Mat colData = processedImage.col(columnIndex);
    cv::Mat projFloat;
    colData.convertTo(projFloat, CV_32F);

    projFloat.row(0).setTo(0);
    projFloat.row(projFloat.rows - 1).setTo(0);

    double sumVal = 0.0;
    double sumValPos = 0.0;
    for (int r = 0; r < projFloat.rows; r++) {
        float val = projFloat.at<float>(r, 0);
        if (val < 0) val = 0;
        sumVal += val;
        sumValPos += val * r;
    }

    int shift = 0;
    if (sumVal > 1e-6) {
        double centerOfMass = sumValPos / sumVal;
        double geometricCenter = projFloat.rows / 2.0;
        shift = static_cast<int>(geometricCenter - centerOfMass);
    }

    cv::Mat centeredProj = cv::Mat::zeros(projFloat.size(), projFloat.type());
    for (int r = 0; r < projFloat.rows; r++) {
        int newR = r + shift;
        if (newR >= 0 && newR < projFloat.rows) {
            centeredProj.at<float>(newR, 0) = projFloat.at<float>(r, 0);
        }
    }
    projFloat = centeredProj;

    if (saveIntermediate) {
        cv::Mat norm;
        cv::normalize(projFloat, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(intermediatePath + "01_centroid_aligned_" + std::to_string(columnIndex) + ".png", norm);
    }

    {
        cv::Scalar m = cv::mean(projFloat);
        projFloat -= static_cast<float>(m[0]);
        int n = projFloat.rows;
        cv::Mat win(n, 1, CV_32F);
        for (int i = 0; i < n; ++i) {
            float x = static_cast<float>(i) / static_cast<float>(n - 1);
            float w = 0.5f * (1.0f - std::cos(2.0f * CV_PI * x));
            win.at<float>(i, 0) = w;
        }
        cv::multiply(projFloat, win, projFloat);
    }

    if (saveIntermediate) {
        cv::Mat norm;
        cv::normalize(projFloat, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(intermediatePath + "02_hann_window_" + std::to_string(columnIndex) + ".png", norm);
    }

    int kSize = static_cast<int>(smoothSigma * 3);
    if (kSize % 2 == 0) kSize++;
    cv::GaussianBlur(projFloat, projFloat, cv::Size(1, kSize), 0, smoothSigma);

    if (saveIntermediate) {
        cv::Mat norm;
        cv::normalize(projFloat, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(intermediatePath + "03_gaussian_smooth_" + std::to_string(columnIndex) + ".png", norm);
    }

#if FFT_BACKPROJECTION_SWITCH
    cv::Mat filteredProj = filterProjectionFFT(projFloat);
    
    int size = projFloat.rows;
    cv::Mat reconstruction = fftBackprojectSlice(filteredProj, size, 180);
#else
    cv::Mat filteredProj = filterProjection(projFloat);

    if (saveIntermediate) {
        cv::Mat norm;
        cv::normalize(filteredProj, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(intermediatePath + "04_ramp_filter_" + std::to_string(columnIndex) + ".png", norm);
    }

    int size = projFloat.rows;
    cv::Mat reconstruction = cv::Mat::zeros(size, size, CV_32F);
    float center = size / 2.0f;
    int numAngles = 180; // Number of projection angles

    // Pre-compute trig tables for speed
    std::vector<float> cosTable(numAngles);
    std::vector<float> sinTable(numAngles);
    for (int theta = 0; theta < numAngles; ++theta) {
        float angle = theta * CV_PI / 180.0f;
        cosTable[theta] = std::cos(angle);
        sinTable[theta] = std::sin(angle);
    }

    for (int theta = 0; theta < numAngles; ++theta) {
        float cosA = cosTable[theta];
        float sinA = sinTable[theta];

        for (int y = 0; y < size; ++y) {
            float yCoord = y - center;
            for (int x = 0; x < size; ++x) {
                float xCoord = x - center;

                float t = xCoord * cosA + yCoord * sinA;
                float tIdx = t + center;

                if (tIdx >= 0 && tIdx < size - 1) {
                    int i0 = static_cast<int>(tIdx);
                    float alpha = tIdx - i0;
                    float val = filteredProj.at<float>(i0, 0) * (1.0f - alpha) + 
                                filteredProj.at<float>(i0 + 1, 0) * alpha;
                    reconstruction.at<float>(y, x) += val;
                }
            }
        }
    }
    
    reconstruction *= (CV_PI / numAngles);

    if (saveIntermediate) {
        cv::Mat norm;
        cv::normalize(reconstruction, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(intermediatePath + "05_backprojection_" + std::to_string(columnIndex) + ".png", norm);
    }
#endif

    {
        float R = radialMaskFactor * (size * 0.5f);
        float taper = static_cast<float>(edgeTaperPixels);
        for (int y = 0; y < size; ++y) {
            float yCoord = y - center;
            for (int x = 0; x < size; ++x) {
                float xCoord = x - center;
                float r = std::sqrt(xCoord * xCoord + yCoord * yCoord);
                if (r >= R) {
                    reconstruction.at<float>(y, x) = 0.0f;
                } else if (r > R - taper && taper > 0.0f) {
                    float w = 0.5f * (1.0f + std::cos(CV_PI * (R - r) / taper));
                    reconstruction.at<float>(y, x) *= w;
                }
            }
        }
    }

    if (saveIntermediate) {
        cv::Mat norm;
        cv::normalize(reconstruction, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(intermediatePath + "06_radial_mask_" + std::to_string(columnIndex) + ".png", norm);
    }

    if (saveIntermediate) {
        cv::Mat norm;
        cv::normalize(reconstruction, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(intermediatePath + "radon_slice_" + std::to_string(columnIndex) + ".png", norm);
    }

    return reconstruction;
}

// New function: Reconstruct single column (2D distribution)
bool RadonReconstructor::reconstructSingleColumn(const std::string& inputImage, int columnIndex, float smoothSigma) {
    if (!loadProcessedImage(inputImage)) return false;

    if (columnIndex < 0 || columnIndex >= processedImage.cols) {
        std::cerr << "Error: Column index " << columnIndex << " is out of range." << std::endl;
        return false;
    }

    std::cout << "Reconstructing single column " << columnIndex << "..." << std::endl;

    // Use saveIntermediate=true to save the result image
    reconstructedImage = reconstructRadonFromColumn(columnIndex, true, smoothSigma);

    if (reconstructedImage.empty()) {
        std::cerr << "Error: Reconstruction returned empty image." << std::endl;
        return false;
    }

    return true;
}

bool RadonReconstructor::perform3DReconstruction(const std::string& inputImage, bool saveIntermediate, float smoothSigma) {
    if (!loadProcessedImage(inputImage)) return false;

    int numColumns = processedImage.cols;
    std::vector<Point3D> points;
    float globalMaxIntensity = 0.0f;

    std::cout << "Starting Radon reconstruction..." << std::endl;
    std::cout << "Assumption: Image X-axis is the axis of rotation (Plasma ejection direction)." << std::endl;
    std::cout << "            Image columns are projections of Y-Z slices." << std::endl;

    for (int col = 0; col < numColumns; ++col) {
        if (col % 50 == 0) std::cout << "Processing column " << col << "/" << numColumns << std::endl;

        bool saveSlice = saveIntermediate && (col == numColumns / 2);
        cv::Mat slice2D = reconstructRadonFromColumn(col, saveSlice, smoothSigma);
        
        if (col == numColumns / 2) {
            reconstructedImage = slice2D.clone();
        }

        double minSliceVal, maxSliceVal;
        cv::minMaxLoc(slice2D, &minSliceVal, &maxSliceVal);
        if (maxSliceVal <= 0.0) continue;

        cv::Mat sliceNorm8U;
        cv::normalize(slice2D, sliceNorm8U, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::Mat bin;
        double otsu = cv::threshold(sliceNorm8U, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        if (saveSlice) {
            cv::imwrite(intermediatePath + "07_otsu_threshold_col" + std::to_string(numColumns / 2) + ".png", bin);
        }

        cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(bin, bin, cv::MORPH_OPEN, se);

        if (saveSlice) {
            cv::imwrite(intermediatePath + "08_morphology_open_col" + std::to_string(numColumns / 2) + ".png", bin);
        }

        int sz = slice2D.rows;
        float centerRC = sz * 0.5f;
        float pcR = radialMaskFactor * centerRC;
        for (int r = 0; r < sz; ++r) {
            float y = r - centerRC;
            for (int c = 0; c < sz; ++c) {
                float x = c - centerRC;
                float rad = std::sqrt(x * x + y * y);
                if (rad >= pcR) {
                    bin.at<uchar>(r, c) = 0;
                }
            }
        }

        if (saveSlice) {
            cv::imwrite(intermediatePath + "09_radial_crop_col" + std::to_string(numColumns / 2) + ".png", bin);
        }
        // Convert to 3D points
        // Coordinate System Mapping:
        // Global X: Ejection direction (Image Column Index)
        // Global Y: Cross-section horizontal (Reconstruction x)
        // Global Z: Cross-section vertical (Reconstruction y) -> This corresponds to original image rows
        
        int centerUV = slice2D.rows / 2;
        float globalX = static_cast<float>(col); // Axis of rotation

        for (int r = 0; r < slice2D.rows; ++r) {     // slice row index (local v)
            for (int c = 0; c < slice2D.cols; ++c) { // slice col index (local u)
                if (bin.at<uchar>(r, c) == 0) continue;
                float val = slice2D.at<float>(r, c);

                float localU = static_cast<float>(c - centerUV);
                float localV = static_cast<float>(r - centerUV);

                // Point3D {x, y, z, intensity}
                // Mapping:
                // x -> globalX (Image Column)
                // y -> localU
                // z -> localV
                Point3D p = {globalX, localU, localV, val};
                
                points.push_back(p);
                if (val > globalMaxIntensity) globalMaxIntensity = val;
            }
        }
    }


    if (points.empty()) return false;

    // Save PLY
    std::string plyPath = outputPath + "radon_reconstruction_points.ply";
    std::ofstream ofs(plyPath);
    if (ofs.is_open()) {
        ofs << "ply\nformat ascii 1.0\nelement vertex " << points.size() << "\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
        
        float minIntensity = points[0].intensity;
        for (const auto& p : points) minIntensity = std::min(minIntensity, p.intensity);
        float intensityRange = globalMaxIntensity - minIntensity;

        for (const auto& p : points) {
            float normalized = (intensityRange > 0) ? (p.intensity - minIntensity) / intensityRange : 0.5f;
            normalized = std::max(0.0f, std::min(1.0f, normalized));
            int gray = static_cast<int>(normalized * 255.0f);
            ofs << p.x << " " << p.y << " " << p.z << " " << p.intensity << " " << gray << " " << gray << " " << gray << "\n";
        }
    }
    
    generateSideView(points, outputPath);
    visualizePointCloudRealtimePCL(points, "TDR Plasma 3D Point Cloud (Radon)");
    
    return true;
}

bool RadonReconstructor::saveResults(const std::string& baseName) {
    if (reconstructedImage.empty()) return false;
    return saveMatTxt(reconstructedImage, outputPath + baseName + "_reconstruction.txt");
}

void RadonReconstructor::generateSideView(const std::vector<Point3D>& points, const std::string& outputPath) const {
    if (points.empty()) return;
    
    float minIntensity = points[0].intensity, maxIntensity = points[0].intensity;
    for (const auto& p : points) {
        minIntensity = std::min(minIntensity, p.intensity);
        maxIntensity = std::max(maxIntensity, p.intensity);
    }
    
    const float theoreticalRadius = 256.0f;
    const float theoreticalZMin = 0.0f;
    const float theoreticalZMax = 511.0f;
    float minX = -theoreticalRadius, maxX = theoreticalRadius;
    float minZ = theoreticalZMin, maxZ = theoreticalZMax;
    
    const int imgSize = 512;
    // Create XZ projection image (side view only)
    cv::Mat xzView(imgSize, imgSize, CV_8UC3, cv::Scalar(0, 0, 0)); // XZ plane (side view)

    float intensityRange = maxIntensity - minIntensity;
    if (intensityRange <= 0.0f) intensityRange = 1.0f;

    // Project points to XZ view only
    for (const auto& p : points) {
        // Normalize intensity for grayscale mapping
        float normalized = (p.intensity - minIntensity) / intensityRange;
        normalized = std::max(0.0f, std::min(1.0f, normalized));

        // Grayscale mapping: low intensity = dark, high intensity = bright
        int grayValue = static_cast<int>(normalized * 255.0f);
        grayValue = std::max(0, std::min(255, grayValue));

        cv::Vec3b color(grayValue, grayValue, grayValue); // RGB all same for grayscale

        // XZ projection (side view: x, z)
        {
            // Note: p.x is global X (ejection axis), p.z is global Z (vertical in cross-section)
            // We want to map global X to image X (horizontal), and global Z to image Y (vertical)
            
            // Map p.x (0 to numColumns) to image X (0 to imgSize)
            // minX/maxX here were set to theoreticalRadius (-256, 256), but p.x is actually positive (0 to 512).
            // Let's adjust the mapping logic for X.
            
            float normX = p.x / theoreticalZMax; // Assuming p.x goes from 0 to 511
            int x_pixel = static_cast<int>(normX * (imgSize - 1));
            
            // Map p.z (-256 to 256) to image Y (0 to imgSize)
            // In image coordinates, Y is top-down. We want +Z to be up.
            // So pixel Y = (maxZ - z) / range * height
            
            // Re-use minZ/maxZ variables but interpret them as vertical range (-256, 256)
            float vMin = -theoreticalRadius;
            float vMax = theoreticalRadius;
            
            int z_pixel = static_cast<int>((vMax - p.z) / (vMax - vMin) * (imgSize - 1));

            if (x_pixel >= 0 && x_pixel < imgSize && z_pixel >= 0 && z_pixel < imgSize) {
                xzView.at<cv::Vec3b>(z_pixel, x_pixel) = color;
            }
        }
    }
    
    cv::imwrite(outputPath + "point_cloud_side_view.png", xzView);
    std::cout << "Side view saved: " << outputPath + "point_cloud_side_view.png" << std::endl;
}

void RadonReconstructor::setFilterType(int type) {
    if (type == 0) filterType = RamLak;
    else if (type == 1) filterType = HannRamp;
    else if (type == 2) filterType = SheppLogan;
}

void RadonReconstructor::setKernelRadius(int radius) {
    if (radius > 0) kernelRadius = radius;
}

void RadonReconstructor::setRadialMaskFactor(float factor) {
    if (factor > 0.0f && factor <= 1.0f) radialMaskFactor = factor;
}

void RadonReconstructor::setEdgeTaperPixels(int pixels) {
    if (pixels >= 0) edgeTaperPixels = pixels;
}

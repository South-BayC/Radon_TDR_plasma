#include "radon_reconstruction.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <direct.h>

// File-local helper: save matrix as text file
static bool saveMatTxt(const cv::Mat& mat, const std::string& filePath) {
    if (mat.empty()) return false;
    std::ofstream file(filePath);
    if (!file.is_open()) return false;
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            file << mat.at<float>(i, j) << (j < mat.cols - 1 ? " " : "");
        }
        file << "\n";
    }
    return true;
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

// FFT_BACKPROJECTION_SWITCH is defined in include/config.h
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

static cv::Mat filterProjectionFFT(const cv::Mat& projection) {
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
    cv::mulSpectrums(fftProj, fftFilter, fftFiltered, 0);
    
    cv::dft(fftFiltered, fftFiltered, cv::DFT_INVERSE | cv::DFT_SCALE);
    
    cv::Mat result;
    cv::split(fftFiltered, planes);
    result = planes[0](cv::Rect(0, 0, 1, n)).clone();
    
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
        cv::Mat filteredLine = planes[0](cv::Rect(0, 0, 1, size)).clone();
        
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
    float center = size / 2.0f;
    cv::Mat reconstruction = fftBackprojectSlice(filteredProj, size, NUM_PROJECTION_ANGLES);
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
    int numAngles = NUM_PROJECTION_ANGLES; // Number of projection angles (from config.h)

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

    // ROI auto-cropping: scan image rows to find the plasma signal range,
    // crop away pure-background rows to reduce reconstruction artifacts.
    const int signalThreshold = 10; // pixel values <= this are considered background
    int rowMin = 0, rowMax = processedImage.rows - 1;
    for (int r = 0; r < processedImage.rows; ++r) {
        const uchar* rowPtr = processedImage.ptr<uchar>(r);
        bool hasSignal = false;
        for (int c = 0; c < processedImage.cols; ++c) {
            if (rowPtr[c] > signalThreshold) { hasSignal = true; break; }
        }
        if (hasSignal) { rowMin = r; break; }
    }
    for (int r = processedImage.rows - 1; r >= 0; --r) {
        const uchar* rowPtr = processedImage.ptr<uchar>(r);
        bool hasSignal = false;
        for (int c = 0; c < processedImage.cols; ++c) {
            if (rowPtr[c] > signalThreshold) { hasSignal = true; break; }
        }
        if (hasSignal) { rowMax = r; break; }
    }
    // Add small padding to avoid cutting signal edges
    rowMin = std::max(0, rowMin - 8);
    rowMax = std::min(processedImage.rows - 1, rowMax + 8);
    int cropOffset = rowMin;
    cv::Rect roi(0, rowMin, processedImage.cols, rowMax - rowMin + 1);
    if (roi.height < processedImage.rows) {
        processedImage = processedImage(roi).clone();
        imageHeight = processedImage.rows;
        std::cout << "  ROI cropped: rows [" << rowMin << ", " << rowMax
                  << "], new size: " << imageWidth << "x" << imageHeight << std::endl;
    } else {
        cropOffset = 0;
    }

    int numColumns = processedImage.cols;
    std::vector<Point3D> points;

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
                Point3D p = {globalX, localU, localV + static_cast<float>(cropOffset), val};
                points.push_back(p);
            }
        }
    }


    if (points.empty()) return false;

    // Export results via PointCloudIO
    PointCloudIO::savePLY(points, outputPath + "radon_reconstruction_points.ply");
    PointCloudIO::generateSideView(points, outputPath);

#if ENABLE_VISUALIZATION
    std::cout << "[Visualization] Opening PCL 3D viewer..." << std::endl;
    std::cout << "[Visualization] NOTE: If your desktop icons rearrange after closing the viewer," << std::endl;
    std::cout << "[Visualization] set ENABLE_VISUALIZATION=0 in include/config.h and use an external" << std::endl;
    std::cout << "[Visualization] viewer (CloudCompare, MeshLab) to inspect the PLY file." << std::endl;
    PointCloudIO::visualizePCL(points, "TDR Plasma 3D Point Cloud (Radon)");
#else
    std::cout << "[Visualization] PCL viewer is disabled (ENABLE_VISUALIZATION=0)." << std::endl;
    std::cout << "[Visualization] Point cloud data saved to PLY file. Use an external viewer (e.g., CloudCompare, MeshLab) to inspect." << std::endl;
#endif
    
    return true;
}

bool RadonReconstructor::saveResults(const std::string& baseName) {
    if (reconstructedImage.empty()) return false;
    return saveMatTxt(reconstructedImage, outputPath + baseName + "_reconstruction.txt");
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

#include "preprocess.h"
#include <iostream>
#include <windows.h>  // Windows API
#include <algorithm>
#include <direct.h>   // for _mkdir
#include <fstream>

// Use default values if macros are not defined
#ifndef TARGET_IMAGE_SIZE
#define TARGET_IMAGE_SIZE 512
#endif

#ifndef MEDIAN_FILTER_KERNEL_SIZE
#define MEDIAN_FILTER_KERNEL_SIZE 3
#endif
#ifndef MASK_THRESHOLD
#define MASK_THRESHOLD 0.1
#endif

// New feature parameters (from main.cpp)
#ifndef ENABLE_MEAN_FILTERING
#define ENABLE_MEAN_FILTERING true
#endif

#ifndef MEAN_FILTER_KERNEL_SIZE
#define MEAN_FILTER_KERNEL_SIZE 8
#endif


// Check if directory exists
bool directoryExists(const std::string& path) {
	DWORD dwAttrib = GetFileAttributesA(path.c_str());
	return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
		(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

// Create directory if it doesn't exist
bool createDirectory(const std::string& path) {
	return _mkdir(path.c_str()) == 0;
}

// Get file extension
std::string getFileExtension(const std::string& filename) {
	size_t dotPos = filename.find_last_of(".");
	if (dotPos != std::string::npos) {
		return filename.substr(dotPos + 1);
	}
	return "";
}

// Convert string to lower case
std::string toLower(const std::string& str) {
	std::string result = str;
	std::transform(result.begin(), result.end(), result.begin(), ::tolower);
	return result;
}

// Constructor
ImagePreprocessor::ImagePreprocessor()
	: targetSize(TARGET_IMAGE_SIZE) {
	// Set default paths
	inputPath = "data/input/";
	outputPath = "data/proceed/";

	// Create output directory
	createDirectory(outputPath);
}

ImagePreprocessor::ImagePreprocessor(const std::string& inputDir, const std::string& outputDir)
	: inputPath(inputDir), outputPath(outputDir), targetSize(TARGET_IMAGE_SIZE) {

	// Ensure path ends with slash
	if (!inputPath.empty() && inputPath.back() != '\\' && inputPath.back() != '/') {
		inputPath += '\\';
	}
	if (!outputPath.empty() && outputPath.back() != '\\' && outputPath.back() != '/') {
		outputPath += '\\';
	}

	// Create output directory
	createDirectory(outputPath);
}

// Path settings
void ImagePreprocessor::setInputPath(const std::string& path) {
	inputPath = path;
	if (!inputPath.empty() && inputPath.back() != '\\' && inputPath.back() != '/') {
		inputPath += '\\';
	}
}

void ImagePreprocessor::setOutputPath(const std::string& path) {
	outputPath = path;
	if (!outputPath.empty() && outputPath.back() != '\\' && outputPath.back() != '/') {
		outputPath += '\\';
	}
	createDirectory(outputPath);
}

// Resize and pad image
cv::Mat ImagePreprocessor::resizeAndPad(const cv::Mat& input) {
	cv::Mat output;

	if (input.rows == targetSize && input.cols == targetSize) {
		return input.clone();
	}

	output = cv::Mat::zeros(targetSize, targetSize, input.type());

	// Calculate center position
	int x = (targetSize - input.cols) / 2;
	int y = (targetSize - input.rows) / 2;

	x = std::max(0, x);
	y = std::max(0, y);

	int width = std::min(input.cols, targetSize - x);
	int height = std::min(input.rows, targetSize - y);

	if (width <= 0 || height <= 0) {
		std::cerr << "Error: Cannot adapt image to target size." << std::endl;
		return cv::Mat();
	}

	cv::Mat roi = output(cv::Rect(x, y, width, height));
	cv::Mat inputRoi = input(cv::Rect(0, 0, width, height));
	inputRoi.copyTo(roi);

	return output;
}

// Apply median filter
cv::Mat ImagePreprocessor::applyMedianFilter(const cv::Mat& input, int kernelSize) {
	cv::Mat filtered;

	if (kernelSize % 2 == 0) {
		kernelSize++;
	}

	cv::medianBlur(input, filtered, kernelSize);
	return filtered;
}

// Create mask (binarization)
cv::Mat ImagePreprocessor::createMask(const cv::Mat& input, double threshold) {
	if (input.empty()) {
		return cv::Mat();
	}

	cv::Mat gray;
	if (input.channels() == 3) {
		cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
	}
	else {
		gray = input.clone();
	}

	cv::Mat normalized;
	gray.convertTo(normalized, CV_32F, 1.0 / 255.0);

	cv::Mat mask;
	cv::threshold(normalized, mask, threshold, 1.0, cv::THRESH_BINARY);

	cv::Mat mask8u;
	mask.convertTo(mask8u, CV_8U, 255);

	return mask8u;
}

// Process image
bool ImagePreprocessor::processImage(const std::string& filename, bool saveIntermediate) {
	std::string fullInputPath = inputPath + filename;

	DWORD dwAttrib = GetFileAttributesA(fullInputPath.c_str());
	if (dwAttrib == INVALID_FILE_ATTRIBUTES || (dwAttrib & FILE_ATTRIBUTE_DIRECTORY)) {
		std::cerr << "Error: File not found - " << fullInputPath << std::endl;
		return false;
	}

	cv::Mat image = cv::imread(fullInputPath, cv::IMREAD_GRAYSCALE);
	if (image.empty()) {
		std::cerr << "Error: Failed to load image - " << fullInputPath << std::endl;
		return false;
	}

	std::cout << "Processing: " << filename << std::endl;
	std::cout << "  Original size: " << image.cols << "x" << image.rows << std::endl;

	// NOTE: Skip resizing/padding step. Use original image size directly.
	// If needed in the future, resizeAndPad(image) can be re-enabled here.

	// Step 1: Median filter - used to generate mask (preserve edges, remove salt-pepper noise)
	cv::Mat filtered = applyMedianFilter(image, MEDIAN_FILTER_KERNEL_SIZE);

	if (saveIntermediate) {
		std::string filteredPath = outputPath + "filtered_" + filename;
		cv::imwrite(filteredPath, filtered);
		std::cout << "  Saved filtered image (for mask): " << filteredPath << std::endl;
	}

	// Step 2: Mean filter on original image - used to smooth data
	cv::Mat meanFiltered = image.clone();
	if (ENABLE_MEAN_FILTERING) {
		int kernelSize = MEAN_FILTER_KERNEL_SIZE;
		if (kernelSize % 2 == 0) kernelSize++;
		cv::blur(image, meanFiltered, cv::Size(kernelSize, kernelSize));

		std::string meanFilteredPath = outputPath + "mean_filtered_" + filename;
		cv::imwrite(meanFilteredPath, meanFiltered);
		std::cout << "  Saved mean filtered image: " << meanFilteredPath << std::endl;
	}

	// Step 3: Create mask based on median filtered image
	cv::Mat mask = createMask(filtered, MASK_THRESHOLD);

	if (!mask.empty() && saveIntermediate) {
		std::string maskPath = outputPath + "mask_" + filename;
		cv::imwrite(maskPath, mask);
		std::cout << "  Saved mask: " << maskPath << std::endl;
	}

	// Step 4: Apply mask
	cv::Mat finalImage;
	if (!mask.empty()) {
		cv::Mat maskFloat;
		mask.convertTo(maskFloat, meanFiltered.type(), 1.0 / 255.0);
		cv::multiply(meanFiltered, maskFloat, finalImage);
	}
	else {
		finalImage = meanFiltered.clone();
	}

	// Save final result
	std::string finalPath = outputPath + "processed_" + filename;
	cv::imwrite(finalPath, finalImage);
	std::cout << "  Saved processed image: " << finalPath << std::endl;

	std::cout << "  Processing completed successfully." << std::endl;
	return true;
}

// Process all images in directory
bool ImagePreprocessor::processAllImages(bool saveIntermediate) {
	if (!directoryExists(inputPath)) {
		std::cerr << "Error: Input directory not found - " << inputPath << std::endl;
		return false;
	}

	std::cout << "Processing all images in: " << inputPath << std::endl;

	bool allSuccess = true;
	int processedCount = 0;

	std::string searchPath = inputPath + "*.*";
	WIN32_FIND_DATAA findFileData;
	HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findFileData);

	if (hFind == INVALID_HANDLE_VALUE) {
		std::cerr << "Error: Cannot open directory - " << inputPath << std::endl;
		return false;
	}

	do {
		if (findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			continue;
		}

		std::string filename = findFileData.cFileName;
		std::string ext = getFileExtension(filename);
		std::string lowerExt = toLower(ext);

		if (lowerExt == "png" || lowerExt == "jpg" || lowerExt == "jpeg" ||
			lowerExt == "bmp" || lowerExt == "tif" || lowerExt == "tiff") {

			if (processImage(filename, saveIntermediate)) {
				processedCount++;
			}
			else {
				allSuccess = false;
				std::cerr << "Processing failed: " << filename << std::endl;
			}
		}
	} while (FindNextFileA(hFind, &findFileData) != 0);

	FindClose(hFind);

	std::cout << "Total processed: " << processedCount << " images" << std::endl;
	return allSuccess;
}

// Set target size
void ImagePreprocessor::setTargetSize(int size) {
	if (size > 0) {
		targetSize = size;
	}
	else {
		std::cerr << "Warning: Invalid target size. Using default " << TARGET_IMAGE_SIZE << "." << std::endl;
		targetSize = TARGET_IMAGE_SIZE;
	}
}

// Placeholder functions
cv::Mat ImagePreprocessor::getProcessedImage() const {
	return cv::Mat();
}

cv::Mat ImagePreprocessor::getMask() const {
	return cv::Mat();
}
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class ImagePreprocessor {
private:
	// Private members
	std::string inputPath;
	std::string outputPath;
	int targetSize;

public:
	// Constructors
	ImagePreprocessor();
	ImagePreprocessor(const std::string& inputDir, const std::string& outputDir);

	// Set paths
	void setInputPath(const std::string& path);
	void setOutputPath(const std::string& path);

	// Core processing
	bool processImage(const std::string& filename, bool saveIntermediate = true);
	bool processAllImages(bool saveIntermediate = true);

	// Helper functions
	cv::Mat resizeAndPad(const cv::Mat& input);
	cv::Mat applyMedianFilter(const cv::Mat& input, int kernelSize);
	cv::Mat createMask(const cv::Mat& input, double threshold);

	// Getters
	cv::Mat getProcessedImage() const;
	cv::Mat getMask() const;

	// Setters
	void setTargetSize(int size);
};

#endif

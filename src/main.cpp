#include <iostream>
#include "preprocess.h"

// FFT-based backprojection toggle (1 = use FFT acceleration, 0 = use original algorithm)
// This MUST be defined BEFORE including radon_reconstruction.h
#define FFT_BACKPROJECTION_SWITCH    1

#include "radon_reconstruction.h"

// I/O path configuration
#define INPUT_IMAGE_NAME          "9KV.png"              // Input image name (in data/input/)
#define INPUT_PATH                "../data/input/"          // Input path
#define PROCESSED_PATH            "../data/proceed/"        // Preprocessed results path
#define OUTPUT_PATH               "../data/output/"         // Final output path

// Preprocessing parameters
#define TARGET_IMAGE_SIZE         512                    // Target image size
#define MEDIAN_FILTER_KERNEL_SIZE 3                      // Median filter kernel size (must be odd)
#define MASK_THRESHOLD            0.1                    // Mask threshold (0.0-1.0)
#define ENABLE_MEAN_FILTERING     true                   // Enable mean filtering
#define MEAN_FILTER_KERNEL_SIZE   8                      // Mean filter kernel size

// Reconstruction parameters
#define POINT_CLOUD_THRESHOLD_RATIO 0.01f                // Point cloud threshold ratio
#define SAVE_INTERMEDIATE_RESULTS   true                 // Save intermediate results
#define ENABLE_POINT_CLOUD_GENERATION true              // Toggle for 3D point cloud generation
#define TARGET_COLUMN_INDEX         256                  // Target column for 2D reconstruction (default: 256)
#define RADIAL_SMOOTH_SIGMA         12.0f                // Radial Gaussian smoothing (higher = smoother radial profile)

// Reconstruction filter settings (tunable)
// RECON_FILTER_TYPE: 0=RamLak (ideal ramp), 1=HannRamp (windowed ramp, recommended), 2=SheppLogan (sinc-weighted ramp)
#define RECON_FILTER_TYPE           1
// RECON_KERNEL_RADIUS: half-width of the convolution kernel in samples (effective taps = 2*radius+1). Recommended 31–63.
#define RECON_KERNEL_RADIUS         63
// RECON_RADIAL_MASK_FACTOR: relative radius (0.0–1.0) of the circular mask applied to each reconstructed slice. Typical 0.85–0.95.
#define RECON_RADIAL_MASK_FACTOR    0.92f
// RECON_EDGE_TAPER_PIXELS: cosine taper width at the slice mask border, in pixels. Typical 5–12.
#define RECON_EDGE_TAPER_PIXELS     8

int main() {
    std::cout << "Plasma 3D Reconstruction - Radon Transform Version" << std::endl;
    std::cout << "=====================================================" << std::endl;

    try {
        std::string originalImage = INPUT_IMAGE_NAME;
        std::string processedImage = "processed_" + originalImage;

        std::cout << "\n[Step 1] Image Preprocessing" << std::endl;
        std::cout << "------------------------------------------" << std::endl;

        ImagePreprocessor preprocessor;
        preprocessor.setInputPath(INPUT_PATH);
        preprocessor.setOutputPath(PROCESSED_PATH);
        preprocessor.setTargetSize(TARGET_IMAGE_SIZE);

        std::cout << "Processing original image: " << originalImage << std::endl;
        
        // Ensure paths end with separator
        std::string inputPathStr = INPUT_PATH;
        if (inputPathStr.back() != '/' && inputPathStr.back() != '\\') inputPathStr += "/";
        
        // Check if input file exists
        std::string fullInputPath = inputPathStr + originalImage;
        FILE* f = fopen(fullInputPath.c_str(), "rb");
        if (!f) {
             std::cerr << "Error: Input file not found at " << fullInputPath << std::endl;
             std::cerr << "Current working directory is likely incorrect." << std::endl;
             return 1;
        }
        fclose(f);

        if (!preprocessor.processImage(originalImage, SAVE_INTERMEDIATE_RESULTS)) {
            std::cerr << "Preprocessing failed, exiting." << std::endl;
            return 1;
        }

        std::cout << "Preprocessing completed successfully." << std::endl;
        std::cout << "Processed image saved as: " << processedImage << std::endl;

        std::cout << "\n[Step 2] Radon 3D Reconstruction" << std::endl;
        std::cout << "------------------------------------------" << std::endl;

        RadonReconstructor reconstructor;
        reconstructor.setInputPath(PROCESSED_PATH);
        reconstructor.setOutputPath(OUTPUT_PATH);
        reconstructor.setIntermediatePath(PROCESSED_PATH);
        // Apply reconstruction filter parameters from macros
        reconstructor.setFilterType(RECON_FILTER_TYPE);
        reconstructor.setKernelRadius(RECON_KERNEL_RADIUS);
        reconstructor.setRadialMaskFactor(RECON_RADIAL_MASK_FACTOR);
        reconstructor.setEdgeTaperPixels(RECON_EDGE_TAPER_PIXELS);

        std::cout << "Loading processed image: " << processedImage << std::endl;

        if (ENABLE_POINT_CLOUD_GENERATION) {
            std::cout << "\n[Full Reconstruction] Starting full 3D reconstruction..." << std::endl;
            if (reconstructor.perform3DReconstruction(processedImage, SAVE_INTERMEDIATE_RESULTS, RADIAL_SMOOTH_SIGMA)) {
                std::cout << "3D Reconstruction completed successfully." << std::endl;
                
                std::cout << "\n[Saving Results]" << std::endl;
                reconstructor.saveResults("result");
                std::cout << "Results saved to: " << OUTPUT_PATH << std::endl;
            } else {
                std::cerr << "3D Reconstruction failed." << std::endl;
                return 1;
            }
        } else {
            std::cout << "\n[Single Column Reconstruction] Processing column " << TARGET_COLUMN_INDEX << "..." << std::endl;
            if (reconstructor.reconstructSingleColumn(processedImage, TARGET_COLUMN_INDEX, RADIAL_SMOOTH_SIGMA)) {
                std::cout << "Single column reconstruction completed successfully." << std::endl;
                std::cout << "Result saved to: " << PROCESSED_PATH << "radon_slice_" << TARGET_COLUMN_INDEX << ".png" << std::endl;
            } else {
                std::cerr << "Single column reconstruction failed." << std::endl;
                return 1;
            }
        }

        std::cout << "\n[Complete] Processing finished successfully!" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();

    return 0;
}
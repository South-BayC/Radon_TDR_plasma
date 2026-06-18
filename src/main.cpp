#include "config.h"
#include <iostream>
#include "preprocess.h"
#include "radon_reconstruction.h"

static void pauseOnError() {
    std::cerr << "\nPress Enter to exit..." << std::endl;
    std::cin.get();
}

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
             pauseOnError();
             return 1;
        }
        fclose(f);

        if (!preprocessor.processImage(originalImage, SAVE_INTERMEDIATE_RESULTS)) {
            std::cerr << "Preprocessing failed, exiting." << std::endl;
            pauseOnError();
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
        // Apply reconstruction filter parameters from config
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
                pauseOnError();
                return 1;
            }
        } else {
            std::cout << "\n[Single Column Reconstruction] Processing column " << TARGET_COLUMN_INDEX << "..." << std::endl;
            if (reconstructor.reconstructSingleColumn(processedImage, TARGET_COLUMN_INDEX, RADIAL_SMOOTH_SIGMA)) {
                std::cout << "Single column reconstruction completed successfully." << std::endl;
                std::cout << "Result saved to: " << PROCESSED_PATH << "radon_slice_" << TARGET_COLUMN_INDEX << ".png" << std::endl;
            } else {
                std::cerr << "Single column reconstruction failed." << std::endl;
                pauseOnError();
                return 1;
            }
        }

        std::cout << "\n[Complete] Processing finished successfully!" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        pauseOnError();
        return 1;
    }

    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();

    return 0;
}

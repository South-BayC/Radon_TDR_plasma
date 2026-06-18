#ifndef CONFIG_H
#define CONFIG_H

// ============================================================
// Radon_TDR_plasma - Global Configuration
// ============================================================
// All tunable parameters are centralized in this single header.
// Modify values here to adjust behavior without touching any
// source code.
// ============================================================

// --------------------------------------------------
// FFT Backprojection Toggle
//   1 = Use FFT-accelerated backprojection (recommended)
//   0 = Use original spatial-domain backprojection
// NOTE: Must be a macro (used in preprocessor #if directives)
// --------------------------------------------------
// NOTE: Currently using non-FFT (spatial-domain) backprojection (FFT_BACKPROJECTION_SWITCH=0).
// The FFT path (FFT_BACKPROJECTION_SWITCH=1) has a known double-filtering bug in
// fftBackprojectSlice() — it re-filters an already-filtered projection. Set to 1 only
// after fixing that function to skip the redundant ramp filter.
#define FFT_BACKPROJECTION_SWITCH    0

// --------------------------------------------------
// I/O Paths
// --------------------------------------------------
#define INPUT_IMAGE_NAME          "9KV.png"
#define INPUT_PATH                "../data/input/"
#define PROCESSED_PATH            "../data/proceed/"
#define OUTPUT_PATH               "../data/output/"

// --------------------------------------------------
// Preprocessing Parameters
// --------------------------------------------------
#define TARGET_IMAGE_SIZE         512
#define MEDIAN_FILTER_KERNEL_SIZE 3
#define MASK_THRESHOLD            0.1
#define ENABLE_MEAN_FILTERING     true
#define MEAN_FILTER_KERNEL_SIZE   8

// --------------------------------------------------
// Reconstruction Parameters
// --------------------------------------------------
#define POINT_CLOUD_THRESHOLD_RATIO 0.01f
#define SAVE_INTERMEDIATE_RESULTS   true
#define ENABLE_POINT_CLOUD_GENERATION true
#define TARGET_COLUMN_INDEX         256
#define RADIAL_SMOOTH_SIGMA         12.0f
#define NUM_PROJECTION_ANGLES       180

// --------------------------------------------------
// Visualization Settings
// --------------------------------------------------
// ENABLE_VISUALIZATION: Set to 1 to open the PCL 3D viewer window.
//    Set to 0 to skip the interactive viewer and only save results to disk.
//    NOTE: PCL/VTK OpenGL window creation on some Windows systems can trigger
//    a desktop refresh that rearranges desktop icons. Disable this option
//    if you experience that issue. The point cloud data is always saved
//    to the PLY file regardless of this setting.
#define ENABLE_VISUALIZATION        1

// --------------------------------------------------
// Reconstruction Filter Settings
// --------------------------------------------------
// RECON_FILTER_TYPE: 0=RamLak, 1=HannRamp, 2=SheppLogan
#define RECON_FILTER_TYPE           1
#define RECON_KERNEL_RADIUS         63
#define RECON_RADIAL_MASK_FACTOR    0.92f
#define RECON_EDGE_TAPER_PIXELS     8

#endif // CONFIG_H

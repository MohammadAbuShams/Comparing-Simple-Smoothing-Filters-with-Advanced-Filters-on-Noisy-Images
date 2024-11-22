# Comparing Simple Smoothing Filters with Advanced Filters on Noisy Images


## Overview

This project focuses on the task of image denoising, which aims to remove unwanted noise from images while preserving important details such as edges. The project tests multiple filters and compares their performance under various conditions of noise (Gaussian and Salt-and-Pepper) with different kernel sizes.

## Table of Contents

- [Project Summary](#project-summary)
- [Filters Implemented](#filters-implemented)
- [Noise Types](#noise-types)
- [Experimental Setup](#experimental-setup)
- [Performance Metrics](#performance-metrics)
- [Results](#results)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

---

## Project Summary

In this assignment, the following denoising filters were tested:
- **Box Filter**: A simple filter that averages pixel values over a small neighborhood.
- **Gaussian Filter**: Uses a Gaussian kernel to smooth the image while preserving edges.
- **Median Filter**: Replaces each pixel with the median of its neighboring pixels, effective for Salt-and-Pepper noise.
- **Bilateral Filter**: Uses both spatial and intensity information to preserve edges while reducing noise.
- **Adaptive Filters**: Adjust the kernel size dynamically based on local image statistics, including Adaptive Mean and Adaptive Median filters.

The experiment involved testing these filters on three images of animals (cat, dog, bird) under different noise levels and kernel sizes. Various performance metrics were used to evaluate the filters.

---

## Filters Implemented

### 1. **Box Filter**
- Averages the pixel values within a fixed-size neighborhood.
- Simple and fast, but causes blurring of edges.

### 2. **Gaussian Filter**
- Uses a Gaussian kernel, giving more weight to the center pixels.
- Works well for Gaussian noise while preserving edges better than the Box filter.

### 3. **Median Filter**
- Replaces each pixel with the median value of its neighbors.
- Particularly effective at removing Salt-and-Pepper noise, while maintaining edge sharpness.

### 4. **Bilateral Filter**
- Applies both spatial and intensity weighting to reduce noise while preserving sharp edges.
- Computationally expensive but provides excellent edge preservation.

### 5. **Adaptive Filters**
- Adaptive Mean: Changes the kernel size based on local image statistics.
- Adaptive Median: Similar to the Adaptive Mean but uses the median of the neighborhood.
- These filters provide the best noise reduction while preserving edges but are more computationally intensive.

---

## Noise Types

- **Gaussian Noise**: Introduces random variations in pixel values, creating a grainy effect.
- **Salt-and-Pepper Noise**: Adds random black and white pixels, disrupting image details.

Noise was introduced at three different levels:
- Low Intensity (0.05)
- Medium Intensity (0.1)
- High Intensity (0.2)

---

## Experimental Setup

### Images
The following images were used in the experiments:
- Bird
- Cat
- Dog

### Filters Tested
- Box filter, Gaussian filter, Median filter, Bilateral filter, Adaptive mean, and Adaptive median.
  
### Kernel Sizes
- 3, 7, 11, 15

### Performance Metrics
- **MSE (Mean Squared Error)**: Measures the difference between the filtered and clean image.
- **PSNR (Peak Signal-to-Noise Ratio)**: Indicates the quality of the denoised image; higher PSNR means better quality.
- **Edge Preservation**: Measures how well edges are preserved after applying the filter.
- **Processing Time**: Time taken by each filter to process the image.

---

## Performance Metrics

The following metrics were used to evaluate the effectiveness of the filters:
- **MSE (Mean Squared Error)**: The lower the MSE, the better the noise reduction.
- **PSNR (Peak Signal-to-Noise Ratio)**: A higher PSNR indicates better image quality after filtering.
- **Edge Preservation**: This metric evaluates how well the filter maintains edge sharpness while removing noise.
- **Processing Time**: Evaluates the computational efficiency of the filter.

---

## Results

### Filter Comparisons
- **MSE and PSNR Comparison**: Shows the effectiveness of each filter at different kernel sizes and noise levels.
- **Edge Preservation Comparison**: Demonstrates how well each filter preserves edges.
- **Computational Time Comparison**: Compares the processing time required by each filter at different kernel sizes.

### Key Findings:
- **Median and Adaptive filters** performed the best in preserving edges while reducing noise, particularly for Salt-and-Pepper noise.
- **Bilateral filter** was great for edge preservation but was computationally expensive.
- **Box filter** was fast but caused noticeable blurring with larger kernels.
- **Gaussian filter** provided a balance between noise reduction and edge preservation.

---



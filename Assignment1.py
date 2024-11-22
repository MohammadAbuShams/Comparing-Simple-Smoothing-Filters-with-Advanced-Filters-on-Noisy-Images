# Mohammad AbuShams
# 1200549

# Import libraries
import cv2
import numpy as np
import glob
import time
import os
import csv
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# Load images from a folder
def load_images_from_folder(folder):
    paths = glob.glob(folder + '/*.jpg')
    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is not None:
            images.append((img, path))
    print(f"Loaded {len(images)} images.")
    return images

# Add noise to an image
def add_noise(img, noise_type="gaussian", intensity=0.1):
    if img.ndim == 2:  # Grayscale image
        shape = img.shape
    else:  # Color image
        row, col, _ = img.shape
        shape = (row, col, 3)

    if noise_type == "gaussian":
        gauss = np.random.normal(0, intensity * 255, shape)
        noisy = img + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "salt_pepper":
        out = np.copy(img)
        num_salt = np.ceil(intensity * img.size * 0.5).astype(int)
        coords_salt = (np.random.randint(0, img.shape[0], num_salt),
                       np.random.randint(0, img.shape[1], num_salt))
        out[coords_salt] = 255
        num_pepper = np.ceil(intensity * img.size * 0.5).astype(int)
        coords_pepper = (np.random.randint(0, img.shape[0], num_pepper),
                         np.random.randint(0, img.shape[1], num_pepper))
        out[coords_pepper] = 0
        return out

# Apply specified filter to an image
def apply_filter(img, filter_type="gaussian", kernel_size=3):
    if filter_type == "box":
        return cv2.blur(img, (kernel_size, kernel_size))
    elif filter_type == "gaussian":
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif filter_type == "median":
        return cv2.medianBlur(img, kernel_size)
    elif filter_type == "bilateral":
        return cv2.bilateralFilter(img, d=kernel_size, sigmaColor=75, sigmaSpace=75)
    elif filter_type == "adaptive_mean":
        return adaptive_mean_filter(img, kernel_size)
    elif filter_type == "adaptive_median":
        return adaptive_median_filter(img, kernel_size)

# Adaptive mean filter function
def adaptive_mean_filter(src, max_kernel_size=15):
    # Create a copy of the source image to avoid modifying the original image
    temp = src.copy()
    # Iterate over every pixel in the image
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            # Start with a 3x3 kernel and increase size up to max_kernel_size
            k = 3
            while k <= max_kernel_size:
                # Define the neighborhood around the current pixel
                neighborhood = src[max(i - k // 2, 0): min(i + k // 2 + 1, src.shape[0]),
                               max(j - k // 2, 0): min(j + k // 2 + 1, src.shape[1])]
                # Calculate the mean of the neighborhood
                mean_val = np.mean(neighborhood)
                # Replace the center pixel with the mean value
                temp[i, j] = mean_val
                break
            # Increase the window size and continue filtering
            k += 2
    # Return the filtered image
    return temp

# Adaptive median filter function
def adaptive_median_filter(src, max_kernel_size=15):
    # Create a copy of the source image to avoid modifying the original image
    temp = src.copy()
    # Iterate over every pixel in the image
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            # Start with a 3x3 kernel and increase size up to max_kernel_size
            k = 3
            while k <= max_kernel_size:
                # Define the neighborhood around the current pixel
                neighborhood = src[max(i - k // 2, 0): min(i + k // 2 + 1, src.shape[0]), max(j - k // 2, 0): min(j + k // 2 + 1, src.shape[1])]
                # Calculate the median of the neighborhood
                med = np.median(neighborhood)
                min_value = np.min(neighborhood)
                max_value = np.max(neighborhood)
                if med != min_value and med != max_value:
                    # Replace the center pixel with the mean value
                    temp[i, j] = med
                    break
                # Increase the window size and continue filtering
                k += 2
    # Return the filtered image
    return temp

# Measure performance of the filter
def measure_performance(img, filtered_img):
    mse_val = mse(img, filtered_img)
    psnr_val = psnr(img, filtered_img)
    return mse_val, psnr_val

# Measure edge preservation
def edge_preservation(img, filtered_img):
    edges_original = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
    edges_filtered = cv2.Canny(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY), 100, 200)
    edge_diff = np.mean(np.abs(edges_original - edges_filtered))
    return edge_diff, edges_original, edges_filtered

# Save image to a file
def save_image(image, path, prefix, img_name):
    filename = os.path.join(path, f"{prefix}_{img_name}")
    cv2.imwrite(filename, image)
    return filename

# Plot noisy images
def plot_noises(images, results_path, noise_types, noise_levels):
    for img_tuple in images:
        img, img_path = img_tuple
        img_name = os.path.basename(img_path)

        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        fig.suptitle(f"Noise Results for {img_name}", fontsize=16)
        index = 0

        for noise_type in noise_types:
            for intensity in noise_levels:
                noisy_img_path = os.path.join(results_path, f"noisy_{noise_type}_{intensity}_{img_name}")
                noisy_img = cv2.imread(noisy_img_path)
                noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)

                ax = axes[index // 3, index % 3]
                ax.imshow(noisy_img)
                ax.axis('off')
                ax.set_title(f"{noise_type}, intensity={intensity}")
                index += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

folder_path = 'images'
images = load_images_from_folder(folder_path)
results_path = "results"
kernel_sizes = [3, 7, 11, 15]
filter_types = ["box", "gaussian", "median", "bilateral", "adaptive_mean", "adaptive_median"]
noise_types = ["gaussian", "salt_pepper"]
noise_levels = [0.05, 0.1, 0.2]  # low noise, medium noise, high noise

# Process images with different noise levels, filter types, and kernel sizes
def process_images(folder_path):
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, "performance_metrics.csv")

    if not images:
        print("No images to process.")
        return []

    # Store performance data for comparison
    performance_data = {
        "filter_type": [],
        "kernel_size": [],
        "noise_type": [],
        "intensity": [],
        "mse": [],
        "psnr": [],
        "edge_preservation": [],
        "processing_time": []
    }

    # Write the header row to the CSV file
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Noise Type", "Intensity", "Filter Type", "Kernel Size", "MSE", "PSNR",
                         "Edge Preservation", "Processing Time"])

    for img_tuple in images:
        img, img_path = img_tuple
        img_name = os.path.basename(img_path)
        print("*****************************************************************************************************************************************************************************************************************************************")
        print(f"Processing image: {img_name}")

        for noise_type in noise_types:
            for intensity in noise_levels:
                noisy_img = add_noise(img, noise_type, intensity)
                noisy_img_path = save_image(noisy_img, results_path, f"noisy_{noise_type}_{intensity}", img_name)

                for filter_type in filter_types:
                    for kernel_size in kernel_sizes:
                        start_time = time.time()
                        try:
                            filtered_img = apply_filter(noisy_img, filter_type, kernel_size)
                            filtered_img_path = save_image(filtered_img, results_path,
                                                           f"filtered_{filter_type}_{kernel_size}_{noise_type}_{intensity}", img_name)
                            mse_val, psnr_val = measure_performance(img, filtered_img)
                            edge_pres, _, _ = edge_preservation(img, filtered_img)
                            processing_time = time.time() - start_time

                            # Store data for plotting later
                            performance_data["filter_type"].append(filter_type)
                            performance_data["kernel_size"].append(kernel_size)
                            performance_data["noise_type"].append(noise_type)
                            performance_data["intensity"].append(intensity)
                            performance_data["mse"].append(mse_val)
                            performance_data["psnr"].append(psnr_val)
                            performance_data["edge_preservation"].append(edge_pres)
                            performance_data["processing_time"].append(processing_time)

                            with open(results_file, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([img_name, noise_type, intensity, filter_type, kernel_size,
                                                 f"{mse_val:.2f}", f"{psnr_val:.2f}", f"{edge_pres:.4f}",
                                                 f"{processing_time:.4f}"])

                            print(f"Noise: {noise_type}, Intensity: {intensity}, Filter: {filter_type}, "
                                  f"Kernel Size: {kernel_size}, MSE: {mse_val:.2f}, PSNR: {psnr_val:.2f} dB, "
                                  f"Edge Preservation: {edge_pres:.4f}, Time: {processing_time:.4f} seconds")

                        except Exception as e:
                            print(f"Error processing {img_name} with {filter_type} filter: {e}")

    print("Processing complete.")
    return performance_data

# Plot MSE and PSNR for comparison across noise levels, filter types, and kernel sizes
def plot_mse_psnr_comparison(performance_data):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].set_title("MSE Comparison")
    axes[0].set_xlabel("Filter Type, Kernel Size")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True)

    axes[1].set_title("PSNR Comparison")
    axes[1].set_xlabel("Filter Type, Kernel Size")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(True)

    filter_kernel_labels = [
        f"{filter_type}, k={kernel_size}"
        for filter_type in filter_types
        for kernel_size in kernel_sizes
    ]
    filter_kernel_index = {
        label: i for i, label in enumerate(filter_kernel_labels)
    }

    for i, filter_type in enumerate(performance_data["filter_type"]):
        kernel_size = performance_data["kernel_size"][i]
        mse_value = performance_data["mse"][i]
        psnr_value = performance_data["psnr"][i]
        label = f"{filter_type}, k={kernel_size}"

        axes[0].scatter(filter_kernel_index[label], mse_value, label=label, color='blue', alpha=0.5)

        axes[1].scatter(filter_kernel_index[label], psnr_value, label=label, color='red', alpha=0.5)

    axes[0].set_xticks(range(len(filter_kernel_labels)))
    axes[0].set_xticklabels(filter_kernel_labels, rotation=90)
    axes[1].set_xticks(range(len(filter_kernel_labels)))
    axes[1].set_xticklabels(filter_kernel_labels, rotation=90)

    plt.tight_layout()
    plt.show()

# Plot Edge Preservation Comparison
def plot_edge_preservation_comparison(performance_data):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_title("Edge Preservation Comparison", fontsize=16)
    ax.set_xlabel("Filter Type, Kernel Size", fontsize=12)
    ax.set_ylabel("Edge Preservation", fontsize=12)
    ax.grid(True)

    filter_kernel_labels = [
        f"{filter_type}, k={kernel_size}"
        for filter_type in filter_types
        for kernel_size in kernel_sizes
    ]
    filter_kernel_index = {
        label: i for i, label in enumerate(filter_kernel_labels)
    }

    for i, filter_type in enumerate(performance_data["filter_type"]):
        kernel_size = performance_data["kernel_size"][i]
        edge_preservation = performance_data["edge_preservation"][i]
        label = f"{filter_type}, k={kernel_size}"

        ax.scatter(filter_kernel_index[label], edge_preservation, label=label, alpha=0.5)

    ax.set_xticks(range(len(filter_kernel_labels)))
    ax.set_xticklabels(filter_kernel_labels, rotation=90)

    plt.tight_layout()
    plt.show()

# Plot Computational Time Comparison for different kernel sizes and filter types
def plot_computational_time_comparison(performance_data):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_title("Computational Time Comparison", fontsize=16)
    ax.set_xlabel("Kernel Size", fontsize=12)
    ax.set_ylabel("Computational Time (seconds)", fontsize=12)
    ax.grid(True)

    for filter_type in filter_types:
        filter_times = []
        filter_labels = []

        for kernel_size in kernel_sizes:
            times = []
            for i, ftype in enumerate(performance_data["filter_type"]):
                if ftype == filter_type and performance_data["kernel_size"][i] == kernel_size:
                    times.append(performance_data["processing_time"][i])

            filter_times.append(np.mean(times))  # Average time for each kernel size

        ax.plot(kernel_sizes, filter_times, label=filter_type, marker='o', linestyle='-', markersize=8)

    ax.legend(title="Filter Types")
    plt.tight_layout()
    plt.show()

# Plot edge preservation maps for different noise levels
def plot_edge_maps(images, filter_types, kernel_sizes, results_path):
    noise_combinations = [
        ("gaussian", 0.05, 3),  # Gaussian noise with intensity 0.05 and k=3
        ("salt_pepper", 0.1, 7),  # Salt-and-pepper noise with intensity 0.1 and k=7
        ("gaussian", 0.2, 11),  # Gaussian noise with intensity 0.2 and k=11
        ("salt_pepper", 0.05, 15)  # Salt-and-pepper noise with intensity 0.05 and k=15
    ]

    for img_tuple in images:
        img, img_path = img_tuple
        img_name = os.path.basename(img_path)

        fig, axes = plt.subplots(len(noise_combinations), len(filter_types), figsize=(15, 7))
        fig.suptitle(f"Edge Preservation Maps for {img_name}", fontsize=16)

        for i, (noise_type, noise_level, kernel_size) in enumerate(noise_combinations):
            for j, filter_type in enumerate(filter_types):
                noisy_img = add_noise(img, noise_type, noise_level)
                filtered_img = apply_filter(noisy_img, filter_type, kernel_size)
                _, edges_original, edges_filtered = edge_preservation(img, filtered_img)

                ax = axes[i, j] if len(noise_combinations) > 1 else axes[j]
                ax.imshow(edges_filtered, cmap='gray')
                ax.axis('off')
                ax.set_title(f"{filter_type}\n{noise_type} L={noise_level}, k={kernel_size}")

        plt.subplots_adjust(hspace=0.5)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

performance_data = process_images(folder_path)
plot_noises(images, results_path, noise_types, noise_levels)
plot_mse_psnr_comparison(performance_data)
plot_edge_maps(images, filter_types, kernel_sizes, results_path)
plot_edge_preservation_comparison(performance_data)
plot_computational_time_comparison(performance_data)

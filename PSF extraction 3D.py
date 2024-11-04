import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
import os

# 1. Dialog to Select Folder of .tif Files
def open_folder_dialog():
    app = QApplication(sys.argv)
    folder_dialog = QFileDialog()
    folder_dialog.setFileMode(QFileDialog.Directory)
    if folder_dialog.exec_():
        folder_path = folder_dialog.selectedFiles()[0]
        return folder_path
    sys.exit()

# 2. Function to find brightest points in the 3D image (with Gaussian blur and distance exclusion)
def find_brightest_points_3d(image, threshold=0.6, num_points=30, min_distance=25):
    # Normalize image
    image_normalized = image / np.max(image)
    
    # Find points above threshold
    bright_points = np.where(image_normalized >= threshold)
    intensities = image_normalized[bright_points]
    
    # Get the brightest points sorted by intensity
    brightest_indices = np.argsort(intensities)[::-1]  # Sort in descending order of intensity
    
    selected_points = []
    
    for idx in brightest_indices:
        point = (bright_points[0][idx], bright_points[1][idx], bright_points[2][idx])  # (z, y, x) point
        
        # Check if the point is at least 'min_distance' away from all previously selected points
        too_close = any(np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2 + (point[2] - p[2])**2) < min_distance for p in selected_points)
        
        if not too_close:
            selected_points.append(point)
        
        # Stop when we have enough points
        if len(selected_points) >= num_points:
            break
    
    # Filter out points that are too close to the edges
    height, width, depth = image.shape
    filtered_points = []
    
    for z, y, x in selected_points:
        if (min_distance <= x < width - min_distance) and (min_distance <= y < height - min_distance) and (min_distance <= z < depth - min_distance):
            filtered_points.append((z, y, x))
    
    return filtered_points

# 3. 3D Gaussian Function
def gaussian_3d(x, y, z, amp, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z):
    return (amp / (2 * np.pi * sigma_x * sigma_y * sigma_z)) * np.exp(
        -(((x - mu_x)**2) / (2 * sigma_x**2) + ((y - mu_y)**2) / (2 * sigma_y**2) + ((z - mu_z)**2) / (2 * sigma_z**2))
    )

# Function to fit Gaussian and get FWHM in 3D
def fit_gaussian_3d(data, x_axis, y_axis, z_axis):
    # Initial guess: amplitude, mean values, and sigma values
    initial_guess = [np.max(data), np.mean(x_axis), np.mean(y_axis), np.mean(z_axis), 1, 1, 1]

    # Flatten the data and create meshgrid for x, y, z axes
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')
    data_flat = data.flatten()
    x_flat = x_mesh.flatten()
    y_flat = y_mesh.flatten()
    z_flat = z_mesh.flatten()

    # Fit Gaussian to the data
    popt, _ = curve_fit(lambda xyz, amp, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z: gaussian_3d(xyz[0], xyz[1], xyz[2], amp, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z),
                        (x_flat, y_flat, z_flat), data_flat, p0=initial_guess)

    amp, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z = popt
    fwhm_x = 2 * np.sqrt(2 * np.log(2)) * np.abs(sigma_x)  # FWHM for X
    fwhm_y = 2 * np.sqrt(2 * np.log(2)) * np.abs(sigma_y)  # FWHM for Y
    fwhm_z = 2 * np.sqrt(2 * np.log(2)) * np.abs(sigma_z)  # FWHM for Z
    return (fwhm_x, fwhm_y, fwhm_z), (amp, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z)

# 4. Main function to handle files and processing
def process_images(folder_path):
    # Load all .tif files in the selected folder and stack them
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.tif'):
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path).convert('L')  # Convert image to grayscale
            images.append(np.array(image))

    # Convert the list of images to a 3D NumPy array (stack)
    image_stack = np.array(images)

    # Print folder path and number of images found
    print(f"Selected Folder: {folder_path}")
    print(f"Number of .tif images found: {len(images)}")

    # Apply Gaussian blur to the entire image stack to reduce noise
    blurred_image_stack = gaussian_filter(image_stack, sigma=3)

    # Find the brightest points in the blurred 3D image
    brightest_points = find_brightest_points_3d(blurred_image_stack)
    print(f"Brightest Points (z, y, x): {brightest_points}")

    # Fit Gaussian for each filtered brightest point and get FWHM
    fwhm_results = []
    z_axis = np.arange(blurred_image_stack.shape[0])  # Assuming Z-axis is the stack index
    y_axis = np.arange(blurred_image_stack.shape[1])
    x_axis = np.arange(blurred_image_stack.shape[2])
    
    for z, y, x in brightest_points:
        # Extract the 3D data around the point
        data_region = blurred_image_stack[z-1:z+2, y-25:y+25, x-25:x+25] / np.max(blurred_image_stack[z-1:z+2, y-25:y+25, x-25:x+25]) # A 3D region around the point
        
        # Fit Gaussian to the 3D data
        fwhm_values, params = fit_gaussian_3d(data_region, x_axis[x-25:x+25], y_axis[y-25:y+25], z_axis[z-1:z+2])

        fwhm_results.append({
            'point': (x, y, z),
            'fwhm': fwhm_values,
            'params': params
        })

        # Plotting: 50x50 pixel region and Gaussian fits for X, Y, Z
        plot_gaussian_fits(blurred_image_stack, (x, y, z), params, x_axis, y_axis, z_axis)

    return fwhm_results

# Function to plot region and Gaussian fits for X, Y, and Z axes
def plot_gaussian_fits(image_stack, point, params, x_axis, y_axis, z_axis):
    x, y, z = point
    amp, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z = params
    
    # Extract 50x50 region around the point
    region = image_stack[z, y-25:y+25, x-25:x+25]

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot 50x50 region
    axs[0, 0].imshow(region, cmap='gray')
    axs[0, 0].set_title('50x50 Region (x,y)')
    axs[0, 0].scatter(25, 25, color='red')  # Mark the bright point in the center

    # Plot X-axis intensity profile and Gaussian fit
    x_profile = image_stack[z, y, x-25:x+25]
    x_fit = gaussian_1d(np.arange(x-25, x+25), amp, mu_x, sigma_x)
    axs[0, 1].plot(np.arange(x-25, x+25), x_profile, label='X Profile')
    axs[0, 1].plot(np.arange(x-25, x+25), x_fit, label='Gaussian Fit', linestyle='--')
    axs[0, 1].set_title('X-axis Profile')
    axs[0, 1].legend()

    # Plot Y-axis intensity profile and Gaussian fit
    y_profile = image_stack[z, y-25:y+25, x]
    y_fit = gaussian_1d(np.arange(y-25, y+25), amp, mu_y, sigma_y)
    axs[1, 0].plot(np.arange(y-25, y+25), y_profile, label='Y Profile')
    axs[1, 0].plot(np.arange(y-25, y+25), y_fit, label='Gaussian Fit', linestyle='--')
    axs[1, 0].set_title('Y-axis Profile')
    axs[1, 0].legend()

    # Plot Z-axis intensity profile and Gaussian fit
    z_profile = image_stack[z-1:z+2, y, x]
    z_fit = gaussian_1d(np.arange(z-1, z+2), amp, mu_z, sigma_z)
    axs[1, 1].plot(np.arange(z-1, z+2), z_profile, label='Z Profile')
    axs[1, 1].plot(np.arange(z-1, z+2), z_fit, label='Gaussian Fit', linestyle='--')
    axs[1, 1].set_title('Z-axis Profile')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

# 5. Function to compute FWHM, print averages, and plot histograms
def analyze_fwhm(fwhm_results):
    x_fwhms = []
    y_fwhms = []
    z_fwhms = []
    
    # Collect FWHM values for x, y, and z
    for result in fwhm_results:
        fwhm_x, fwhm_y, fwhm_z = result['fwhm']
        x_fwhms.append(fwhm_x)
        y_fwhms.append(fwhm_y)
        z_fwhms.append(fwhm_z)
    
    # Calculate averages
    avg_fwhm_x = np.mean(x_fwhms)
    avg_fwhm_y = np.mean(y_fwhms)
    avg_fwhm_z = np.mean(z_fwhms)
    
    print(f'Average FWHM for X-axis: {avg_fwhm_x:.2f}')
    print(f'Average FWHM for Y-axis: {avg_fwhm_y:.2f}')
    print(f'Average FWHM for Z-axis: {avg_fwhm_z:.2f}')
    
    # Plot histograms for FWHM values (X, Y, Z)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(x_fwhms, bins=10, color='blue', alpha=0.7, label='X FWHM')
    plt.title('Histogram of X FWHM')
    plt.xlabel('FWHM (X)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(y_fwhms, bins=10, color='green', alpha=0.7, label='Y FWHM')
    plt.title('Histogram of Y FWHM')
    plt.xlabel('FWHM (Y)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(z_fwhms, bins=10, color='red', alpha=0.7, label='Z FWHM')
    plt.title('Histogram of Z FWHM')
    plt.xlabel('FWHM (Z)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 6. 1D Gaussian function for fitting
def gaussian_1d(x, amp, mu, sigma):
    return amp * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Main Execution
if __name__ == "__main__":
    folder_path = open_folder_dialog()
    fwhm_results = process_images(folder_path)
    analyze_fwhm(fwhm_results)

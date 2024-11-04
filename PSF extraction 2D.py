import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image

# 1. Dialog to Select Image File
def open_file_dialog():
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    file_dialog.setNameFilter("Images (*.tif)")
    if file_dialog.exec_():
        file_path = file_dialog.selectedFiles()[0]
        return file_path
    sys.exit()


# 2. Function to find brightest points in the image (with Gaussian blur and distance exclusion)
def find_brightest_points(image, threshold=0.6, num_points=10, blur_sigma=3, min_distance=50):
    # Apply Gaussian blur to the image to reduce noise
    image_blurred = gaussian_filter(image, sigma=blur_sigma)
    
    # Normalize blurred image
    image_normalized = image_blurred / np.max(image_blurred)
    
    # Find points above threshold
    bright_points = np.where(image_normalized >= threshold)
    intensities = image_normalized[bright_points]
    
    # Get the brightest points sorted by intensity
    brightest_indices = np.argsort(intensities)[::-1]  # Sort in descending order of intensity
    
    selected_points = []
    
    for idx in brightest_indices:
        point = (bright_points[0][idx], bright_points[1][idx])  # (y, x) point
        
        # Check if the point is at least 'min_distance' away from all previously selected points
        too_close = any(np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) < min_distance for p in selected_points)
        
        if not too_close:
            selected_points.append(point)
        
        # Stop when we have enough points
        if len(selected_points) >= num_points:
            break
    
    return selected_points


# 3. 1D Gaussian Function
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Function to fit Gaussian and get FWHM
def fit_gaussian_1d(data, axis_values):
    mean = np.sum(axis_values * data) / np.sum(data)
    sigma = np.sqrt(np.sum(data * (axis_values - mean) ** 2) / np.sum(data))

    popt, _ = curve_fit(gaussian, axis_values, data, p0=[np.max(data), mean, sigma])
    amplitude, mu, sigma_fitted = popt
    fwhm = 2 * np.sqrt(2 * np.log(2)) * np.abs(sigma_fitted)  # FWHM formula
    return fwhm, amplitude, mu, sigma_fitted

# 4. Main function to handle file and processing
def process_image(file_path):
    image = Image.open(file_path).convert('L')  # Convert image to grayscale
    image_np = np.array(image)
    
    # Find the brightest points in the image
    brightest_points = find_brightest_points(image_np)

    # Fit Gaussian for each brightest point and get FWHM
    fwhm_results = []
    for y, x in brightest_points:
        # Extract 1D profiles along x and y
        x_profile = image_np[y, :]
        y_profile = image_np[:, x]
        
        x_axis = np.arange(x_profile.shape[0])
        y_axis = np.arange(y_profile.shape[0])
        
        fwhm_x, amp_x, mu_x, sigma_x = fit_gaussian_1d(x_profile, x_axis)
        fwhm_y, amp_y, mu_y, sigma_y = fit_gaussian_1d(y_profile, y_axis)

        fwhm_results.append({
            'point': (x, y),
            'fwhm_x': fwhm_x,
            'fwhm_y': fwhm_y,
            'x_profile': (x_axis, x_profile, amp_x, mu_x, sigma_x),
            'y_profile': (y_axis, y_profile, amp_y, mu_y, sigma_y)
        })

    return fwhm_results

# 5. Plotting function (showing 50x50 pixel region and the Gaussian fit for X and Y profiles)
def plot_fits(fwhm_results, image, region_size=50):
    half_region = region_size // 2
    
    for result in fwhm_results:
        x, y = result['point']
        x_axis, x_profile, amp_x, mu_x, sigma_x = result['x_profile']
        y_axis, y_profile, amp_y, mu_y, sigma_y = result['y_profile']

        # Define region limits around the bright point
        x_min = max(0, x - half_region)
        x_max = min(image.shape[1], x + half_region)
        y_min = max(0, y - half_region)
        y_max = min(image.shape[0], y + half_region)
        
        # Extract the 50x50 pixel region of the image
        image_region = image[y_min:y_max, x_min:x_max]
        
        # Slice profiles to show only the region around the point
        x_region = np.arange(x_min, x_max)
        y_region = np.arange(y_min, y_max)
        
        x_profile_region = x_profile[x_min:x_max]
        y_profile_region = y_profile[y_min:y_max]
        
        # Plot the 50x50 pixel region around the bright point
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image_region, cmap='gray')
        plt.title(f'50x50 Region around Point {result["point"]}')
        plt.colorbar()
        
        # Plot X profile and Gaussian fit
        plt.subplot(1, 3, 2)
        plt.plot(x_region, x_profile_region, label='X Profile', color='blue')
        plt.plot(x_region, gaussian(x_region, amp_x, mu_x, sigma_x), label='Gaussian Fit (X)', color='red', linestyle='--')
        plt.title(f'Point {result["point"]} - X Profile (Region)')
        plt.xlabel('X Axis')
        plt.ylabel('Intensity')
        plt.legend()

        # Plot Y profile and Gaussian fit
        plt.subplot(1, 3, 3)
        plt.plot(y_region, y_profile_region, label='Y Profile', color='green')
        plt.plot(y_region, gaussian(y_region, amp_y, mu_y, sigma_y), label='Gaussian Fit (Y)', color='orange', linestyle='--')
        plt.title(f'Point {result["point"]} - Y Profile (Region)')
        plt.xlabel('Y Axis')
        plt.ylabel('Intensity')
        plt.legend()

        plt.tight_layout()
        plt.show()


# Helper function to calculate FWHM from Gaussian fit parameters
def calculate_fwhm(sigma):
    return 2.355 * sigma  # FWHM for a Gaussian distribution

# 6. Function to compute FWHM, print averages, and plot histograms
def analyze_fwhm(fwhm_results):
    x_fwhms = []
    y_fwhms = []
    x_and_y_fwhms = []
    
    # Collect FWHM values for x and y
    for result in fwhm_results:
        _, _, _, _, sigma_x = result['x_profile']
        _, _, _, _, sigma_y = result['y_profile']
        
        # Calculate FWHM for both X and Y using sigma
        fwhm_x = calculate_fwhm(sigma_x)
        fwhm_y = calculate_fwhm(sigma_y)
        
        x_fwhms.append(fwhm_x)
        y_fwhms.append(fwhm_y)
        x_and_y_fwhms.append(fwhm_x)
        x_and_y_fwhms.append(fwhm_y)
    
    # Calculate averages
    avg_fwhm_x = np.mean(x_fwhms)
    avg_fwhm_y = np.mean(y_fwhms)
    avg_fwhm_xy = np.mean(x_and_y_fwhms)
    
    print(f'Average FWHM for X-axis: {avg_fwhm_x:.2f}')
    print(f'Average FWHM for Y-axis: {avg_fwhm_y:.2f}')
    print(f'Average FWHM for X & Y-axes: {avg_fwhm_xy:.2f}')
    
    # Plot histograms for FWHM values (X and Y)
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
    plt.hist(x_and_y_fwhms, bins=10, color='red', alpha=0.7, label='XY FWHM')
    plt.title('Histogram of XY FWHM')
    plt.xlabel('FWHM (XY)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == "__main__":
    file_path = open_file_dialog()
    fwhm_results = process_image(file_path)
    analyze_fwhm(fwhm_results)
    image = Image.open(file_path).convert('L')
    plot_fits(fwhm_results, np.array(image))

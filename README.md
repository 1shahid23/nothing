import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image
image = cv2.imread("cat.jpeg")  # Replace 'image.jpg' with your image file

# Convert BGR to RGB (since OpenCV loads images in BGR format)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split the RGB channels
R, G, B = cv2.split(image)

# Create blank (zero) arrays with the same shape
zeros = np.zeros_like(R)

# Generate images with only one channel active
red_only = cv2.merge([R, zeros, zeros])
green_only = cv2.merge([zeros, G, zeros])
blue_only = cv2.merge([zeros, zeros, B])

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Convert to binary (Thresholding)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Function to display images
def show_image(title, img, cmap=None):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()
    
# Display the original and split images
show_image("Original Image", image)
show_image("Red Channel", red_only)
show_image("Green Channel", green_only)
show_image("Blue Channel", blue_only)
show_image("Grayscale Image", gray, cmap="gray")
show_image("Binary Image", binary, cmap="gray")

# Plot histograms for each image
def plot_histogram(image, title , color = "black"):
    plt.figure(figsize=(6,4))
    plt.hist(image.ravel(), bins=256, range=[0, 256], color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

# Display histograms
plot_histogram(R, "Histogram of Red Channel", color = "Red")
plot_histogram(G, "Histogram of Green Channel", color = "green")
plot_histogram(B, "Histogram of Blue Channel", color = "blue")
plot_histogram(gray, "Histogram of Grayscale Image")
plot_histogram(binary, "Histogram of Binary Image")


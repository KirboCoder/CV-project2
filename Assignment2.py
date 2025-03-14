import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('bicycle.bmp', cv2.IMREAD_GRAYSCALE)

#box filter no openCV
def box_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    # // is integer division aka truncating the result
    padded_image = cv2.copyMakeBorder(image, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2, cv2.BORDER_REPLICATE)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sum_value = 0
            for k in range(kernel_size):
                for l in range(kernel_size):
                    sum_value += padded_image[i+k, j+l] * kernel[k, l]
            filtered_image[i, j] = sum_value
            
    return filtered_image.astype(np.uint8)

#sovel filter for x no opencv
def sobel_filterx(image, kernel_size):
    if kernel_size == 3:
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif kernel_size == 5:
        sobel_x = np.array([[-2, -1, 0, 1, 2], 
                            [-3, -2, 0, 2, 3], 
                            [-4, -3, 0, 3, 4], 
                            [-3, -2, 0, 2, 3], 
                            [-2, -1, 0, 1, 2]])
    return cv2.filter2D(image, -1, sobel_x)

# sobel filter for y no opencv
def sobel_filtery(image, kernel_size):
    if kernel_size == 3:
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif kernel_size == 5:
        sobel_y = np.array([[-2, -3, -4, -3, -2], 
                            [-1, -2, -3, -2, -1], 
                            [0, 0, 0, 0, 0], 
                            [1, 2, 3, 2, 1], 
                            [2, 3, 4, 3, 2]])
    return cv2.filter2D(image, -1, sobel_y)

# sobel fil;ter combine both x and y
def sobel_filterxy(image, kernel_size):
    sobel_x = sobel_filterx(image, kernel_size)
    sobel_y = sobel_filtery(image, kernel_size)
    return np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)

#All filters using opencv
box_filter_cv3x3 = cv2.blur(image, (3, 3))
box_filter_cv5x5 = cv2.blur(image, (5, 5))
sobelCV_x3x3 = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobelCV_x5x5 = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobelCV_y3x3 = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobelCV_y5x5 = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobelCV_xy3x3 = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
sobelCV_xy5x5 = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
gaussian_blur3x3 = cv2.GaussianBlur(image, (3, 3), 0)
gaussian_blur5x5 = cv2.GaussianBlur(image, (5, 5), 0)


#applying all custom filters
box3x3 = box_filter(image, 3)
box5x5 = box_filter(image, 5)
sobel_x3x3 = sobel_filterx(image, 3)
sobel_x5x5 = sobel_filterx(image, 5)
sobel_y3x3 = sobel_filtery(image, 3)
sobel_y5x5 = sobel_filtery(image, 5)
sobel_xy3x3 = sobel_filterxy(image, 3)
sobel_xy5x5 = sobel_filterxy(image, 5)

# Display results
filters = [
    ("Box Filter 3x3", box3x3), ("Box Filter 5x5", box5x5),
    ("Sobel X 3x3", sobel_x3x3), ("Sobel X 5x5", sobel_x5x5),
    ("Sobel Y 3x3", sobel_y3x3), ("Sobel Y 5x5", sobel_y5x5),
    ("Sobel XY 3x3", sobel_xy3x3), ("Sobel XY 5x5", sobel_xy5x5),
    ("Box Filter OpenCV 3x3", box_filter_cv3x3), ("Box Filter OpenCV 5x5", box_filter_cv5x5),
    ("Sobel X OpenCV 3x3", sobelCV_x3x3), ("Sobel X OpenCV 5x5", sobelCV_x5x5),
    ("Sobel Y OpenCV 3x3", sobelCV_y3x3), ("Sobel Y OpenCV 5x5", sobelCV_y5x5),
    ("Sobel XY OpenCV 3x3", sobelCV_xy3x3), ("Sobel XY OpenCV 5x5", sobelCV_xy5x5),
    ("Gaussian Blur OpenCV 3x3", gaussian_blur3x3), ("Gaussian Blur OpenCV 5x5", gaussian_blur5x5)
]


plt.figure(figsize=(20, 16))
for i in range(len(filters)):
    plt.subplot(5, 4, i+1)
    plt.imshow(filters[i][1], cmap='gray')
    plt.title(filters[i][0])
    plt.axis("off")
plt.tight_layout()
plt.show()


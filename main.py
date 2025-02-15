import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2
from sklearn.cluster import KMeans


def apply_perspective_transformation(image, homography):
    transformed_image = cv2.warpPerspective(image, homography, (image.shape[1], image.shape[0]))
    return transformed_image


def detect_lines(edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    return lines


def detect_vanishing_points(lines):
    vanishing_points = []
    for line1 in lines:
        for line2 in lines:
            if line1[0][1] != line2[0][1]:
                rho1, theta1 = line1[0]
                rho2, theta2 = line2[0]
                x = (rho1 * np.sin(theta2) - rho2 * np.sin(theta1)) / (
                        np.sin(theta1) * np.sin(theta2) - np.cos(theta1) * np.cos(theta2))
                y = (rho1 * np.cos(theta2) - rho2 * np.cos(theta1)) / (
                        np.sin(theta1) * np.sin(theta2) - np.cos(theta1) * np.cos(theta2))
                vanishing_points.append((x, y))
    return vanishing_points


def apply_sobel_filter(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return sobel_x, sobel_y


def calculate_image_moments(image):
    moments = cv2.moments(image)
    return moments


def estimate_homography(src_points, dst_points):
    homography, _ = cv2.findHomography(src_points, dst_points)
    return homography

def cluster_vanishing_points(vanishing_points):
    # Use k-means clustering to reduce the number of vanishing points
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(vanishing_points)
    clustered_points = kmeans.cluster_centers_
    return clustered_points

def compute_horizon_line(vanishing_points):
    # Compute the horizon line using the vanishing points
    x1, y1 = vanishing_points[0]
    x2, y2 = vanishing_points[1]
    horizon_line = (y2 - y1) / (x2 - x1)
    return horizon_line

class DataAnalyzer:
    def __init__(self, xyz_path, image_path):
        self.image = None
        self.xyz_path = xyz_path
        self.image_path = image_path
        self.xyz_data = self.load_xyz_data()
        self.image_data = self.load_image_data()
        max_x = int(np.max(self.xyz_data[:, 0])) # Columns
        max_y = int(np.max(self.xyz_data[:, 1])) # Rows
        self.grid_data = self.xyz_data[:, 2].reshape(max_x + 1, max_y + 1)
        self.image = cv2.cvtColor(np.array(self.image_data), cv2.COLOR_RGB2BGR)

    def transform_xyz_data(self):

        transformation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        transformed_xyz_data = np.dot(self.xyz_data, transformation_matrix)

        min_x, min_y, min_z = np.min(transformed_xyz_data, axis=0)
        max_x, max_y, max_z = np.max(transformed_xyz_data, axis=0)
        scaled_xyz_data = (transformed_xyz_data - [min_x, min_y, min_z]) / [max_x - min_x, max_y - min_y, max_z - min_z]
        scaled_xyz_data[:, 0] *= self.image.shape[1]
        scaled_xyz_data[:, 1] *= self.image.shape[0]

        return scaled_xyz_data

    def load_xyz_data(self):
        return np.loadtxt(self.xyz_path)

    def load_image_data(self):
        return Image.open(self.image_path)

    def get_image(self):
        return self.image

    def detect_edges(self):
        blurred_image = cv2.GaussianBlur(self.image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 150, 250)
        return edges

    def plot_data(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.grid_data, cmap='viridis')
        plt.title('XYZ Data')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(self.image_data)
        plt.title('Image Data')
        plt.axis('off')

        plt.show()

    def analyze_image(self):
        edges = self.detect_edges()
        lines = detect_lines(edges)
        vanishing_points = detect_vanishing_points(lines)
        clustered_points = cluster_vanishing_points(vanishing_points)

    def analyze_image0(self):
        edges = self.detect_edges()
        lines = detect_lines(edges)
        vanishing_points = detect_vanishing_points(lines)
        # clustered_points = cluster_vanishing_points(vanishing_points)
        # horizon_line = compute_horizon_line(clustered_points)
        # camera_orientation = estimate_camera_orientation(clustered_points, horizon_line)
        sobel_x, sobel_y = apply_sobel_filter(self.image)
        moments = calculate_image_moments(edges)

        plt.figure(figsize=(12, 10))
        plt.subplot(3, 2, 1)
        plt.imshow(edges, cmap='gray')
        plt.title('Edges')
        plt.axis('off')

        plt.subplot(3, 2, 2)
        plt.imshow(cv2.cvtColor(np.array(self.image_data), cv2.COLOR_RGB2BGR))
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.title('Lines')
        plt.axis('off')

        plt.subplot(3, 2, 3)
        plt.imshow(sobel_x, cmap='gray')
        plt.title('Sobel X')
        plt.axis('off')

        plt.subplot(3, 2, 4)
        plt.imshow(sobel_y, cmap='gray')
        plt.title('Sobel Y')
        plt.axis('off')

        plt.subplot(3, 2, 5)
        plt.imshow(self.image, cmap='gray')
        plt.title('Vanishing Points')
        plt.axis('off')
        for point in vanishing_points:
            plt.plot(point[0], point[1], 'ro')

        plt.subplot(3, 2, 6)
        plt.imshow(edges, cmap='gray')
        plt.title('Image Moments')
        plt.axis('off')
        print("Image Moments:")
        print(moments)

        src_points = np.array([[100, 100], [300, 100], [100, 300], [300, 300]])
        dst_points = np.array(
            [[0, 0], [self.image.shape[1], 0], [0, self.image.shape[0]], [self.image.shape[1], self.image.shape[0]]])
        homography = estimate_homography(src_points, dst_points)

        transformed_image = apply_perspective_transformation(self.image, homography)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
        plt.title('Transformed Image')
        plt.axis('off')
        plt.show()

        transformed_xyz_data = self.transform_xyz_data()

        plt.figure(figsize=(10, 10))
        plt.scatter(transformed_xyz_data[:, 0], transformed_xyz_data[:, 1], c=transformed_xyz_data[:, 2],
                    cmap='viridis')
        plt.title('Transformed XYZ Data')
        plt.axis('off')
        plt.show()


# data_analyzer = DataAnalyzer('3D_Map/australia1.xyz', '3D_Map/australia1.png')
# data_analyzer.plot_data()
# data_analyzer.analyze_image()

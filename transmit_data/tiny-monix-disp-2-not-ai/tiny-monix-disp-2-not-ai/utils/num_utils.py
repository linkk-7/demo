
import re
import numpy as np

def numerical_sort(value: str, index: int = -1) -> int:
    """ Helper function to sort filenames with numerical values correctly. """
    numbers = re.findall(r'\d+', value)
    return int(numbers[index]) if numbers else 0

def calculate_distance_matrix(points):
    return np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)

def average_nearest_neighbors(distance_matrix, k=4):
    average_distances = []
    for i in range(distance_matrix.shape[0]):
        # Get the distances for the current point, excluding itself
        distances = distance_matrix[i]
        
        # Get the indices of the k nearest neighbors (excluding itself)
        nearest_indices = np.argsort(distances)[1:k+1]  # [1:] excludes the point itself
        
        # Calculate the average distance to these nearest neighbors
        avg_distance = np.mean(distances[nearest_indices])
        average_distances.append(avg_distance)
    
    return np.array(average_distances)
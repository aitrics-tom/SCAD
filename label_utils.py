import json
import numpy as np
import os

def save_superclass_labels(superclass_indices, dataset_name, save_dir="labels"):
    """
    Save superclass labels to a JSON file
    
    Args:
        superclass_indices: Dictionary or list of superclass indices
        dataset_name: Dataset name (cifar10, cifar100, stl10, smallimagenet)
        save_dir: Directory to save the file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert dictionary format to list format
    if isinstance(superclass_indices, dict):
        superclass_indices_list = []
        for i in range(len(superclass_indices)):
            superclass_indices_list.append(superclass_indices[i])
    else:
        superclass_indices_list = superclass_indices
    
    save_path = os.path.join(save_dir, f"{dataset_name}_superclass_labels.json")
    
    with open(save_path, 'w') as f:
        json.dump(superclass_indices_list, f, indent=2)
    
    print(f"Superclass labels saved to: {save_path}")
    return save_path

def load_superclass_labels(dataset_name, save_dir="labels"):
    """
    Load saved superclass labels from a JSON file
    
    Args:
        dataset_name: Dataset name (cifar10, cifar100, stl10, smallimagenet)
        save_dir: Directory where the file is saved
        
    Returns:
        superclass_indices_list: List of superclass indices
    """
    load_path = os.path.join(save_dir, f"{dataset_name}_superclass_labels.json")
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Label file not found: {load_path}")
    
    with open(load_path, 'r') as f:
        superclass_indices_list = json.load(f)
    
    print(f"Superclass labels loaded from: {load_path}")
    return superclass_indices_list

def create_coarse_labels_from_list(superclass_indices_list):
    """
    Create coarse labels from a list of superclass indices
    
    Args:
        superclass_indices_list: List of superclass indices
        
    Returns:
        coarse_labels: numpy array of coarse labels
    """
    if not superclass_indices_list:
        return np.array([])
        
    max_index = max(max(sublist) for sublist in superclass_indices_list)
    coarse_labels = np.zeros(max_index + 1, dtype=int)
    
    for superclass, sublist in enumerate(superclass_indices_list):
        for index in sublist:
            coarse_labels[index] = superclass
            
    return coarse_labels

def assign_coarse_labels_from_dict(superclass_indices):
    """
    Create coarse labels from dictionary format superclass indices
    
    Args:
        superclass_indices: Dictionary format superclass indices
        
    Returns:
        coarse_labels: numpy array of coarse labels
    """
    max_index = max([max(indices) for indices in superclass_indices.values()])
    coarse_labels = np.zeros(max_index + 1, dtype=int)

    for superclass, indices in superclass_indices.items():
        for index in indices:
            coarse_labels[index] = superclass

    return coarse_labels

# Default superclass label definitions (for initial saving)
DEFAULT_SUPERCLASS_LABELS = {
    'cifar10': [[0,1,8,9],[2,6],[3,4,5,7]],
    'stl10': [[0,2,8,9],[1,7],[3,4,5,6]],
    'cifar100': [
        [1, 26, 45, 77, 99],
        [27, 29, 44, 73, 78, 93],
        [21, 42, 43, 88],
        [54, 62, 70, 82, 92],
        [58, 69, 85, 89],
        [2, 11, 35, 46, 98],
        [22, 86, 87],
        [6, 7, 14, 18, 24, 79],
        [36, 50, 65, 74, 75, 80],
        [0, 51, 53, 57, 83],
        [5, 20, 25, 94],
        [23, 71],
        [33, 49, 60],
        [9, 10, 28],
        [13, 81, 90],
        [15, 19, 31],
        [17, 37, 76],
        [16, 39, 40, 61, 84],
        [32, 67, 91],
        [30, 95, 72],
        [8, 41, 48],
        [3, 4, 34, 55, 97],
        [12, 68],
        [38, 63, 66, 64],
        [47, 52, 56, 59, 96]    
    ],
    'smallimagenet': [
        [17, 43, 60, 99, 125],
        [16, 58, 118],
        [2, 22, 36, 44, 45, 50, 55, 84, 90, 113],
        [27, 28, 30, 42, 71, 80, 94, 106, 109],
        [29, 34, 41, 47, 53, 110, 115],
        [31, 37, 39, 63, 74, 77, 111],
        [9, 32, 59, 76, 105, 121],
        [19],
        [75, 86, 104],
        [25, 46, 56, 98],
        [38, 79, 107, 108, 124, 126],
        [72, 81, 114],
        [8, 21, 66, 83],
        [11, 14, 54, 65, 68],
        [88, 91],
        [52, 64, 67, 112],
        [18, 35, 93, 119],
        [6, 24, 49, 78, 87, 102, 116, 122],
        [20, 120],
        [95, 103],
        [1, 4, 40, 51],
        [70, 101],
        [3],
        [13, 89, 96, 97, 100],
        [69, 117],
        [0, 73],
        [5, 26, 123],
        [15, 62],
        [12, 23, 48, 57, 85, 92],
        [10],
        [33],
        [7, 61, 82]
    ]
}

def initialize_default_labels():
    """
    Save default superclass labels to files
    """
    for dataset_name, labels in DEFAULT_SUPERCLASS_LABELS.items():
        save_superclass_labels(labels, dataset_name)

if __name__ == "__main__":
    # Save default labels
    initialize_default_labels()
    print("Default superclass labels have been saved!")

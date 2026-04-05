import os
import random
import numpy as np
from PIL import Image

class KolektorSDDLoader:
    """
    Loads images from the KolektorSDD dataset and translates their physical 
    properties into parameters for the TaskDAG subtasks.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.image_paths = self._index_dataset()

    def _index_dataset(self) -> list:
        """
        Scans the dataset directory and collects all image file paths.
        Assuming a standard extracted folder structure for KolektorSDD.
        """
        valid_extensions = ('.jpg', '.png', '.bmp')
        image_paths = []
        
        if not os.path.exists(self.dataset_path):
            print(f"Warning: Dataset path '{self.dataset_path}' not found.")
            print("Running in dummy mode for testing.")
            return []

        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(root, file))
                    
        return image_paths

    def get_random_task_parameters(self) -> dict:
        """
        Picks a random image and generates realistic data sizes and CPU cycle 
        estimates for the 5-stage image recognition DAG.
        """
        if not self.image_paths:
            # Fallback to dummy data if the dataset isn't downloaded yet
            file_size_bits = random.uniform(1e6, 5e6) # 1 to 5 Megabits
            pixels = random.randint(500000, 2000000)
        else:
            # Load actual image properties
            img_path = random.choice(self.image_paths)
            file_size_bytes = os.path.getsize(img_path)
            file_size_bits = file_size_bytes * 8
            
            with Image.open(img_path) as img:
                width, height = img.size
                pixels = width * height

        # The paper outlines specific subtasks for image recognition:
        # 1. Image Extraction, 2. Denoising, 3. Standardization, 
        # 4. Feature Extraction, 5. Detection & Recognition
        
        # We model the data sizes (D), result sizes (R), and CPU cycles (C) 
        # proportionally based on the real image size and pixel count.
        
        task_params = {
            "subtask_1": { # Image Extraction
                "data_size": file_size_bits, 
                "result_size": file_size_bits * 1.2, # Uncompressed data
                "cpu_cycles": pixels * 10 
            },
            "subtask_2": { # Image Denoising
                "data_size": file_size_bits * 1.2,
                "result_size": file_size_bits * 1.2,
                "cpu_cycles": pixels * 50 # Compute intensive filter
            },
            "subtask_3": { # Standardization
                "data_size": file_size_bits * 1.2,
                "result_size": file_size_bits * 0.8, # Resizing/Cropping
                "cpu_cycles": pixels * 20
            },
            "subtask_4": { # Feature Extraction (e.g., passing through CNN layers)
                "data_size": file_size_bits * 0.8,
                "result_size": file_size_bits * 0.1, # Vector representation
                "cpu_cycles": pixels * 200 # Highly compute intensive
            },
            "subtask_5": { # Detection and Recognition
                "data_size": file_size_bits * 0.1,
                "result_size": 256, # Just a classification result/bounding box (small)
                "cpu_cycles": pixels * 50
            }
        }
        
        return task_params
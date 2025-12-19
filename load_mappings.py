import pickle
import os

# Construct the full path to the cat_mappings.pkl file
project_root = os.getcwd() # Assumes the current working directory is the project root
file_path = os.path.join(project_root, "Logs", "Tabnet", "cat_mappings.pkl")

try:
    with open(file_path, "rb") as f:
        cat_mappings = pickle.load(f)
    
    print("Contents of cat_mappings.pkl:")
    for key, value in cat_mappings.items():
        print(f"  Column '{key}': {value}")
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please ensure the model has been trained.")
except Exception as e:
    print(f"An error occurred while loading cat_mappings.pkl: {e}")

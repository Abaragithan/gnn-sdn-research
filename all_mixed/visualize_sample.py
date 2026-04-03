import tensorflow as tf
import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generator import input_fn

def visualize_single_sample():
    # Define path to dataset - assuming standard structure based on other scripts
    # You may need to change this path if your data is located elsewhere
    data_dir = '../data/all_mixed/test' 
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        print("Please update the 'data_dir' variable in this script to point to your dataset.")
        return

    print(f"Loading dataset from {data_dir}...")
    ds = input_fn(data_dir, shuffle=False)
    
    # Take a single sample from the dataset
    for inputs, targets in ds.take(1):
        print("\n" + "="*40)
        print(" SAMPLE INPUT STRUCTURE ")
        print("="*40)
        
        # Iterate through the input dictionary
        for key, value in inputs.items():
            print(f"\nFeature: '{key}'")
            
            if isinstance(value, tf.RaggedTensor):
                print(f"  Type: RaggedTensor")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                # Show a few elements to understand the ragged structure
                print(f"  First 2 rows: {value.to_list()[:2]}")
            elif isinstance(value, tf.Tensor):
                print(f"  Type: Tensor")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  Values (first 5): {value.numpy().flatten()[:5]}")
            else:
                print(f"  Type: {type(value)}")
                print(f"  Value: {value}")

        print("\n" + "="*40)
        print(" SAMPLE OUTPUT (LABEL) ")
        print("="*40)
        print(f"Target: 'delay'")
        print(f"  Shape: {targets.shape}")
        print(f"  Dtype: {targets.dtype}")
        print(f"  Values (first 5): {targets.numpy()[:5]}")
        
        # Break after one sample
        break

if __name__ == "__main__":
    visualize_single_sample()

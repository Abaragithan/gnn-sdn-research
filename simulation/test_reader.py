import sys

# Point to the folder containing datanetAPI.py and the tar.gz files
dataset_dir = "/sim_dataset_v1/train/50/scenario-topo-50-0"
sys.path.append(dataset_dir)

from datanetAPI import DatanetAPI

try:
    print(f"Loading dataset from: {dataset_dir}")
    reader = DatanetAPI(dataset_dir)
    
    # Try to extract the very first sample
    iterator = iter(reader)
    sample = next(iterator)
    
    # Print a few topology characteristics to prove it parsed correctly
    num_nodes = sample.get_topology_object().number_of_nodes()
    
    print("\nSUCCESS! The API successfully read the compressed OMNeT++ data.")
    print(f"Sample 1 contains a network with {num_nodes} nodes.")
    print("Your data generation and ingestion pipeline is complete.")

except Exception as e:
    print(f"Failed to read dataset: {e}")
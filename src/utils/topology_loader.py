import networkx as nx
import torch
import numpy as np

class TopologyLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        # Load the GraphML file using NetworkX
        self.graph = nx.read_graphml(file_path)
        # Standardize node names to integers (important for GNN indexing)
        self.graph = nx.convert_node_labels_to_relabeling_graph(self.graph)

    def get_networkx_graph(self):
        """Returns the raw NetworkX graph for visualization or shortest-path baselines."""
        return self.graph

    def get_gnn_data(self):
        """
        Converts the GEANT2 graph into PyTorch Geometric format.
        This is for Objective #1 (Model Design).
        """
        # Get edges in a format PyTorch Geometric understands (Edge Index)
        adj = nx.to_edgelist(self.graph)
        edge_index = torch.tensor([[int(u), int(v)] for u, v, d in adj], dtype=torch.long).t().contiguous()
        
        # Create initial dummy node features (e.g., node degree)
        # You will later replace this with real traffic data
        node_features = torch.tensor([[self.graph.degree(n)] for n in self.graph.nodes()], dtype=torch.float)
        
        return node_features, edge_index

    def get_mininet_links(self):
        """Returns links as a list of tuples for experiments/geant2_topo.py."""
        return list(self.graph.edges())

    def get_node_count(self):
        return self.graph.number_of_nodes()

# Quick test if run directly
if __name__ == "__main__":
    loader = TopologyLoader('data/topologies/Geant2012.graphml.xml')
    print(f"Successfully loaded GEANT2 with {loader.get_node_count()} nodes.")
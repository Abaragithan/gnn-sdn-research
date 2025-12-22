from src.utils.topology_loader import TopologyLoader
loader = TopologyLoader('data/topologies/Geant2012.graphml.xml')
links = loader.get_mininet_links() # Ready for self.addLink()
import osmnx as ox
import networkx as nx

class OSMDataCollector:
    def __init__(self, area):
        self.area = area

    def collect_road_network(self):
        G = ox.graph_from_place(self.area, network_type='drive')
        return G

    def extract_toll_information(self, G):
        toll_edges = [(u, v, k, d) for u, v, k, d in G.edges(data=True, keys=True) if d.get('toll') == 'yes']
        return toll_edges

    def save_graph(self, G, filename):
        nx.write_gpickle(G, filename)

# Usage
collector = OSMDataCollector("New York City, USA")
G = collector.collect_road_network()
toll_edges = collector.extract_toll_information(G)
collector.save_graph(G, "data/raw/nyc_road_network.gpickle")

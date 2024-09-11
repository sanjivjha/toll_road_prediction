import networkx as nx
import random
import numpy as np

class SyntheticDataGenerator:
    def __init__(self, graph_file):
        self.G = nx.read_gpickle(graph_file)

    def generate_route(self, length):
        nodes = list(self.G.nodes())
        start, end = random.sample(nodes, 2)
        path = nx.shortest_path(self.G, start, end, weight='length')
        return path[:length] if len(path) >= length else self.generate_route(length)

    def simulate_trajectory(self, route, time_step=1):
        trajectory = []
        for i in range(len(route) - 1):
            start, end = route[i], route[i+1]
            edge_data = self.G.get_edge_data(start, end)[0]
            num_steps = int(edge_data['length'] / time_step)
            for step in range(num_steps):
                progress = step / num_steps
                lat = start[0] + progress * (end[0] - start[0])
                lon = start[1] + progress * (end[1] - start[1])
                trajectory.append((lat, lon))
        return trajectory

    def generate_labels(self, route):
        labels = []
        on_toll_road = False
        total_distance = 0
        total_charge = 0
        for i in range(len(route) - 1):
            start, end = route[i], route[i+1]
            edge_data = self.G.get_edge_data(start, end)[0]
            if edge_data.get('toll') == 'yes':
                on_toll_road = True
                total_distance += edge_data['length']
                total_charge += edge_data.get('toll_cost', 1)  # Assume unit cost if not specified
            labels.append((on_toll_road, total_distance, total_charge))
        return labels

    def create_dataset(self, num_samples, route_length, time_step):
        dataset = []
        for _ in range(num_samples):
            route = self.generate_route(route_length)
            trajectory = self.simulate_trajectory(route, time_step)
            labels = self.generate_labels(route)
            dataset.append((trajectory, labels))
        return dataset

# Usage
generator = SyntheticDataGenerator("data/raw/nyc_road_network.gpickle")
dataset = generator.create_dataset(1000, 50, 1)

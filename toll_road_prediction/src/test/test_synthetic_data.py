import unittest
from src.data.synthetic_data_generator import SyntheticDataGenerator

class TestSyntheticDataGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SyntheticDataGenerator("data/raw/nyc_road_network.gpickle")

    def test_generate_route(self):
        route = self.generator.generate_route(50)
        self.assertEqual(len(route), 50)
        self.assertTrue(all(node in self.generator.G.nodes() for node in route))

    def test_simulate_trajectory(self):
        route = self.generator.generate_route(10)
        trajectory = self.generator.simulate_trajectory(route)
        self.assertTrue(len(trajectory) > 0)
        self.assertIsInstance(trajectory[0], tuple)
        self.assertEqual(len(trajectory[0]), 2)  # lat, lon

    def test_generate_labels(self):
        route = self.generator.generate_route(10)
        labels = self.generator.generate_labels(route)
        self.assertEqual(len(labels), len(route) - 1)
        self.assertIsInstance(labels[0], tuple)
        self.assertEqual(len(labels[0]), 3)  # on_toll_road, total_distance, total_charge

    def test_create_dataset(self):
        dataset = self.generator.create_dataset(10, 50, 1)
        self.assertEqual(len(dataset), 10)
        self.assertIsInstance(dataset[0], tuple)
        self.assertEqual(len(dataset[0]), 2)  # trajectory, labels

if __name__ == '__main__':
    unittest.main()

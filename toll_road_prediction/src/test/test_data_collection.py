import unittest
from src.data.osm_data_collector import OSMDataCollector

class TestOSMDataCollector(unittest.TestCase):
    def setUp(self):
        self.collector = OSMDataCollector("New York City, USA")

    def test_collect_road_network(self):
        G = self.collector.collect_road_network()
        self.assertIsNotNone(G)
        self.assertTrue(len(G.nodes()) > 0)
        self.assertTrue(len(G.edges()) > 0)

    def test_extract_toll_information(self):
        G = self.collector.collect_road_network()
        toll_edges = self.collector.extract_toll_information(G)
        self.assertIsInstance(toll_edges, list)
        for edge in toll_edges:
            self.assertEqual(edge[3].get('toll'), 'yes')

if __name__ == '__main__':
    unittest.main()

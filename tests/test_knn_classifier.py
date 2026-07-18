import unittest

import networkx as nx
import numpy as np

from knn_classifier import (
    DATA_DIR,
    bfs_sample,
    extract_features_fast,
    knn_predict,
    load_graph,
    split_data,
)


class KnnClassifierTests(unittest.TestCase):
    def test_included_dataset_metadata_matches_loaded_graph(self):
        graph = load_graph(DATA_DIR / 'CA-GrQc.txt')

        self.assertEqual(graph.number_of_nodes(), 5242)
        self.assertEqual(graph.number_of_edges(), 14496)

    def test_scalability_features_exclude_degree_label_source(self):
        graph = nx.star_graph(4)

        features, labels = extract_features_fast(graph)

        self.assertEqual(features.shape, (5, 1))
        np.testing.assert_allclose(
            features[:, 0],
            [1.0, 4.0, 4.0, 4.0, 4.0],
        )
        np.testing.assert_array_equal(labels, [1, 0, 0, 0, 0])

    def test_bfs_sample_respects_requested_size(self):
        graph = nx.path_graph(10)

        sampled = bfs_sample(graph, start_node=0, size=4)

        self.assertEqual(sampled, {0, 1, 2, 3})

    def test_split_data_is_reproducible(self):
        features = np.arange(20).reshape(10, 2)
        labels = np.arange(10)

        first = split_data(features, labels, seed=7)
        second = split_data(features, labels, seed=7)

        for first_array, second_array in zip(first, second):
            np.testing.assert_array_equal(first_array, second_array)

    def test_knn_predicts_nearest_class(self):
        train_features = np.array([[0.0], [0.2], [0.8], [1.0]])
        train_labels = np.array([0, 0, 1, 1])
        test_features = np.array([[0.1], [0.9]])

        predictions = knn_predict(train_features, train_labels, test_features, k=3)

        np.testing.assert_array_equal(predictions, [0, 1])


if __name__ == '__main__':
    unittest.main()

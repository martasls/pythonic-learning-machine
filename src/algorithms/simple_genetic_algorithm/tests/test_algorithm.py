from data.io import load_samples
from data.extract import get_input_variables, get_target_variable
from algorithms.simple_genetic_algorithm.algorithm import SimpleGeneticAlgorithm
from algorithms.common.neural_network.neural_network import create_network_from_topology
from algorithms.common.stopping_criterion import MaxGenerationsCriterion
from algorithms.simple_genetic_algorithm.selection_operator import SelectionOperatorTournament
from algorithms.simple_genetic_algorithm.mutation_operator import MutationOperatorGaussian
from algorithms.simple_genetic_algorithm.crossover_operator import CrossoverOperatorArithmetic
from algorithms.common.metric import RootMeanSquaredError

import unittest


class TestAlgorithm(unittest.TestCase):
    def setUp(self):
        self.training, self.validation, self.testing = load_samples('r_music', 0)
        topology = create_network_from_topology([2, 2])
        self.ftne = SimpleGeneticAlgorithm(100, MaxGenerationsCriterion(200),
                                          topology, SelectionOperatorTournament(5), MutationOperatorGaussian(0.05),
                                          CrossoverOperatorArithmetic(), 0.01, 0.5)

    def test_fit(self):
        X = get_input_variables(self.training).as_matrix()
        y = get_target_variable(self.training).as_matrix()
        self.ftne.fit(X, y, RootMeanSquaredError, verbose=True)
        self.assertTrue(expr=self.ftne.champion)
        prediction = self.ftne.predict(get_input_variables(self.validation).as_matrix())
        self.assertEqual(len(prediction), len(get_target_variable(self.validation).as_matrix()))

if __name__ == '__main__':
    unittest.main()

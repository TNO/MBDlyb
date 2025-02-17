# -*- coding: utf-8 -*-
"""
    Copyright (c) 2023 - 2025 TNO-ESI
    All rights reserved.
"""

import unittest

import numpy as np

from test_model import load_cdradio_player


class TestCpt(unittest.TestCase):
    def setUp(self) -> None:
        self._fn = load_cdradio_player()
        self._weight = 0.9
        self._fp_rate = 0.002
        self._fn_rate = 0.05

    def compare_cpt(self, cpt: np.ndarray, ground_truth: np.ndarray) -> bool:
        self.assertEqual(cpt.shape, ground_truth.shape)
        self.assertAlmostEqual(np.sum(np.abs(cpt - ground_truth)), 0, delta=1e-15)

    def test_cpt_function(self):
        node = self._fn.get_node('CDRadioPlayer.CDReader.ReadCD')
        cpt = node._compute_cpt()
        ground_truth = np.array([([1, 0], [0, 1]), ([0, 1], [0, 1])])
        self.compare_cpt(cpt, ground_truth)

    def test_cpt_function_weight(self):
        node = self._fn.get_node('CDRadioPlayer.CDReader.ReadCD')
        for rel in node.parent_relations:
            rel.weight = self._weight
        cpt = node._compute_cpt()
        ground_truth = np.array([([1, 0],
                                  [1-self._weight, self._weight]),
                                 ([1-self._weight, self._weight],
                                  [(1-self._weight)*(1-self._weight), 1-(1-self._weight)*(1-self._weight)])])
        self.compare_cpt(cpt, ground_truth)

    def test_cpt_testresult(self):
        node = self._fn.get_node('CDRadioPlayer.CDReader.CD_present_result')
        node.fp_rate = self._fp_rate
        node.fn_rate = self._fn_rate
        cpt = node._compute_cpt()
        ground_truth = np.array([[1-self._fn_rate, self._fn_rate],
                                 [self._fp_rate, 1-self._fp_rate]])
        self.compare_cpt(cpt, ground_truth)

    def test_cpt_testresult_weight(self):
        node = self._fn.get_node('CDRadioPlayer.CDReader.CD_present_result')
        node.fp_rate = self._fp_rate
        node.fn_rate = self._fn_rate
        for rel in node.parent_relations:
            rel.weight = self._weight
        cpt = node._compute_cpt()
        ground_truth = np.array([[1-self._fn_rate, self._fn_rate],
                                 [(1-self._weight)*(1-self._fn_rate) + self._weight*self._fp_rate,
                                  (1-self._weight)*self._fn_rate + self._weight * (1-self._fp_rate)]])
        self.compare_cpt(cpt, ground_truth)

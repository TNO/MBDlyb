# -*- coding: utf-8 -*-
"""
    Copyright (c) 2023 - 2025 TNO-ESI
    All rights reserved.
"""

import unittest

from test_model import load_cdradio_player


class TestFunctional(unittest.TestCase):
    def setUp(self) -> None:
        self._fn = load_cdradio_player()

    def test_model_parameters(self):
        # check the number of nodes and relations in the model
        self.assertEqual(len(self._fn.get_flat_nodes()), 28)
        self.assertEqual(len(self._fn.get_flat_relations()), 32)

    def test_export_to_yed(self):
        # convert graph to yed diagram, check that graph is not empty
        yed = self._fn.to_yed()
        graph = yed.get_graph()
        self.assertGreater(len(graph), 1000)

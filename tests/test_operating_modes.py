# -*- coding: utf-8 -*-
"""
    Copyright (c) 2023 - 2025 TNO-ESI
    All rights reserved.
"""

import unittest

from mbdlyb.operating_mode import OpmSet, OpmVariable, OpmLogicOr, OpmLogicAnd


class TestOperatingMode(unittest.TestCase):
    def test_opm_set(self):
        # parse edges
        a = OpmVariable('a')
        b = OpmVariable('b')
        c = OpmVariable('c')
        d = OpmVariable('d')
        e = OpmVariable('e')
        f = OpmVariable('e')

        # parse nodes
        X = OpmSet()
        X.add_variables([a, b])
        X.add_variables([e, f])
        Y = OpmSet()
        Y.add_variables([c, d])
        
        # expected result: X and Z both containing {a,b,e} and Y containing {c,d}
        # both X and Z not containing f (there is no OpmVariable 'f')
        self.assertEqual(len(X.variable_names), 3)
        self.assertIn('a', X.variable_names)
        self.assertIn('b', X.variable_names)
        self.assertIn('e', X.variable_names)
        self.assertNotIn('f', X.variable_names)
        self.assertNotIn('c', X.variable_names)
        self.assertNotIn('d', X.variable_names)

        self.assertEqual(len(Y.variable_names), 2)
        self.assertIn('c', Y.variable_names)
        self.assertIn('d', Y.variable_names)
        self.assertNotIn('a', Y.variable_names)
        self.assertNotIn('b', Y.variable_names)
        self.assertNotIn('e', Y.variable_names)
        self.assertNotIn('f', Y.variable_names)

    def test_opm_logic(self):
        # Create to OpmLogicOr objects with operating modes (a|b) and (c|d)
        # Create one OpmLogicAnd object which evaluates ((a|b) & (c|d))

        # expected results:  (a | b) True True True
        opm1 = OpmLogicOr.create(['a', 'b'])
        opmset1 = OpmSet()
        opmset1.add_variables(opm1.variables)
        self.assertEqual(str(opm1), '(a | b)')
        self.assertTrue(opm1.eval(['a']))
        self.assertTrue(opm1.eval(['b']))
        self.assertTrue(opm1.eval(['a', 'b']))
        
        # expected results:  (c | d) True True True
        opm2 = OpmLogicOr.create(['c', 'd'])
        opmset2 = OpmSet()
        opmset2.add_variables(opm2.variables)
        self.assertEqual(str(opm2), '(c | d)')
        self.assertTrue(opm2.eval(['c']))
        self.assertTrue(opm2.eval(['d']))
        self.assertTrue(opm2.eval(['c', 'd']))
        
        # expected results: ((a | b) & (c | d)) True True
        opm3 = OpmLogicAnd().add(opm1).add(opm2)
        self.assertEqual(str(opm3), '((a | b) & (c | d))')
        self.assertTrue(opm3.eval(['b', 'c']))
        self.assertTrue(opm3.eval(['a', 'c', 'e']))

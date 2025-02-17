# -*- coding: utf-8 -*-
"""
    Copyright (c) 2023 - 2025 TNO-ESI
    All rights reserved.
"""
import unittest

from mbdlyb.functional import Diagnoser
from mbdlyb.formalisms import BayesNetReasoner, TensorNetReasoner, MarkovNetReasoner

from test_model import load_cdradio_player


class TestDiagnoser:
    def setUp(self) -> None:
        self._fn = load_cdradio_player()
        self._diagnoser = Diagnoser(self._fn, self.reasoner)
        self._diagnoser.set_entropy_limit(10)

    def _execute_test(self, test: str, value: str):
        self._diagnoser.add_evidence({test: value})

    def _execute_operating_mode(self, mode: str, value: str):
        self._diagnoser.add_operating_mode({mode: value})

    def _check_expectation_hardware(self, hardware_list: list[str]):
        df = self._diagnoser.infer()
        least_healthy_hardware = [node.fqn for node in df[df.Healthy <= (df.Healthy.min() + 1.0E-9)].index]
        for hw in hardware_list:
            self.assertIn(hw, least_healthy_hardware)

    def _check_expectation_test(self, test_list: list[str], balance: float = 0.5):
        en = self._diagnoser.compute_entropy_and_cost_of_diagnostic_tests()
        entropy_max = max(en['Entropy'])
        cost_max = max(en['Cost'])
        en['Loss'] = balance * en['Entropy'] / entropy_max + (1.0 - balance) * en['Cost'] / cost_max
        en.sort_values(by='Loss', inplace=True)
        for i, test in enumerate(test_list):
            self.assertEqual(test, en['Diagnostic Test'].iloc[i].fqn)

    def _check_model_structure(self, expected_size: int, forbidden_names: list[str] = []):
        nodes = self._diagnoser.net.get_flat_nodes()
        self.assertEqual(len(nodes), expected_size)
        for name in forbidden_names:
            self.assertFalse(any([name in node.fqn for node in nodes]))

    def test_PlayRadio_BatteryPower(self):
        self._check_model_structure(28)
        self._execute_test('CDRadioPlayer.AudioSystem.IsMusicPlaying', 'NOk')
        self._execute_operating_mode('CDRadioPlayer.AudioSystem.AudioSource', 'PlayRadio')
        self._check_model_structure(23, ['CDReader', 'ReadCD', 'CD_present'])
        self._execute_operating_mode('CDRadioPlayer.PowerSystem.BoardSystem.PowerSource', 'BatteryPower')
        self._check_model_structure(19, ['CDReader', 'ReadCD', 'CD_present', 'WallPower', 'Cable'])
        self._check_expectation_hardware(['CDRadioPlayer.RadioReceiver.Antenna', 'CDRadioPlayer.PowerSystem.BatterySystem.Battery',
                                          'CDRadioPlayer.PowerSystem.BoardSystem.PowerBoard_HW', 'CDRadioPlayer.AudioSystem.Speaker'])
        self._check_expectation_test(['CDRadioPlayer.RadioReceiver.Radio_Signal_present'])

        self._execute_test('CDRadioPlayer.RadioReceiver.Radio_Signal_present_result', 'NOk')
        self._check_expectation_hardware(['CDRadioPlayer.RadioReceiver.Antenna', 'CDRadioPlayer.PowerSystem.BatterySystem.Battery',
                                          'CDRadioPlayer.PowerSystem.BoardSystem.PowerBoard_HW'])
        self._check_expectation_test(['CDRadioPlayer.PowerSystem.BatterySystem.Test_Battery'])

        self._execute_test('CDRadioPlayer.PowerSystem.BatterySystem.Test_Battery_result', 'NOk')
        self._check_expectation_hardware(['CDRadioPlayer.PowerSystem.BatterySystem.Battery'])

    def test_PlayRadio_WallPower(self):
        self._check_model_structure(28)
        self._execute_test('CDRadioPlayer.AudioSystem.IsMusicPlaying', 'NOk')
        self._execute_operating_mode('CDRadioPlayer.AudioSystem.AudioSource', 'PlayRadio')
        self._check_model_structure(23, ['CDReader', 'ReadCD', 'CD_present'])
        self._execute_operating_mode('CDRadioPlayer.PowerSystem.BoardSystem.PowerSource', 'WallPower')
        self._check_model_structure(19, ['CDReader', 'ReadCD', 'CD_present', 'Battery'])
        self._check_expectation_hardware(['CDRadioPlayer.RadioReceiver.Antenna', 'CDRadioPlayer.PowerSystem.WallPowerSystem.PowerCable',
                                          'CDRadioPlayer.PowerSystem.BoardSystem.PowerBoard_HW', 'CDRadioPlayer.AudioSystem.Speaker'])
        self._check_expectation_test(['CDRadioPlayer.RadioReceiver.Radio_Signal_present'])

        self._execute_test('CDRadioPlayer.RadioReceiver.Radio_Signal_present_result', 'Ok')
        self._check_expectation_hardware(['CDRadioPlayer.AudioSystem.Speaker'])

    def test_PlayCD_WallPower(self):
        self._check_model_structure(28)
        self._execute_test('CDRadioPlayer.AudioSystem.IsMusicPlaying', 'NOk')
        self._execute_operating_mode('CDRadioPlayer.AudioSystem.AudioSource', 'PlayCD')
        self._check_model_structure(24, ['Receive', 'Signal', 'Antenna'])
        self._execute_operating_mode('CDRadioPlayer.PowerSystem.BoardSystem.PowerSource', 'WallPower')
        self._check_model_structure(20, ['Receive', 'Signal', 'Antenna', 'Battery'])
        self._check_expectation_hardware(['CDRadioPlayer.CDReader.CDReader', 'CDRadioPlayer.PowerSystem.WallPowerSystem.PowerCable',
                                          'CDRadioPlayer.PowerSystem.BoardSystem.PowerBoard_HW', 'CDRadioPlayer.AudioSystem.Speaker'])
        self._check_expectation_test(['CDRadioPlayer.CDReader.CD_present'])

        self._execute_test('CDRadioPlayer.CDReader.CD_present_result', 'NOk')
        self._check_expectation_hardware(['CDRadioPlayer.CDReader.CDReader', 'CDRadioPlayer.PowerSystem.WallPowerSystem.PowerCable',
                                          'CDRadioPlayer.PowerSystem.BoardSystem.PowerBoard_HW'])
        self._check_expectation_test(['CDRadioPlayer.PowerSystem.WallPowerSystem.Test_Cable'])

        self._execute_test('CDRadioPlayer.PowerSystem.WallPowerSystem.Test_Cable_result', 'NOk')
        self._check_expectation_hardware(['CDRadioPlayer.PowerSystem.WallPowerSystem.PowerCable'])


class TestBayesNetReasoner(TestDiagnoser, unittest.TestCase):
    'Test the Functional Diagnoser with the BayesNet reasoner'
    reasoner = BayesNetReasoner


class TestTensorNetReasoner(TestDiagnoser, unittest.TestCase):
    'Test the Functional Diagnoser with the TensorNet reasoner'
    reasoner = TensorNetReasoner

class TestMarkovNetReasoner(TestDiagnoser, unittest.TestCase):
    'Test the Functional Diagnoser with the MarkovNet reasoner'
    reasoner = MarkovNetReasoner

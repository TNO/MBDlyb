# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from mbdlyb.functional import (Cluster, Function, Hardware, DirectObservable, DiagnosticTest, DiagnosticTestResult,
							   OperatingMode, RealizesRelation, RequiredForRelation, SubfunctionOfRelation,
							   ObservedByRelation, IndicatedByRelation, ResultsInRelation, SelectOperatingModeRelation,
							   Diagnoser, gdb)
from mbdlyb.operating_mode import OpmLogic


BASE_HW_PRIOR = gdb.Hardware.BASE_PRIOR
DO_FPR = gdb.DirectObservable.fp_rate.default
DO_FNR = gdb.DirectObservable.fn_rate.default
DT_FPR = gdb.DiagnosticTestResult.fp_rate.default
DT_FNR = gdb.DiagnosticTestResult.fn_rate.default


def load_cdradio_player() -> Cluster:
	root = Cluster('CDRadioPlayer')
	listen_music = Function('ListenMusic', root)

	# AudioSystem
	audio_system = Cluster('AudioSystem', root)
	play_music = Function('PlayMusic', audio_system)
	speaker = Hardware('Speaker', BASE_HW_PRIOR, audio_system)
	is_music_playing = DirectObservable('IsMusicPlaying', DO_FPR, DO_FNR, audio_system)
	audio_source = OperatingMode('AudioSource', ['PlayCD', 'PlayRadio'], audio_system)

	# PowerSystem
	power_system = Cluster('PowerSystem', root)
	provide_power_low = Function('ProvidePowerLow', power_system)
	provide_power_high = Function('ProvidePowerHigh', power_system)
	power_hardware = Hardware('PowerHardware', BASE_HW_PRIOR, power_system)
	# BoardSystem
	board_system = Cluster('BoardSystem', power_system)
	provide_low_power = Function('ProvideLowPower', board_system)
	provide_high_power = Function('ProvideHighPower', board_system)
	power_board_hw = Hardware('PowerBoard_HW', BASE_HW_PRIOR, board_system)
	power_source = OperatingMode('PowerSource', ['BatteryPower', 'WallPower'], board_system)
	# BatterySystem
	battery_system = Cluster('BatterySystem', power_system)
	provide_battery_power = Function('ProvideBatteryPower', battery_system)
	battery = Hardware('Battery', BASE_HW_PRIOR, battery_system)
	test_battery_result = DiagnosticTestResult('Test_Battery_result', DT_FPR, DT_FNR, battery_system)
	test_battery = DiagnosticTest('Test_Battery', {'Time': 10}, dict(), battery_system)
	# WallPowerSystem
	wall_power_system = Cluster('WallPowerSystem', power_system)
	provide_wall_power = Function('ProvideWallPower', wall_power_system)
	power_cable = Hardware('PowerCable', BASE_HW_PRIOR, wall_power_system)
	test_cable = DiagnosticTest('Test_Cable', {'Time': 10}, dict(), wall_power_system)
	test_cable_result = DiagnosticTestResult('Test_Cable_result', DT_FPR, DT_FNR, wall_power_system)

	# CDReader
	cd_reader = Cluster('CDReader', root)
	read_cd = Function('ReadCD', cd_reader)
	cd_reader_hw = Hardware('CDReader', BASE_HW_PRIOR, cd_reader)
	cd_present = DiagnosticTest('CD_present', {'Time': 5}, dict(), cd_reader)
	cd_present_result = DiagnosticTestResult('CD_present_result', DT_FPR, DT_FNR, cd_reader)

	# RadioReceiver
	radio_receiver = Cluster('RadioReceiver', root)
	receive_radio = Function('ReceiveRadio', radio_receiver)
	antenna = Hardware('Antenna', BASE_HW_PRIOR, radio_receiver)
	radio_signal_present = DiagnosticTest('Radio_Signal_present', {'Time': 5}, dict(), radio_receiver)
	radio_signal_present_result = DiagnosticTestResult('Radio_Signal_present_result', DT_FPR, DT_FNR, radio_receiver)

	# Realizes relations
	RealizesRelation(speaker, play_music)
	RealizesRelation(power_hardware, provide_power_low)
	RealizesRelation(power_hardware, provide_power_high)
	RealizesRelation(power_board_hw, provide_low_power)
	RealizesRelation(power_board_hw, provide_high_power)
	RealizesRelation(battery, provide_battery_power)
	RealizesRelation(power_cable, provide_wall_power)
	RealizesRelation(cd_reader_hw, read_cd)
	RealizesRelation(antenna, receive_radio)

	# Subfunction relations
	SubfunctionOfRelation(play_music, listen_music)
	SubfunctionOfRelation(provide_low_power, provide_power_low)
	SubfunctionOfRelation(provide_high_power, provide_power_high)

	# Required-for relations
	RequiredForRelation(provide_low_power, play_music)
	RequiredForRelation(provide_low_power, receive_radio)
	RequiredForRelation(provide_high_power, read_cd)
	RequiredForRelation(provide_battery_power, provide_low_power, OpmLogic.create(['BatteryPower'], power_source.opm_set))
	RequiredForRelation(provide_wall_power, provide_low_power, OpmLogic.create(['WallPower'], power_source.opm_set))
	RequiredForRelation(provide_wall_power, provide_high_power, OpmLogic.create(['WallPower'], power_source.opm_set))
	RequiredForRelation(read_cd, play_music, OpmLogic.create(['PlayCD'], audio_source.opm_set))
	RequiredForRelation(receive_radio, play_music, OpmLogic.create(['PlayRadio'], audio_source.opm_set))

	# Observed-by relations
	ObservedByRelation(play_music, is_music_playing)

	# Indicated-by relations
	IndicatedByRelation(read_cd, cd_present_result)
	IndicatedByRelation(receive_radio, radio_signal_present_result)
	IndicatedByRelation(battery, test_battery_result)
	IndicatedByRelation(power_cable, test_cable_result)

	# Results-in relations
	ResultsInRelation(cd_present, cd_present_result)
	ResultsInRelation(radio_signal_present, radio_signal_present_result)
	ResultsInRelation(test_battery, test_battery_result)
	ResultsInRelation(test_cable, test_cable_result)

	# Select-operating-mode relations
	SelectOperatingModeRelation(audio_source, play_music)
	SelectOperatingModeRelation(power_source, provide_low_power)
	SelectOperatingModeRelation(power_source, provide_high_power)

	# Post-operations
	root.propagate_operating_modes()

	return root


if __name__ == '__main__':
	system = load_cdradio_player()
	system.to_yed().write_graph('testmodel.graphml')
	system.save_bn('testmodel', overwrite=True)

	from mbdlyb.formalisms.bayesnet import BayesNetReasoner
	from mbdlyb.formalisms.tensor_network import TensorNetReasoner
	from mbdlyb.formalisms.markovnet import MarkovNetReasoner

	opmodes = system.get_all_opm_nodes()
	d1 = Diagnoser(system, BayesNetReasoner)
	d1.add_operating_mode({'CDRadioPlayer.AudioSystem.AudioSource': 'PlayCD',
						   'CDRadioPlayer.PowerSystem.BoardSystem.PowerSource': 'WallPower'})
	d1._net.to_yed().write_graph('testmodel_opm_bn.graphml')

	d2 = Diagnoser(system, TensorNetReasoner)
	d2.add_operating_mode({'CDRadioPlayer.AudioSystem.AudioSource': 'PlayCD',
						  'CDRadioPlayer.PowerSystem.BoardSystem.PowerSource': 'WallPower'})
	d2._net.to_yed().write_graph('testmodel_opm_tn.graphml')

	d3 = Diagnoser(system, MarkovNetReasoner)
	d3.add_operating_mode({'CDRadioPlayer.AudioSystem.AudioSource': 'PlayCD',
						  'CDRadioPlayer.PowerSystem.BoardSystem.PowerSource': 'WallPower'})
	d3._net.to_yed().write_graph('testmodel_opm_mn.graphml')

# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from typing import Type, Union, Callable

import numpy as np
import pandas as pd

from mbdlyb import MBDReasoner, MBDNet
from mbdlyb.functional import Cluster, DiagnosticTest, DiagnosticTestResult, DirectObservable, Hardware
from mbdlyb.formalisms import select_reasoner

from collections import defaultdict
from dataclasses import dataclass


class SystemCondition:
	name: str = None
	setting: bool = False
	_cost_to_true: dict[str, float] = None
	_cost_to_false: dict[str, float] = None

	def __init__(self, name: str, setting: bool = False, cost_to_true: dict[str, float] | float | int = 0,
				 cost_to_false: dict[str, float] | float | int = 0) -> None:
		self.name = name
		self.setting = setting
		self.cost_to_true = cost_to_true
		self.cost_to_false = cost_to_false

	def __repr__(self) -> str:
		return f"SystemCondition({self.name}, {self.setting}, cost_to_true = {self.cost_to_true}, cost_to_false = {self.cost_to_false})"

	@property
	def cost_to_true(self) -> float:
		"""
		Returns cost for setting the system condition from false to true.
		If multiple types of costs are given, then this returns the sum.
		"""
		return sum(self._cost_to_true.values())

	@cost_to_true.setter
	def cost_to_true(self, cost_to_true: dict[str, float] | float | int):
		if isinstance(cost_to_true, (int, float)):
			self._cost_to_true = {'uncategorised': float(cost_to_true)}
		else:
			self._cost_to_true = cost_to_true or dict()

	@cost_to_true.deleter
	def cost_to_true(self):
		self._cost_to_true = dict()

	@property
	def cost_to_false(self) -> float:
		"""
		Returns cost for setting the system condition from true to false.
		If multiple types of costs are given, then this returns the sum.
		"""
		return sum(self._cost_to_false.values())

	@cost_to_false.setter
	def cost_to_false(self, cost_to_false: dict[str, float] | float | int):
		if isinstance(cost_to_false, (int, float)):
			self._cost_to_false = {'uncategorised': float(cost_to_false)}
		else:
			self._cost_to_false = cost_to_false or dict()

	@cost_to_false.deleter
	def cost_to_false(self):
		self._cost_to_false = dict()

	def get_cost(self, required_state: bool) -> float:
		if required_state == self.setting:
			return 0.0  # not changing system state is for free (?)
		elif required_state:
			return self.cost_to_true
		else:
			return self.cost_to_false


@dataclass
class SystemState:
	conditions: dict[str, SystemCondition]

	def get_condition(self, name: str):
		return self.conditions[name]

	def insert_condition(self, system_condition: SystemCondition):
		self.conditions[system_condition.name] = system_condition

	def insert_conditions(self, system_conditions: list[SystemCondition]):
		for system_condition in system_conditions:
			self.insert_condition(system_condition)

	def update_condition_setting(self, name: str, setting: bool):
		self.conditions[name].setting = setting

	def update_conditions_settings(self, updates: dict[str, bool]):
		for name, setting in updates:
			self.update_condition_setting(name, setting)

	def prepare_for_diagnostic_test(self, diagnostic_test: DiagnosticTest):
		"""
		Update system conditions as to satisfy the preconditions for the diagnostic test.
		"""
		if diagnostic_test.preconditions:
			for precondition in diagnostic_test.preconditions:
				self.conditions[precondition].setting = diagnostic_test.preconditions[precondition]

	def calculate_cost_of_diagnostic_test(self, diagnostic_test: DiagnosticTest):
		"""
		Compute the cost of a diagnostic test relative to the system state.

		:param diagnostic_test: diagnostic test of which the cost is calculated.
		"""
		cost = diagnostic_test.get_fixed_cost() or 0
		if diagnostic_test.preconditions:
			for name, required_setting in diagnostic_test.preconditions.items():
				cost += self.conditions[name].get_cost(required_setting)
		return cost


class Diagnoser:
	_net: Cluster = None  # _net is the active net currently being diagnosed
	_orig_net: Cluster = None  # _orig_net is the original net as added to the diagnoser on initialization
	_reasoner: MBDReasoner = None
	_entropy_limit: int = None

	_system_state: SystemState = None
	_operating_modes: dict[str, str] = None
	_diagnostic_tests: dict[str, DiagnosticTest] = None
	_diagnostic_test_results: dict[str, DiagnosticTestResult] = None
	_performed_diagnostic_tests: list[DiagnosticTest] = None
	_computed_next_diagnostic_tests: pd.DataFrame = None
	_total_diagnostic_cost: float = None
	_loss_function: Callable[[float, float], float] = None  # Entropy x Cost -> Loss

	def __init__(self, net: Cluster, reasoner_klass: Type[MBDReasoner] = None):
		self._net = net
		self._orig_net = net
		self.reset(reasoner_klass)  # initialization is handled in the reset-function
		self._operating_modes = dict()

	def _init(self, reasoner_klass: Type[MBDReasoner] = None):
		self._reasoner = select_reasoner(self._net, reasoner_klass)

	@property
	def net(self) -> MBDNet:
		return self._net

	@property
	def reasoner(self) -> MBDReasoner:
		return self._reasoner

	def set_loss_function(self, loss_function: Callable[[float, float], float]):
		self._loss_function = loss_function

	def initialise_system_state(self, state: dict[str, SystemCondition] = None):
		self._system_state = SystemState(state or dict())

	@property
	def total_diagnostic_cost(self):
		return self._total_diagnostic_cost

	def perform_diagnostic_test(self,
								diagnostic_test: DiagnosticTest | str,
								evidence: dict[str, str | list[float]] = None):
		"""
			This method performs a given diagnostic tests, i.e. adds cost to total diagnostic cost, 
			update system state, and (optionally) inserts evidence.

			:param diagnostic_test: DiagnosticTest to be performed.
			:param evidence: Evidence to be inserted.
		"""
		if not isinstance(diagnostic_test, DiagnosticTest):
			diagnostic_test = self._net.get_node(diagnostic_test)
		self._performed_diagnostic_tests.append(diagnostic_test)
		self._total_diagnostic_cost += self._system_state.calculate_cost_of_diagnostic_test(diagnostic_test)
		self._system_state.prepare_for_diagnostic_test(diagnostic_test)
		if evidence:
			self._reasoner.add_evidence(evidence)

	def reset(self, reasoner_klass: Type[MBDReasoner] = None):
		# TODO: Make sure to reload the network!
		if reasoner_klass:
			self._init(reasoner_klass)  # either use the defined type of reasoner 
		elif self._reasoner:
			self._init(self._reasoner.__class__)  # or use the same type of reasoner as on the first use
		else:
			self._init()  # or use the default type of reasoner
		self.initialise_system_state()
		self._reasoner.add_targets(*{n.fqn for n in self._net.get_nodes_of_type(Hardware)}, reset=True)
		self._diagnostic_tests = {test.fqn: test for test in self._net.get_nodes_of_type(DiagnosticTest)}
		self._diagnostic_test_results = {result.fqn: result for test in self._diagnostic_tests.values() for result in
										 test.test_results}

		self._performed_diagnostic_tests = []
		self._total_diagnostic_cost = 0.

	def set_entropy_limit(self, limit: int):
		self._entropy_limit = limit

	def add_evidence(self, evidence: dict[str, Union[str, list[float]]]):
		# map evidence back to tests (which have been executed)
		tests_evidence = defaultdict(dict)
		for result_fqn, e in evidence.items():
			if result_fqn in self._diagnostic_test_results:
				dt = self._diagnostic_test_results[result_fqn]
				tests_evidence[dt.test][dt] = e

		# run test and insert evidence
		for dt, ev in tests_evidence.items():
			self.perform_diagnostic_test(dt, ev)
			for test_result in ev.keys():
				evidence.pop(test_result.fqn)

		# insert remaining evidence (e.g. direct observables)
		self._reasoner.add_evidence(evidence)

	def add_operating_mode(self, operating_modes: dict[str, str]):
		# filter knowledge graph based on the selected operating modes
		self._operating_modes.update(operating_modes)
		removed_nodes, _ = self._net.filter_operating_modes(list(self._operating_modes.values()))
		removed_nodes = {node.fqn for node in removed_nodes}

		# update the target (HW nodes) and re-generate the Bayes / Tensor net
		self._reasoner.drop_targets(*removed_nodes)
		self._reasoner.init()

		# update evidence
		evidence = dict()
		for n, ev in operating_modes.items():
			evidence[n] = [1. if s == ev else 0. for s in self._net.get_opm_node(n).states]
		self._reasoner.add_evidence(evidence)

		# filter the diagnostic tests which can be performed, remove any deleted tests
		self._diagnostic_tests = {k: v for k, v in self._diagnostic_tests.items() if k not in removed_nodes}

	def infer(self, *nodes: str) -> pd.DataFrame:
		df = self._reasoner.infer(*nodes)

		# sort nodes based on (1) rounded health value, (2) level, and (3) name
		df['Level'] = df.apply(lambda row: self._net.get_node(row.name).level, axis='columns')
		df['Health_rounded'] = round(df['Healthy'], ndigits=9)
		df.index.name = 'HW'
		df.sort_values(by=['Health_rounded', 'Level', 'HW'], inplace=True)
		df.index.name = None

		df.index = df.index.map(lambda x: self._net.get_node(x))
		return df.drop(columns=['Level', 'Health_rounded'])

	def compute_entropy_and_cost_of_diagnostic_tests(self, evidence: dict[str, str] = None,
													 limit: int = None) -> pd.DataFrame:
		evidence = evidence or self._reasoner.evidence

		if not self.get_unperformed_diagnostic_tests():
			return pd.DataFrame()

		relative_to: list[Hardware] = self._net.get_nodes_of_type(Hardware)
		relative_to_set = {node.fqn for node in relative_to}

		limit = limit or self._entropy_limit
		if limit:
			relative_to_set = set(node.fqn for node in self.infer().head(limit).index).intersection(relative_to_set)

		df = pd.DataFrame(
			[node for node in self._diagnostic_tests.values() if node not in self._performed_diagnostic_tests])

		df = df.rename({0: "Diagnostic Test"}, axis=1)
		df["Entropy"] = df.apply(lambda x: self._reasoner.conditional_entropy(relative_to_set, {r.fqn for r in x.iloc[0].test_results}, evidence),
								 axis=1)
		if not df.empty:
			df = df.sort_values(by="Entropy", axis=0, ascending=True)

		# Cost diagnosis starts here

		# Calculated like this, information gain can be negative, if measuring variable is expected
		# to increase uncertainty again. Therefore, we better deal with entropy directly to avoid 
		# dealing with negative numbers (and get problems with logarithms etc.).
		# base_entropy = self._reasoner.entropy(relative_to_set, evidence=evidence)
		# df["Information Gain"] = df.apply(lambda x: base_entropy - x["Entropy"], axis=1)

		df["Cost"] = df.apply(lambda x: self._system_state.calculate_cost_of_diagnostic_test(x["Diagnostic Test"]),
							  axis=1)
		df["Fixed Cost Dict"] = df.apply(lambda x: x["Diagnostic Test"].fixed_cost, axis=1)
		df["Dynamic Cost"] = df.apply(lambda x: x["Cost"] - x["Diagnostic Test"].get_fixed_cost(), axis=1)
		return df

	def compute_next_diagnostic_tests(self, evidence: dict[str, str] = None,
									  limit: int = None) -> pd.DataFrame:

		df = self.compute_entropy_and_cost_of_diagnostic_tests(evidence=evidence, limit=limit)
		self._computed_next_diagnostic_tests = df

		return self.sort_next_diagnostic_tests()

	def sort_next_diagnostic_tests(self) -> pd.DataFrame:
		df = self._computed_next_diagnostic_tests
		if df is None:
			return pd.DataFrame()

		# apply loss function on normalized (!) cost and entropy
		if self._loss_function and not df.empty:
			df["Cost_normalized"] = (df["Cost"] - df["Cost"].min()) / (df["Cost"].max() - df["Cost"].min())
			df['Entropy_rounded'] = round(df['Entropy'], ndigits=9)
			df["Entropy_normalized"] = (df["Entropy_rounded"] - df["Entropy_rounded"].min()) / (df["Entropy_rounded"].max() - df["Entropy_rounded"].min())
			df["Loss"] = df.apply(lambda x: self._loss_function(x["Entropy_normalized"], x["Cost_normalized"]), axis=1)
			df = df.sort_values(by=['Loss', 'Entropy_rounded', 'Cost', 'Diagnostic Test'])

		self._computed_next_diagnostic_tests = df
		return df

	def compute_next_diagnostic_test(self,
									 evidence: dict[str, str] = None,
									 limit: int = None) -> DiagnosticTest:
		df = self.compute_next_diagnostic_tests(evidence=evidence, limit=limit)
		return df.iloc[0]["Diagnostic Test"] if not df.empty else None

	def compute_loss_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		min_entropy = self._computed_next_diagnostic_tests["Entropy"].min()
		max_entropy = self._computed_next_diagnostic_tests["Entropy"].max()
		delta_entropy = max_entropy - min_entropy
		min_costs = self._computed_next_diagnostic_tests["Cost"].min()
		max_costs = self._computed_next_diagnostic_tests["Cost"].max()
		delta_costs = max_costs - min_costs

		x = np.linspace(min_entropy - 0.10 * delta_entropy, max_entropy + 0.10 * delta_entropy, 40)
		y = np.linspace(min_costs - 0.10 * delta_costs, max_costs + 0.10 * delta_costs, 40)
		zx, zy = np.meshgrid(x, y)
		entropy_norm = (zx - min_entropy) / (delta_entropy or 1.)
		costs_norm = (zy - min_costs) / (delta_costs or 1.)
		loss = self._loss_function(entropy_norm, costs_norm)
		return x, y, loss

	def get_unperformed_diagnostic_tests(self) -> list[DiagnosticTest]:
		# We have to compute the names here because for simulations we create
		# deep copies of the diagnoser in each step. These copies contain themselves
		# copies of the DiagnosticTests, i.e. these are new objects. Therefore, 
		# directly comparing the DiagnosticTests doesn't work; we must compare on 
		# names to get correct results.
		return list(set(self._diagnostic_tests.values()).difference(self._performed_diagnostic_tests))

	def get_direct_observables(self) -> pd.DataFrame:
		df = pd.DataFrame([node for node in self._net.get_nodes_of_type(DirectObservable) if node.fqn not in self._reasoner.evidence])
		df = df.rename({0: 'Direct Observable'}, axis=1)
		if not df.empty:
			df = df.sort_values(by=['Direct Observable'])
		return df

	def get_operating_modes(self) -> pd.DataFrame:
		df = pd.DataFrame([node for node in self._net.get_all_opm_nodes().values() if node.fqn not in self._reasoner.evidence])
		df = df.rename({0: 'Operating mode'}, axis=1)
		if not df.empty:
			df = df.sort_values(by=['Operating mode'])
		return df

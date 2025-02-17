# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from __future__ import annotations  # PEP 563 - Postponed Evaluation of Annotations - https://peps.python.org/pep-0563/
import io
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union

from mbdlyb.functional import (
	DirectObservable,
	DiagnosticTest,
	DiagnosticTestResult,
	FunctionalNode,
	Function,
	Hardware,
	Cluster
)
from mbdlyb.formalisms import select_reasoner, TensorNetReasoner, BayesNetReasoner, MarkovNetReasoner


class Analyzer:
	_operating_modes: dict[str, str] = None
	_cached_failure_obs: dict[frozenset[FunctionalNode], pd.DataFrame] = None

	_reasoner: Union[TensorNetReasoner, BayesNetReasoner, MarkovNetReasoner] = None
	_diagnostic_test_results: list[DiagnosticTestResult] = None
	_direct_observables: list[DirectObservable] = None

	dps: dict[str, pd.DataFrame] = None
	dptrees: dict[str, DPTreeNode] = None
	overview: pd.DataFrame = None
	signatures: pd.DataFrame = None
	costs: pd.DataFrame = None

	def __init__(self, cluster: Cluster, reasoner_klass: Union[TensorNetReasoner, BayesNetReasoner, MarkovNetReasoner] = None):
		self._cluster = cluster
		self._operating_modes = dict()
		self._reasoner = select_reasoner(self._cluster, reasoner_klass)
		self._cached_failure_obs = dict()

	def _group_by_dp(self, failure_obs: pd.DataFrame) -> pd.DataFrame:
		"""Groups failure observabilty by diagnostic procedure (DP)."""
		grouped_by_dp = failure_obs[self._direct_observables].reset_index(names=["Node", "Failure"])
		if self._direct_observables:
			grouped_by_dp = (
				grouped_by_dp.groupby(failure_obs[self._direct_observables].columns.to_list())
				.agg(list)
				.sort_values("Node", ascending=False, key=lambda x: x.str.len())
				.reset_index()
				.rename(index=lambda x: f"DP_{x + 1}")
			)
		else:
			grouped_by_dp.insert(0, "DP", "DP_1")
			grouped_by_dp = (
				grouped_by_dp.groupby("DP")
				.agg(list)
				.reset_index()
				.rename(index=grouped_by_dp["DP"])
				.drop("DP", axis=1)
			)
		return grouped_by_dp

	def _infer_discrete(self) -> pd.Series:
		posteriors = self._reasoner.infer()
		beliefs: dict[FunctionalNode, tuple[str, float]] = dict()
		for observable, obs_posteriors in posteriors.iterrows():
			beliefs[observable] = max(obs_posteriors.items(), key=lambda x: x[1])[0]
		return pd.Series(beliefs)

	def compute_failure_observability(self) -> pd.DataFrame:
		"""
		Computes failure observability of failure modes based on a given list
		of observables.
		"""
		observables = [*self._diagnostic_test_results, *self._direct_observables]
		observable_fqns = [o.fqn for o in observables]

		if frozenset(observables) in self._cached_failure_obs:
			return self._cached_failure_obs[frozenset(observables)]

		hardware_nodes: list[Hardware] = self._cluster.get_nodes_of_type(Hardware)
		base_evidence = {n.fqn: "Healthy" for n in hardware_nodes}
		elementary_hardware_nodes = [hwn for hwn in hardware_nodes if hwn.is_elementary]

		self._reasoner.add_targets(*observable_fqns, reset=True)
		failure_obs = pd.DataFrame(
			columns=observable_fqns,
			index=pd.MultiIndex.from_tuples([], names=["Node", "State"])
		)

		for n in elementary_hardware_nodes:
			for failure in [state for state in n.states if state != "Healthy"]:
				self._reasoner.add_evidence(
					{**base_evidence, n.fqn: failure}
				)
				failure_obs.loc[(n, failure), :] = self._infer_discrete()

		failure_obs = failure_obs.rename(
			lambda x: self._cluster.get_node(x), axis=1
		)
		self._cached_failure_obs[frozenset(observables)] = failure_obs
		return failure_obs

	@staticmethod
	def _compute_combined(array: np.ndarray) -> np.ndarray:
		"""Computes a combined array where each column is weighted by a
		factor based on the number of unique values in that column."""
		factor = 1
		combined = np.zeros(array.shape[1])
		for col in array[::-1]:
			combined += col * factor
			factor *= len(np.unique(col))
		return combined

	@staticmethod
	def _signatures_to_ints(signatures: pd.DataFrame) -> np.ndarray:
		"""Converts categorical signature strings in a DataFrame to
		NumPy array with integer codes."""
		return signatures.apply(
			lambda col: col.astype("category").cat.codes
		).T.to_numpy()

	def compute_dissimilarity_value(self, sign: pd.DataFrame) -> int:
		"""Computes a dissimilarity value based on the dissimilarity of combined
		signature values in the given DataFrame."""
		array = self._signatures_to_ints(sign)
		combined = self._compute_combined(array)
		similarity_matrix = (combined[:, None] != combined).astype(int)
		return np.sum(similarity_matrix)

	@staticmethod
	def _compute_test_losses(
			costs: list, dissimilarity_vals: list, balance: float = 0.5
	) -> pd.Series:
		"""Computes test losses as a weighted combination of normalized
		costs and dissimilarity values."""
		costs = pd.Series(costs) / max(costs)
		dissimilarity_vals = pd.Series(dissimilarity_vals) / max(dissimilarity_vals)
		return (1 - balance) * costs + balance * (1 - dissimilarity_vals)

	def _sort_and_filter_dp(self, dp: pd.DataFrame) -> pd.DataFrame:
		"""Sorts DiagnosticTestResult nodes by computed test loss within
		diagnostic procedure. Filters out nodes without discriminative power
		and removes duplicates based on signatures."""
		if len(dp) > 1:  # drop results with same signature for failure modes
			dp = dp.loc[:, dp.nunique() > 1]

		if dp.empty:  # return if dp has no tests
			return dp

		df = pd.DataFrame(index=dp.columns)
		df["Test"] = df.index.to_series().apply(lambda x: x.test)
		grouped_by_test = df.reset_index(names="Results").groupby("Test")

		dissimilarity_values = []
		costs = []
		for test, df_test in grouped_by_test:
			sign = dp[df_test["Results"]]
			dissimilarity_values.append(self.compute_dissimilarity_value(sign))
			costs.append(sum(test.fixed_cost.values()))

		losses = dict(zip(
			grouped_by_test.groups.keys(),
			self._compute_test_losses(costs, dissimilarity_values)
		))

		dp_t = dp.T
		dp_t["Test"] = dp_t.index.map(df["Test"].to_dict())
		dp_t["Loss"] = dp_t.index.map(df["Test"].map(losses).to_dict())
		dp_sorted = dp_t.sort_values(["Loss", "Test"]).drop(["Loss", "Test"], axis=1)
		return dp_sorted.drop_duplicates().T

	def compute_diagnostic_procedures(
			self, failure_obs: pd.DataFrame
	) -> dict[str, pd.DataFrame]:
		"""Computes diagnostic procedures (DP)."""
		grouped_by_dp = self._group_by_dp(failure_obs)

		dps = dict()
		without_dps = pd.DataFrame()

		num_groups = len(grouped_by_dp)
		for _, row in grouped_by_dp.iterrows():
			failures = list(zip(row["Node"], row["Failure"]))
			dp = failure_obs.loc[failures, self._diagnostic_test_results]
			dp = self._sort_and_filter_dp(dp)

			# handle FMs without tests
			if dp.empty:
				dp["Diagnosable"] = len(dp.index) == 1
				without_dps = pd.concat([without_dps, dp])
				continue

			# if only one failure mode, no tests needed
			if len(dp) == 1:
				dp = dp.drop([c for c in dp.columns], axis=1)
				dp["Diagnosable"] = True

			dp["Diagnosable"] = ~dp.duplicated(keep=False)
			dps[f"DP_{str(len(dps) + 1).zfill(len(str(num_groups)))}"] = dp

		if not without_dps.index.empty:
			dps["No_DP"] = without_dps

		return dps

	@staticmethod
	def extract_tests_from_dps(dps: pd.DataFrame) -> list[DiagnosticTest]:
		tests = list()
		result: DiagnosticTestResult
		for result in dps.columns[:-1]:
			if result.test not in tests:
				tests.append(result.test)
		return tests

	def get_node(self, test_fqn: str) -> FunctionalNode:
		return self._cluster.get_node(test_fqn)

	@staticmethod
	def _get_ufm_clusters(dp: pd.DataFrame) -> dict:
		"""
		Groups unidentifiable failure modes in diagnostic procedure (DP)
		into clusters.
		"""
		ufms = dp[dp["Diagnosable"] == False]
		ufms_grouped = (
			ufms.reset_index(names=["Node", "Failure"])
			.groupby(ufms.columns.to_list())
			.agg(list)
		)

		ufms_grouped = [
			[f"{node} {fm}" for node, fm in zip(r["Node"], r["Failure"])]
			for _, r in ufms_grouped[ufms_grouped["Node"].apply(len) > 1].iterrows()
		]

		return {
			f"Cluster {idx + 1}": ", ".join(c)
			for idx, c in enumerate(ufms_grouped)
		}

	def analyze(self) -> None:
		self._diagnostic_test_results = sorted(self._cluster.get_nodes_of_type(DiagnosticTestResult))
		self._direct_observables = sorted(self._cluster.get_nodes_of_type(DirectObservable))

		failure_obs = self.compute_failure_observability()
		self.dps = self.compute_diagnostic_procedures(failure_obs)
		self.dptrees = dict()

		# costs
		costs = pd.DataFrame(index=failure_obs.index)
		for name, dp in self.dps.items():
			dps = self.dps[name]
			tests = self.extract_tests_from_dps(dps)
			cost_lookup = dict()
			self.dptrees[name] = self._build_diagnostic_procedure_tree(dps, tests, cost_lookup)

			fms = dp.index
			cost = fms.map(lambda x: cost_lookup[x])
			prior = fms.map(lambda x: x[0].fault_rates[x[1]])
			costs.loc[fms, "DP"] = name
			costs.loc[fms, "Cost"] = cost
			costs.loc[fms, "Prior"] = prior
			costs.loc[fms, "WeightedCost"] = prior * cost
		self.costs = costs

		# signatures
		self.signatures = pd.DataFrame(self.costs[["DP", "Cost"]])
		self.signatures[self._direct_observables] = failure_obs[self._direct_observables]

		# overview
		overview = pd.DataFrame(columns=["#FMs", "#Tests", "#Diagnosable"])
		for name, dp in self.dps.items():
			overview.loc[name] = {
				"#FMs": len(dp),
				"#Tests": len(dp.columns) - 1,
				"#Diagnosable": sum(dp["Diagnosable"]),
			}
			for clust, nodes in self._get_ufm_clusters(dp).items():
				overview.loc[name, clust] = nodes
		self.overview = overview

	def add_operating_mode(self, operating_modes: dict[str, str]):
		# filter knowledge graph based on the selected operating modes
		self._operating_modes.update(operating_modes)
		self._cluster.filter_operating_modes(list(self._operating_modes.values()))

	@property
	def selected_operating_modes(self) -> dict[str, str]:
		return self._operating_modes

	def _write_to_excel(self, file):
		opm_rows = [[opm, self._operating_modes[opm]] for opm in sorted(self._operating_modes)]
		unselected_opm = [opm for opm in self._cluster.get_all_opm_nodes().values() if opm.fqn not in self._operating_modes]
		opm_rows += [[opm.fqn, ' | '.join(sorted(opm.states))] for opm in sorted(unselected_opm, key=lambda opm: opm.fqn)]
		operating_modes_df = pd.DataFrame(opm_rows, columns=['Operating mode selector', 'Selected mode'])

		with pd.ExcelWriter(file) as writer:
			self.overview.to_excel(writer, sheet_name="Overview")
			operating_modes_df.to_excel(writer, sheet_name="Operating modes", index=False)
			self.signatures.to_excel(writer, sheet_name="Signatures")
			self.costs.to_excel(writer, sheet_name="Costs")
			for dp_name, dp_table in self.dps.items():
				dp_table.to_excel(writer, sheet_name=dp_name)

	def to_excel(self, xlsx_file: Path) -> None:
		"""Write analyzer results to Excel file."""
		self._write_to_excel(xlsx_file)

	def export(self) -> bytes:
		"""Export analyzer results to Excel file in bytes array."""
		bytes = io.BytesIO()
		self._write_to_excel(bytes)
		bytes.seek(0,0)
		return bytes.read()

	def get_diagnostic_procedure_tree(self, dp_name: str) -> DPTreeNode:
		return self.dptrees[dp_name]

	def _build_diagnostic_procedure_tree(self, dps: dict[str, pd.DataFrame], tests: list[DiagnosticTest], cost_lookup: dict,
									  prev_node: Optional[DPTreeNode] = None, edge: Optional[DPTreeEdge] = None, costs: float = 0) -> DPTreeNode:
		if len(tests) > 0:  # add test nodes
			tst = tests[0]
			dps_results = dps[[tr for tr in tst.test_results if tr in dps]]
			indices = dps_results.value_counts().sort_index(ascending=False)

			# only add a test-node if it has more than one possible outcomes
			if len(indices) > 1:
				new_node = DPTreeNode(tst.name, DPTreeNode.CLASS_MAP[tst.__class__], tst.get_fixed_cost())
				if edge:
					edge.set_target(new_node)
				costs += tst.get_fixed_cost()
			else:
				new_node = prev_node

			# connect to next tests
			for index in indices.keys():
				new_results = dps
				for result_name, value in zip(dps_results, index):
					new_results = new_results[new_results[result_name] == value]
				if len(indices) > 1:
					edge = DPTreeEdge([(result.name, value) for result, value in zip(dps_results, index)])
					new_node.add_edge(edge)
				self._build_diagnostic_procedure_tree(new_results, tests[1:], cost_lookup, new_node, edge, costs)

		else:  # add hardware nodes
			for i in dps.index:
				cost_lookup[i] = costs
			hw_nodes = sorted([node for (node,_) in dps.index])
			hw_names = '\n'.join([hw.name for hw in hw_nodes])
			type = DPTreeNode.CLASS_MAP[hw_nodes[0].__class__] if len(hw_nodes) == 1 else 'NonDiagnosableHW'
			new_node = DPTreeNode(hw_names, type, costs)
			if edge:
				edge.set_target(new_node)
		return new_node


class DPTreeNode:
	name: str
	type: str
	cost: float
	edges: list[DPTreeEdge]

	CLASS_MAP: dict[FunctionalNode, str] = {
		Function: 'Function',
		Hardware: 'Hardware',
		DirectObservable: 'DirectObservable',
		DiagnosticTest: 'DiagnosticTest',
		DiagnosticTestResult: 'DiagnosticTestResult',
	}

	@classmethod
	def get_color_mapping(cls) -> dict[str,str]:
		color_map = {node_name: node_type.get_color() for node_type, node_name in cls.CLASS_MAP.items()}
		return color_map | {'NonDiagnosableHW': 'red'}

	def __init__(self, name: str, type:str, cost: float):
		self.name = name
		self.type = type
		self.cost = cost
		self.edges = []

	def add_edge(self, edge: DPTreeEdge):
		self.edges.append(edge)


class DPTreeEdge:
	results: list[tuple[str,str]]
	target: Optional[DPTreeNode]

	def __init__(self, results: list[tuple[str,str]]):
		self.results = results
		self.target = None

	def set_target(self, target: DPTreeNode):
		self.target = target

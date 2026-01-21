# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""
from typing import Union, Optional

import numpy as np
import networkx as nx

from mbdlyb import MBDNode, MBDNet, MBDRelation, MBDNetReasonerView, MBDElement, longest_common_fqn
from mbdlyb.operating_mode import OpmLogic, OpmLogicAnd, OpmLogicOr, OpmSet
from mbdlyb.formalisms.bayesnet import BayesNode, BayesAutoSplitNode, BayesNet, BNRelation
from mbdlyb.formalisms.tensor_network import TensorNode, TensorAutoSplitNode, TensorNet, TNRelation
from mbdlyb.formalisms.markovnet import MarkovNode, MarkovAutoSplitNode, MarkovNet, MNRelation


class FunctionalNode(MBDNode):
	DEFAULT_STATES: list[str] = []
	_EQ_MAP = {
		'Hardware': 'Healthy',
		'Function': 'Ok',
	}

	_operating_modes: OpmSet = None

	parent_relations: set['FunctionalRelation'] = None
	child_relations: set['FunctionalRelation'] = None

	@property
	def operating_modes(self) -> OpmSet:
		return self._operating_modes

	def set_operating_modes(self, operating_modes: OpmSet):
		self._operating_modes = operating_modes


class OperatingMode(BayesNode, TensorNode, MarkovNode, FunctionalNode):
	_TYPE_NAME = 'Select'
	_color = '#FF99FF'

	opm_set: OpmSet = None

	def __init__(self, name: str, operating_modes: list[str], cluster: 'Cluster'):
		self.opm_set = OpmSet(operating_modes)
		super().__init__(name, operating_modes, cluster)

	def _compute_cpt_line(self, values: dict[MBDNode, str]) -> list[float]:
		return [1. / len(self.states)] * len(self.states)

	def _compute_array_line(self, values: dict[MBDNode, str]) -> list[float]:
		return self._compute_cpt_line(values)

	def _compute_factor_line(self, values: dict[FunctionalNode, str]) -> list[float]:
		return self._compute_cpt_line(values)


class Function(BayesAutoSplitNode, TensorAutoSplitNode, MarkovAutoSplitNode, FunctionalNode):
	DEFAULT_STATES = ['Ok', 'NOk']
	_TYPE_NAME = 'Function'
	_color = '#CC99FF'
	_false_positive_weight: float = 0.
	_split_preserve_classes = [OperatingMode]

	_EQ_MAP = {
		'Hardware': 'Healthy',
		'Function': 'Ok'
	}

	def __init__(self, name: str, states: list[str], cluster: 'Cluster'):
		super().__init__(name, states, cluster)
		self._false_positive_weight = 0.

	def _compute_cpt_line(self, values: dict[FunctionalNode, str]) -> list[float]:
		# split incoming values in operating modes and parent nodes
		active_opmodes = [v for (k, v) in values.items() if isinstance(k, OperatingMode)]
		parent_nodes = {k: v for (k, v) in values.items() if not isinstance(k, OperatingMode)}

		# split parent nodes over operating modes
		result = 1.
		for pn, val in parent_nodes.items():
			# find the relation to this parent node
			rel = [p for p in self.parent_relations if p.source == pn][0]
			weight = rel.weight
			if not isinstance(weight, float):
				print(f'Weight is {weight.__class__.__name__} for {rel}!')
			fp = rel.source._false_positive_weight if isinstance(rel.source, Function) else 0.
			# check if value is a pass or fail
			value_is_ok = self._EQ_MAP.get(pn.__class__.__name__) == val

			# update result if relation to parent has an operating mode which is active, or has no operating mode
			if (rel.operating_modes and rel.operating_modes.eval(active_opmodes)) or (not rel.operating_modes):
				if value_is_ok:
					result *= (1. - fp)
				else:
					result *= (1. - weight)
		return [result, 1. - result]

	def _compute_array_line(self, values: dict[FunctionalNode, str]) -> list[float]:
		return self._compute_cpt_line(values)

	def _compute_factor_line(self, values: dict[FunctionalNode, str]) -> list[float]:
		return self._compute_cpt_line(values)

	def _compute_join_cpt(self, split_count: int) -> np.ndarray:
		return np.reshape([1., 0.] + [0., 1.] * (pow(2, split_count) - 1), [2] * (1 + split_count))

	def _compute_join_array(self, split_count: int) -> np.ndarray:
		return self._compute_join_cpt(split_count)

	def _compute_join_factor(self, split_count: int) -> np.ndarray:
		return self._compute_join_cpt(split_count)

	@property
	def has_subfunctions(self) -> bool:
		return any(isinstance(r, SubfunctionOfRelation) for r in self.parent_relations)

	@property
	def subfunctions(self) -> list['Function']:
		return [r.source for r in self.parent_relations if isinstance(r, SubfunctionOfRelation)]

	@property
	def has_superfunctions(self) -> bool:
		return any(isinstance(r, SubfunctionOfRelation) for r in self.child_relations)

	@property
	def superfunctions(self) -> list['Function']:
		return [r.target for r in self.child_relations if isinstance(r, SubfunctionOfRelation)]

	def superfunction_chain(self, as_fqn: bool = False) -> list[list[Union['Function', str]]]:
		return [[sf.fqn if as_fqn else sf] + sf.superfunction_chain(as_fqn) for sf in self.superfunctions]


class Hardware(BayesNode, TensorNode, MarkovNode, FunctionalNode):
	DEFAULT_STATES = ['Healthy']
	_TYPE_NAME = 'Hardware'
	_color = '#FF9900'

	fault_rates: dict[str, float] = None

	def __init__(self, name: str, fault_rates: dict[str, float], cluster: 'Cluster'):
		super().__init__(name, ['Healthy'] + list(fault_rates.keys()), cluster)
		self.fault_rates = fault_rates.copy()

	def _compute_cpt_line(self, values: dict[MBDNode, str]) -> list[float]:
		fr = {'Healthy': 1. - sum(self.fault_rates.values()), **self.fault_rates}
		return [fr[f] for f in self._states]

	def _compute_array_line(self, values: dict[MBDNode, str]) -> list[float]:
		return self._compute_cpt_line(values)

	def _compute_factor_line(self, values: dict[FunctionalNode, str]) -> list[float]:
		return self._compute_cpt_line(values)

	@property
	def is_elementary(self) -> bool:
		return not any(r.target.has_subfunctions for r in self.child_relations if isinstance(r, RealizesRelation))


class ObservableNode(BayesNode, TensorNode, MarkovNode, FunctionalNode):
	DEFAULT_STATES = ['Ok', 'NOk']
	_TYPE_NAME = 'Observable'

	fp_rate: float = 0.
	fn_rate: float = 0.

	def __init__(self, name: str, states: list[str], fp_rate: float, fn_rate: float, cluster: 'Cluster'):
		super().__init__(name, states, cluster)
		self.fp_rate = fp_rate
		self.fn_rate = fn_rate

	def _compute_cpt_line(self, values: dict[MBDNode, str]) -> list[float]:
		result = 1.
		for pn, val in values.items():
			# find the relation to this parent node
			rel = [p for p in self.parent_relations if p.source == pn][0]
			weight = rel.weight
			fp = rel.source._false_positive_weight if isinstance(rel.source, Function) else 0.
			# check if value is a pass or fail
			value_is_ok = self._EQ_MAP.get(pn.__class__.__name__) == val
			# update result
			if value_is_ok:
				result *= (1. - fp)
			else:
				result *= (1. - weight)
		# combine connection weight (results) with fp_rate and fn_rate
		return [result * (1. - self.fn_rate) + (1. - result) *       self.fp_rate,
		        result *       self.fn_rate  + (1. - result) * (1. - self.fp_rate)]

	def _compute_array_line(self, values: dict[MBDNode, str]) -> list[float]:
		return self._compute_cpt_line(values)

	def _compute_factor_line(self, values: dict[FunctionalNode, str]) -> list[float]:
		return self._compute_cpt_line(values)


class DirectObservable(ObservableNode):
	_TYPE_NAME = 'DirectObservable'
	_color = '#CCFFCC'

	def __init__(self, name: str, states: list[str], fp_rate: float, fn_rate: float, cluster: 'Cluster'):
		super().__init__(name, states, fp_rate, fn_rate, cluster)


class ObservedError(DirectObservable):
	DEFAULT_STATES = ['Absent', 'Present']
	_TYPE_NAME = 'ErrorMessage'
	_color = '#FFFF99'

	error_code: str
	observation_path: set['RequiredForRelation']

	def __init__(self, name: str, states: list[str], fp_rate: float, fn_rate: float, error_code: str,
				 cluster: 'Cluster'):
		super().__init__(name, states, fp_rate, fn_rate, cluster)
		self.error_code = error_code
		self.observation_path = set()

	def _compute_cpt_line(self, values: dict[MBDNode, str]) -> list[float]:
		if all(v == 'Ok' for p, v in values.items() if p in self.enabling_functions):
			if values[self.observed_function] == 'NOk':
				return [self.fp_rate, 1. - self.fp_rate]
			else:
				return [1. - self.fn_rate, self.fn_rate]
		return [1., 0.]

	def _compute_array_line(self, values: dict[MBDNode, str]) -> list[float]:
		return self._compute_cpt_line(values)

	def _compute_factor_line(self, values: dict[FunctionalNode, str]) -> list[float]:
		return self._compute_cpt_line(values)

	@property
	def observed_function(self) -> Function:
		return [pr for pr in self.parent_relations if isinstance(pr, YieldsErrorRelation)][0].source

	@property
	def enabling_functions(self) -> list[Function]:
		return [pr.source for pr in self.parent_relations if isinstance(pr, ReportsErrorRelation)]


class DiagnosticTest(FunctionalNode):
	_TYPE_NAME = 'DiagnosticTest'
	_color = '#99CCFF'

	fixed_cost: dict[str, float] = None
	preconditions: dict[str, str] = None

	def __init__(self, name: str, fixed_cost: dict[str, float], preconditions: dict[str, str], cluster: 'Cluster'):
		super().__init__(name, ['Ok', 'NOk'], cluster)
		self.fixed_cost = fixed_cost
		self.preconditions = preconditions

	def get_fixed_cost(self, cost_type: Optional[str] = None) -> float:
		"""
		Fixed cost of the service action. Returns sum of fixed costs if there are
		multiple types of fixed costs.
		"""

		if cost_type is not None:
			return self.fixed_cost[cost_type] if cost_type in self.fixed_cost else 0.
		return sum(self.fixed_cost.values())

	@property
	def test_results(self) -> list['DiagnosticTestResult']:
		return [cr.target for cr in self.child_relations if isinstance(cr, ResultsInRelation)]


class DiagnosticTestResult(ObservableNode):
	_TYPE_NAME = 'DiagnosticTestResult'
	_color = '#339966'

	def __init__(self, name: str, states: list[str], fp_rate: float, fn_rate: float, cluster: 'Cluster'):
		super().__init__(name, states, fp_rate, fn_rate, cluster)

	@property
	def test(self) -> DiagnosticTest:
		return [r.source for r in self.parent_relations if isinstance(r, ResultsInRelation)][0]


class Cluster(BayesNet, TensorNet, MarkovNet, MBDNet):
	_TYPE_NAME = 'Cluster'

	nodes: dict[str, Union[FunctionalNode, 'Cluster']] = None
	_opmsets: dict[OpmSet, MBDNode] = None

	relations: set['FunctionalRelation'] = None

	def __init__(self, name: str, cluster: 'Cluster' = None):
		super().__init__(name, cluster)
		self._opm_nodes = dict()
		self._opmsets = dict()

	def add_node(self, element: Union[MBDNode, 'MBDNet']):
		if isinstance(element, OperatingMode):
			self._opmsets[element.opm_set] = element
		super().add_node(element)

	def get_node(self, fqn: str | list[str], skipped_root: bool = False) -> MBDElement:
		if not fqn:
			return self
		if isinstance(fqn, str):
			fqn = fqn.split('.')
		if not skipped_root and self.at_root and self.name == fqn[0]:
			return self.get_node(fqn[1:], True)
		if fqn[0] not in self.nodes:
			raise KeyError(f'Node {fqn[0]} not found in {self.fqn}!')
		n = self.nodes[fqn[0]]
		return n.get_node(fqn[1:]) if fqn[1:] else n

	def get_opm_node(self, fqn: str) -> Optional[MBDNode]:
		candidate_nodes = {n.fqn: n for n in self.get_all_opm_nodes().values()}
		return candidate_nodes.get(fqn, None)

	def get_opm_node_from_set(self, opm_set: OpmSet) -> Optional[MBDNode]:
		return self._opmsets.get(opm_set, None)

	def get_all_opm_nodes(self) -> dict[OpmSet, MBDNode]:
		return {**self._opmsets,
				**{k: v for subnet in self.subnets.values() for k, v in subnet.get_all_opm_nodes().items()}}

	def create_reasoner_view(self) -> MBDNetReasonerView:
		return ClusterReasonerView(self)

	def propagate_operating_modes(self, verbose=False):
		if not self.get_nodes_of_type(OperatingMode):
			return  # Skip if no Operating Mode nodes are in the network
		function_nodes: list[Function] = self.get_nodes_of_type(Function)

		# Operating mode propagation - step 1
		#   - propagation between function nodes via required-for relations
		#   - propagation from function to hardware nodes via realizes- and affects-relations
		#   - propagation between functions via subfunction-of relations
		while len(function_nodes) > 0:
			# find all nodes for which the operating mode on the incoming edges can be inferred
			nodes_to_infer = {node for node in function_nodes if
							  all([rel.inferred_operating_modes is not None for rel in node.child_relations if
								   isinstance(rel, RequiredForRelation)]) and all(
								  [rel.inferred_operating_modes is not None for rel in node.parent_relations if
								   isinstance(rel, SubfunctionOfRelation)])}
			# also infer operating modes if one of the child required-for relations has no operating mode
			nodes_to_infer |= {node for node in function_nodes if
							   any([rel.inferred_operating_modes is not None and not rel.inferred_operating_modes for
									rel in node.child_relations if isinstance(rel, RequiredForRelation)])}
			for node in nodes_to_infer:
				# compute the combined operating mode of all incoming required-for relations and all outgoing subfunction-of relations
				opmode_children = OpmLogicOr()
				relations = [rel for rel in node.child_relations if isinstance(rel, RequiredForRelation)] \
				          + [rel for rel in node.parent_relations if isinstance(rel, SubfunctionOfRelation)]
				for rel in relations:
					# if one of the relations has no operating modes (i.e. always active), do not propagate operating modes
					if not rel.inferred_operating_modes:
						opmode_children = OpmLogicOr()
						break
					else:
						opmode_children.add(rel.inferred_operating_modes)

				# set the operating mode on all outgoing affect, realizes and required-for relations
				for parent_rel in node.parent_relations:
					if parent_rel.__class__ not in {AffectsRelation, RealizesRelation, RequiredForRelation}:
						continue
					if parent_rel.operating_modes is not None:
						opmode = OpmLogicAnd().add(parent_rel.operating_modes).add(opmode_children)
					else:
						opmode = OpmLogic.from_copy(opmode_children)
					parent_rel.set_inferred_operating_modes(opmode)

				# set the operating mode on all outgoing subfunction-of, yields-error and communicates-through relations
				for child_rel in node.child_relations:
					if not isinstance(child_rel, (SubfunctionOfRelation, YieldsErrorRelation, CommunicatesThroughRelation)):
						continue
					opmode = OpmLogic.from_copy(opmode_children)
					child_rel.set_inferred_operating_modes(opmode)
				function_nodes.remove(node)

		# Operating mode propagation - step 2
		# propagation from function and hardware nodes to diagnostic tests and direct observables
		for node in self.get_nodes_of_type(DiagnosticTestResult, DirectObservable):
			if isinstance(node, ObservedError):
				continue
			for parent_rel in node.parent_relations:
				opmode = OpmLogicOr()
				for rel in parent_rel.source.child_relations:
					if rel.__class__ not in {AffectsRelation, RealizesRelation, RequiredForRelation}:
						continue
					opmode.add(rel.inferred_operating_modes)
				for rel in parent_rel.source.parent_relations:
					if not isinstance(rel, SubfunctionOfRelation):
						continue
					opmode.add(rel.inferred_operating_modes)
				parent_rel.set_inferred_operating_modes(opmode)

		# Operating mode propagation - step 3
		# propagation from yields-error relation to reports-error relations on observed errors
		for node in self.get_nodes_of_type(ObservedError):
			opmode = OpmLogicOr()
			re_relations: set[ReportsErrorRelation] = set()
			for parent_rel in node.parent_relations:
				if isinstance(parent_rel, YieldsErrorRelation):
					opmode.add(parent_rel.inferred_operating_modes)
				elif isinstance(parent_rel, ReportsErrorRelation):
					re_relations.add(parent_rel)
			for parent_rel in re_relations:
				parent_rel.set_inferred_operating_modes(opmode)

		# Operating mode propagation - step 4
		# propagation from IndicatedRelation of TestResultNode to ResultsRelation of TestResultNode
		for node in self.get_nodes_of_type(DiagnosticTestResult):
			opmode = OpmLogicOr()
			for rel in [rel for rel in node.relations if isinstance(rel, IndicatedByRelation)]:
				if not rel.inferred_operating_modes:
					opmode = OpmLogicOr()
					break
				else:
					opmode.add(rel.inferred_operating_modes)

			for rel in [rel for rel in node.relations if isinstance(rel, ResultsInRelation)]:
				rel.set_inferred_operating_modes(opmode)

		# print all inferred operating modes on request
		if verbose:
			for rel in self.get_flat_relations():
				print(rel, '({opms})'.format(opms=rel.inferred_operating_modes))

	def filter_operating_modes(self, operating_modes) -> tuple[set[MBDNode], set['FunctionalRelation']]:
		removed_nodes: set[FunctionalNode] = set()
		removed_relations: set[FunctionalRelation] = set()

		# step 1: remove all edges which ar disabled due to selected operating modes
		for rel in self.get_flat_relations():
			if rel.inferred_operating_modes is not None:
				if not rel.inferred_operating_modes.eval(operating_modes):
					removed_relations.add(rel)
					rel.remove()

		# step 2: remove all test-result nodes which have a removed indicated-by or yields-error relation
		for node in [rel.target for rel in removed_relations if
					 isinstance(rel, (IndicatedByRelation, YieldsErrorRelation))]:
			if node not in removed_nodes:
				removed_nodes.add(node)
				node.remove()

		# step 3: remove all function nodes which have a removed sub-function relation and no remaining sub-function relations
		for node in [rel.target for rel in removed_relations if isinstance(rel, SubfunctionOfRelation)]:
			if node not in removed_nodes and not [rel for rel in node.parent_relations if
												  isinstance(rel, SubfunctionOfRelation)]:
				removed_nodes.add(node)
				node.remove()

		# step 4: remove all nodes without any parent or child relations
		for node in self.get_flat_nodes():
			if len(node.relations) == 0:
				removed_nodes.add(node)
				node.remove()

				# also remove a subnet if it contains no more nodes
				subnet = node.net
				if len(subnet.nodes) == 0 and subnet.net is not None:
					subnet.remove()

		# step 5: remove all diagnostic-test nodes which have no results-in relations
		for node in self.get_nodes_of_type(DiagnosticTest):
			if len(node.child_relations) == 0:
				removed_nodes.add(node)
				node.remove()

				# also remove a subnet if it contains no more nodes
				subnet = node.net
				if len(subnet.nodes) == 0 and subnet.net is not None:
					subnet.remove()

		return removed_nodes, removed_relations


class FunctionalRelation(MBDRelation):
	_TYPE_NAME = 'FunctionalRelation'

	weight: float = 1.

	_operating_modes: OpmLogic = None
	_inferred_opm: Optional[OpmLogic] = None

	def __init__(self, source: MBDNode, target: MBDNode, weight: float = 1., net: 'Cluster' = None):
		super().__init__(source, target, net)
		self.weight = weight
		self._operating_modes = OpmLogic()
		self._inferred_opm = None

	@property
	def label(self) -> str:
		if self._label is None:
			raise NotImplementedError(f'Label for {self.__class__.__name__} is undefined!')
		if self._operating_modes:
			return f'{self._label} [{self._operating_modes}]'
		elif self._inferred_opm:
			return f'{self._label} [[{self._inferred_opm}]]'
		else:
			return self._label

	@property
	def operating_modes(self) -> OpmLogic:
		return self._operating_modes

	@property
	def inferred_operating_modes(self) -> OpmLogic:
		return self._inferred_opm

	def set_inferred_operating_modes(self, operating_modes: OpmLogic):
		self._inferred_opm = operating_modes


class RealizesRelation(BNRelation, TNRelation, MNRelation, FunctionalRelation):
	_TYPE_NAME = 'RealizesRelation'
	_label = 'realizes'
	_color = '#FF9900'

	source: Hardware
	target: Function


class RequiredForRelation(BNRelation, TNRelation, MNRelation, FunctionalRelation):
	_TYPE_NAME = 'RequiredForRelation'
	_label = 'required_for'
	_color = '#CC99FF'

	source: Function
	target: Function

	error_codes: list[ObservedError]

	def __init__(self, source: MBDNode, target: MBDNode, operating_modes: OpmLogic = OpmLogic(),
				 error_codes: list[ObservedError] = None, weight: float = 1., net: Cluster = None):
		super().__init__(source, target, weight, net)
		self._operating_modes = operating_modes
		self.error_codes = error_codes or []
		for ec in self.error_codes:
			ec.observation_path.add(self)

	@property
	def label(self) -> str:
		l = super().label
		if self.error_codes:
			l += f' ![{", ".join(n.error_code for n in self.error_codes)}]'
		return l

	def remove(self):
		super().remove()
		for ec in self.error_codes:
			ec.observation_path.remove(self)


class AffectsRelation(BNRelation, TNRelation, MNRelation, FunctionalRelation):
	_TYPE_NAME = 'AffectsRelation'
	_label = 'affects'
	_color = '#FF6600'

	source: Hardware
	target: Function


class SubfunctionOfRelation(BNRelation, TNRelation, MNRelation, FunctionalRelation):
	_TYPE_NAME = 'SubFunctionRelation'
	_label = 'subfunction_of'
	_color = '#FF99CC'

	source: Function
	target: Function


class ObservedByRelation(BNRelation, TNRelation, MNRelation, FunctionalRelation):
	_TYPE_NAME = 'ObservableRelation'
	_label = 'observed_by'
	_color = '#CCFFCC'

	source: Union[Hardware, Function]
	target: DirectObservable


class IndicatedByRelation(BNRelation, TNRelation, MNRelation, FunctionalRelation):
	_TYPE_NAME = 'IndicatedRelation'
	_label = 'indicated_by'
	_color = '#339966'

	source: Union[Hardware, Function]
	target: DiagnosticTestResult


class ResultsInRelation(FunctionalRelation):
	_TYPE_NAME = 'ResultsInRelation'
	_label = 'results_in'
	_color = '#99CCFF'

	source: DiagnosticTest
	target: DiagnosticTestResult


class CommunicatesThroughRelation(FunctionalRelation):
	_TYPE_NAME = 'CommunicatesThroughRelation'
	_label = 'communicates_through'
	_color = '#FFFF99'

	_error_codes: list[ObservedError]

	source: Function
	target: Function

	def __init__(self, source: MBDNode, target: MBDNode, error_codes: list[ObservedError] = None, weight: float = 1.,
				 net: 'Cluster' = None):
		super().__init__(source, target, weight, net)
		self._error_codes = error_codes

	@property
	def label(self) -> str:
		return f'{super().label} [{", ".join(ec.error_code for ec in self._error_codes)}]'

	@property
	def error_codes(self) -> list[ObservedError]:
		return self._error_codes


class YieldsErrorRelation(BNRelation, TNRelation, MNRelation, FunctionalRelation):
	_TYPE_NAME = 'YieldsErrorRelation'
	_label = 'yields_errors'
	_color = '#FFFF99'

	source: Function
	target: ObservedError


class ReportsErrorRelation(BNRelation, TNRelation, MNRelation, FunctionalRelation):
	_TYPE_NAME = 'Reports'
	_label = 'reports_error'
	_color = '#FFFF99'

	source: Function
	target: ObservedError


class SelectOperatingModeRelation(BNRelation, TNRelation, MNRelation, FunctionalRelation):
	_TYPE_NAME = 'SelectOperatingModeRelation'
	_label = 'select'
	_color = '#FF99FF'

	source: OperatingMode
	target: Function


class ClusterReasonerView(MBDNetReasonerView):
	net: Cluster = None

	_cycles: list[list[Function]] = None
	_cycle_functions: list[Optional[Function]] = None
	_cycle_relations: list[Optional[RequiredForRelation]] = None

	_duplicated_reporting_functions: dict[Function, set[Function]] = None
	_duplicated_reporting_chains: dict[RequiredForRelation, set[RequiredForRelation]] = None
	_duplicated_reporting_dependencies: dict[FunctionalRelation, set[FunctionalRelation]] = None

	def __init__(self, net: Cluster):
		self.net = net

	def _detect_and_cut_cycles(self):
		graph = self.net.to_nx(Function, relation_types=[RequiredForRelation])
		self._cycles: list[list[Function]] = [[self.net.get_node(fqn) for fqn in cycle] for cycle in
											  nx.simple_cycles(graph)]
		self._cycle_functions = []
		self._cycle_relations = []
		for idx, cycle in enumerate(self._cycles):
			if len(cycle) > 1:
				if any(f.states != Function.DEFAULT_STATES for f in cycle):
					print(f'Cannot cut loops with custom states: {", ".join(n.fqn for n in cycle)}.')
				else:
					self._cut_cycle(idx, cycle)
			else:
				print(f'Cannot cut self-dependency of {", ".join(n.fqn for n in cycle)}, results may be inaccurate!')

	def _cut_cycle(self, idx: int, cycle: list[Function]):
		cluster: Cluster = self.net.get_node(longest_common_fqn(*cycle))
		cycle_relations: list[RequiredForRelation] = []
		for _idx, fn in enumerate(cycle):
			cycle_predecessors = [_r for _r in fn.parent_relations if
								  isinstance(_r, RequiredForRelation) and _r.source == cycle[_idx - 1]]
			if len(cycle_predecessors) == 1:
				cycle_relations.append(cycle_predecessors[0])
			else:
				self._cycle_functions.append(None)
				self._cycle_relations.append(None)
				return
		hardware_relations, function_relations = self._collect_dependencies(cycle)
		cycle_fn = Function(f'Cycle_{idx}', Function.DEFAULT_STATES, cluster)
		self._cycle_functions.append(cycle_fn)
		for _idx, r in enumerate(cycle_relations):
			_r = RequiredForRelation(cycle_fn, r.target, r.operating_modes, r.error_codes, r.weight, cluster)
			if _idx == 0:
				self._cycle_relations.append(_r)
				r.remove()
		for hw, rs in hardware_relations.items():
			RealizesRelation(hw, cycle_fn, sum(r.weight for r in rs) / len(rs), cluster)
		for fn, rs in function_relations.items():
			RequiredForRelation(fn, cycle_fn, weight=sum(r.weight for r in rs) / len(rs),
								error_codes=list({ec for r in rs for ec in r.error_codes}), net=cluster)

	def _collect_dependencies(self, cycle: list[Function]) -> tuple[
		dict[Hardware, list[RealizesRelation]], dict[Function, list[RequiredForRelation]]]:
		dependent_hardware_relations: dict[Hardware, list[RealizesRelation]] = dict()
		dependent_functions_relations: dict[Function, list[RequiredForRelation]] = dict()
		for fn in cycle:
			for r in fn.parent_relations:
				if isinstance(r, RealizesRelation):
					if r.source not in dependent_hardware_relations:
						dependent_hardware_relations[r.source] = [r]
					else:
						dependent_hardware_relations[r.source].append(r)
				elif isinstance(r,
								RequiredForRelation) and r.source not in cycle and not r.source not in self._cycle_functions:
					if r.source not in dependent_functions_relations:
						dependent_functions_relations[r.source] = [r]
					else:
						dependent_functions_relations[r.source].append(r)
		return dependent_hardware_relations, dependent_functions_relations

	def _restore_cycles(self):
		for idx, cycle_fn in enumerate(self._cycle_functions):
			if cycle_fn is None:
				continue
			cycle_rel = self._cycle_relations[idx]
			RequiredForRelation(self._cycles[idx][-1], cycle_rel.target, cycle_rel.operating_modes,
								cycle_rel.error_codes, cycle_rel.weight)
			cycle_fn.remove()

	def _unfold_reporting_chains(self):
		self._duplicated_reporting_functions = dict()
		self._duplicated_reporting_chains = dict()
		self._duplicated_reporting_dependencies = dict()
		unique_paths: dict[frozenset[RequiredForRelation], list[ObservedError]] = dict()
		for rn in self.net.get_nodes_of_type(ObservedError):
			if not rn.observation_path:
				continue
			fs = frozenset(rn.observation_path)
			if fs in unique_paths:
				unique_paths[fs].append(rn)
			else:
				unique_paths[fs] = [rn]
		_idx = 0
		for path, observed_nodes in unique_paths.items():
			if len(observed_nodes) > 1:
				_idx += 1
				suffix = f'__{_idx}'
			else:
				suffix = f'__{observed_nodes[0].error_code}'
			nodes_to_duplicate: set[Function] = {fn for rfr in path for fn in (rfr.source, rfr.target)}
			replacements: dict[Function, Function] = dict()
			for ntd in nodes_to_duplicate:
				dn = Function(ntd.name + suffix, ntd.states, ntd.net)
				replacements[ntd] = dn
				if ntd in self._duplicated_reporting_functions:
					self._duplicated_reporting_functions[ntd].add(dn)
				else:
					self._duplicated_reporting_functions[ntd] = {dn}
				for pr in ntd.parent_relations:
					if pr not in path:
						if isinstance(pr, (RequiredForRelation, CommunicatesThroughRelation)):
							if pr.error_codes and set(pr.error_codes) != set(observed_nodes):
								continue
							if isinstance(pr, RequiredForRelation):
								dr = RequiredForRelation(pr.source, dn, pr.operating_modes, pr.error_codes, pr.weight, pr.net)
							elif isinstance(pr, CommunicatesThroughRelation):
								dr = CommunicatesThroughRelation(pr.source, dn, pr.error_codes, pr.weight, pr.net)
						else:
							dr = pr.__class__(pr.source, dn, pr.weight, pr.net)
						if pr in self._duplicated_reporting_dependencies:
							self._duplicated_reporting_dependencies[pr].add(dr)
						else:
							self._duplicated_reporting_dependencies[pr] = {dr}
				for cr in ntd.child_relations:
					if cr not in path:
						if isinstance(cr, ReportsErrorRelation) and cr.target not in observed_nodes:
							continue
						if isinstance(cr, RequiredForRelation):
							dr = RequiredForRelation(dn, cr.target, cr.operating_modes, cr.error_codes, cr.weight, cr.net)
						else:
							dr = cr.__class__(dn, cr.target, cr.weight, cr.net)
						if cr in self._duplicated_reporting_dependencies:
							self._duplicated_reporting_dependencies[cr].add(dr)
						else:
							self._duplicated_reporting_dependencies[cr] = {dr}
			for rfr in path:
				rrfr = RequiredForRelation(replacements[rfr.source], replacements[rfr.target], rfr.operating_modes,
										   rfr.error_codes, rfr.weight, rfr.net)
				if rfr in self._duplicated_reporting_chains:
					self._duplicated_reporting_chains[rfr].add(rrfr)
				else:
					self._duplicated_reporting_chains[rfr] = {rrfr}
		for ntr in self._duplicated_reporting_functions.keys():
			ntr.remove()

	def _fold_reporting_chains(self):
		for df, rfs in self._duplicated_reporting_functions.items():
			df.net.add_node(df)
			for rf in rfs:
				rf.remove()
		for rfr in self._duplicated_reporting_chains.keys():
			rfr.create()
		for dr in self._duplicated_reporting_dependencies.keys():
			dr.create()

	def __enter__(self):
		self._detect_and_cut_cycles()
		self._unfold_reporting_chains()
		self.net.propagate_operating_modes()

	def __exit__(self, exc_type, exc_value, exc_tb):
		self._fold_reporting_chains()
		self._restore_cycles()

# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from typing import Union, Optional

import numpy as np

from mbdlyb import MBDNode, MBDNet, MBDRelation, MBDNetReasonerView, MBDElement
from mbdlyb.operating_mode import OpmLogic, OpmLogicAnd, OpmLogicOr, OpmSet
from mbdlyb.formalisms.bayesnet import BayesNode, BayesAutoSplitNode, BayesNet, BNRelation
from mbdlyb.formalisms.tensor_network import TensorNode, TensorAutoSplitNode, TensorNet, TNRelation
from mbdlyb.formalisms.markovnet import MarkovNode, MarkovAutoSplitNode, MarkovNet, MNRelation


class FunctionalNode(MBDNode):
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
	_TYPE_NAME = 'Function'
	_color = '#CC99FF'
	_false_positive_weight: float = 0.
	_split_preserve_classes = [OperatingMode]

	_EQ_MAP = {
		'Hardware': 'Healthy',
		'Function': 'Ok'
	}

	def __init__(self, name: str, cluster: 'Cluster'):
		super().__init__(name, ['Ok', 'NOk'], cluster)
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
	_TYPE_NAME = 'Observable'

	fp_rate: float = 0.
	fn_rate: float = 0.

	def __init__(self, name: str, fp_rate: float, fn_rate: float, cluster: 'Cluster'):
		super().__init__(name, ['Ok', 'NOk'], cluster)
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

	def __init__(self, name: str, fp_rate: float, fn_rate: float, cluster: 'Cluster'):
		super().__init__(name, fp_rate, fn_rate, cluster)


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

	def __init__(self, name: str, fp_rate: float, fn_rate: float, cluster: 'Cluster'):
		super().__init__(name, fp_rate, fn_rate, cluster)

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
							rel.__class__ == RequiredForRelation]) and
						all([rel.inferred_operating_modes is not None for rel in node.parent_relations if
							rel.__class__ == SubfunctionOfRelation])}
			# also infer operating modes if one of the child required-for relations has no operating mode
			nodes_to_infer |=  {node for node in function_nodes if
						any([rel.inferred_operating_modes is not None and not rel.inferred_operating_modes for rel in node.child_relations if
							rel.__class__ == RequiredForRelation])}
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

				# set the operating mode on all outgoing subfunction-of relations
				for child_rel in node.child_relations:
					if not isinstance(child_rel, SubfunctionOfRelation):
						continue
					opmode = OpmLogic.from_copy(opmode_children)
					child_rel.set_inferred_operating_modes(opmode)
				function_nodes.remove(node)

		# Operating mode propagation - step 2
		# propagation from function and hardware nodes to diagnostic tests and direct observables
		for node in self.get_nodes_of_type(ObservableNode):
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

		# step 2: remove all test-result nodes which have a removed indicated-by relation
		for node in [rel.target for rel in removed_relations if isinstance(rel, IndicatedByRelation)]:
			if node not in removed_nodes:
				removed_nodes.add(node)
				node.remove()

		# step 3: remove all function nodes which have a removed sub-function relation and no remaining sub-function relations
		for node in [rel.target for rel in removed_relations if isinstance(rel, SubfunctionOfRelation)]:
			if node not in removed_nodes and not [rel for rel in node.parent_relations if isinstance(rel, SubfunctionOfRelation)]:
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

	def __init__(self, source: MBDNode, target: MBDNode, operating_modes: OpmLogic = OpmLogic(), weight: float = 1.,
				 net: Cluster = None):
		super().__init__(source, target, weight, net)
		self._operating_modes = operating_modes


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


class SelectOperatingModeRelation(BNRelation, TNRelation, MNRelation, FunctionalRelation):
	_TYPE_NAME = 'SelectOperatingModeRelation'
	_label = 'select'
	_color = '#FF99FF'

	source: OperatingMode
	target: Function


# TODO: Reconsider whether we need this class or not.
class ClusterReasonerView(MBDNetReasonerView):
	net: Cluster = None

	def __init__(self, net: Cluster):
		self.net = net
		self.net.propagate_operating_modes()

	def __enter__(self):
		pass

	def __exit__(self, exc_type, exc_value, exc_tb):
		pass

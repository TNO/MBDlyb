# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from typing import Union, Type, Optional
from abc import ABC, abstractmethod
import networkx as nx
import pyyed as yed
import pandas as pd


class TypedElement:
	_TYPE_NAME: str = 'TypedElement'

	@classmethod
	def get_type_name(cls) -> str:
		return cls._TYPE_NAME


class MBDElement(TypedElement):
	_TYPE_NAME: str = 'MBDElement'

	name: str = None
	_net: 'MBDNet' = None

	def __init__(self, name: str):
		super().__init__()
		self.name = name

	def __repr__(self):
		return self.fqn

	def __str__(self):
		return self.fqn

	def __hash__(self):
		return hash(f'{self.__class__.__name__}({self.fqn})')

	def __eq__(self, other) -> bool:
		if isinstance(other, MBDElement):
			return self.net == other.net and self.fqn == other.fqn
		return False

	def __gt__(self, other):
		if not isinstance(other, MBDElement):
			return str(self) > str(other)
		return self.level > other.level or (self.level == other.level and self.fqn > other.fqn)

	def __lt__(self, other):
		if not isinstance(other, MBDElement):
			return str(self) < str(other)
		return self.level < other.level or (self.level == other.level and self.fqn < other.fqn)

	def remove(self):
		raise NotImplementedError(f'Method \'remove\' is not yet implemented by {self.__class__.__name__}.')

	@property
	def net(self) -> Optional['MBDNet']:
		return self._net

	def set_net(self, net: 'MBDNet'):
		self._net = net

	@property
	def at_root(self) -> bool:
		return self.net is None

	@property
	def root(self) -> 'MBDElement':
		return self if self.at_root else self.net.root

	@property
	def level(self) -> int:
		return 0 if self.at_root else self.net.level + 1

	@property
	def fqn(self) -> str:
		return self.name if self.at_root else f'{self.net.fqn}.{self.name}'

	def rfqn(self, ref: str) -> str:
		ref = ref.split('.')
		_first_diff_idx, _ = first_diff_idx([self.fqn.split('.'), ref])
		return '.'.join(ref[_first_diff_idx:])


class MBDNode(MBDElement):
	_TYPE_NAME = 'MBDNode'

	_states: list[str] = None
	_color: str = '#FFFFFF'

	parent_relations: list['MBDRelation'] = None
	child_relations: list['MBDRelation'] = None

	def __init__(self, name: str, states: list[str], net: 'MBDNet'):
		super().__init__(name)
		self._states = states
		self.parent_relations = []
		self.child_relations = []
		net.add_nodes(self)

	def remove(self):
		for relation in self.relations:
			relation.remove()
		self.net.nodes.pop(self.name)

	@property
	def root(self) -> Union['MBDNode', 'MBDNet']:
		return self.net.root

	@property
	def relations(self) -> list['MBDRelation']:
		return self.parent_relations + self.child_relations

	@property
	def parents(self) -> list['MBDNode']:
		return [pr.source for pr in self.parent_relations]

	@property
	def children(self) -> list['MBDNode']:
		return [cr.target for cr in self.child_relations]

	@classmethod
	def get_color(cls) -> str:
		return cls._color

	@property
	def states(self) -> list[str]:
		return self._states


class MBDNet(MBDElement):
	_TYPE_NAME = 'MBDNet'

	nodes: dict[str, MBDElement] = None
	relations: list['MBDRelation'] = None

	def __init__(self, name: str, net: 'MBDNet' = None):
		super().__init__(name)
		self.nodes = dict()
		self.relations = []
		if net is not None:
			net.add_nodes(self)

	@property
	def root(self) -> 'MBDNet':
		return self if self.at_root else self.net.root

	def add_node(self, element: Union[MBDNode, 'MBDNet']):
		if element.name in self.nodes:
			raise KeyError(f'Element with name {element.name} already exists in {self.fqn}!')
		self.nodes[element.name] = element
		element.set_net(self)

	def add_nodes(self, *elements: Union[MBDNode, 'MBDNet']):
		for element in elements:
			self.add_node(element)

	def remove(self):
		for node in self.nodes.values():
			node.remove()
		if len(self.relations) != 0:
			raise RuntimeError(f'No relations should be contained in {self.name} after removing all of its childrens!')
		if not self.at_root:
			self.net.nodes.pop(self.name)

	def has_node(self, fqn: str | list[str]):
		try:
			self.get_node(fqn)
			return True
		except KeyError:
			return False

	def get_flat_nodes(self) -> list[MBDNode]:
		return [_n for n in self.nodes.values() for _n in (n.get_flat_nodes() if isinstance(n, MBDNet) else [n])]

	def get_flat_relations(self) -> list['MBDRelation']:
		return self.relations + [r for sn in self.subnets.values() for r in sn.get_flat_relations()]

	@property
	def subnets(self) -> dict[str, 'MBDNet']:
		return {name: node for name, node in self.nodes.items() if isinstance(node, MBDNet)}

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

	def get_nodes_of_type(self, *klass: Type[MBDNode], depth=1e6) -> list[MBDNode]:
		if depth < 0:
			return []
		return [n for n in self.nodes.values() if isinstance(n, klass)] + [n for subnet in self.nodes.values() if
																		   isinstance(subnet, MBDNet) for n in
																		   subnet.get_nodes_of_type(*klass,
																									depth=depth - 1)]

	def to_nx(self, *node_types: Type[MBDNode], relation_types: list[Type['MBDRelation']] = None) -> nx.DiGraph:
		node_types = tuple(node_types) or (MBDNode,)
		relation_types = tuple(relation_types) if relation_types else (MBDRelation,)
		nodes = [n for n in self.get_flat_nodes() if isinstance(n, node_types)]
		graph = nx.DiGraph()
		graph.add_nodes_from([
			(node.fqn, {'fqn': node.fqn, 'label': node.name, 'color': node.get_color()}) for node in nodes
		])
		graph.add_edges_from(
			[(r.source.fqn, r.target.fqn, {'label': r.label, 'color': r.get_color()}) for r in self.get_flat_relations()
			 if isinstance(r, relation_types) and r.source in nodes and r.target in nodes])
		return graph

	def to_yed(self, *node_types: Type[MBDNode], group: yed.Group = None,
			   is_graph_root: bool = True) -> yed.Graph | yed.Group:
		group: yed.Group | yed.Graph = group or yed.Graph()
		graph = group
		if is_graph_root:
			group = graph.add_group(self.fqn, label=self.name, transparent='true', fill='#FFFFFF',
									shape='roundrectangle')
		valid_node_types = tuple(node_types) or (MBDNode,)
		for node in self.nodes.values():
			if isinstance(node, MBDNet):
				_group = group.add_group(node.fqn, label=node.name, fill='#FEFEFE', shape='roundrectangle',
										 url=node.fqn)
				node.to_yed(*valid_node_types, group=_group, is_graph_root=False)
			elif isinstance(node, valid_node_types):
				group.add_node(node.fqn, label=node.name, shape_fill=node.get_color(), url=node.fqn)
		if is_graph_root:
			for r in self.get_flat_relations():
				if not (r.source.fqn in graph.existing_entities and r.target.fqn in graph.existing_entities):
					continue
				_edge = group.add_edge(r.source.fqn, r.target.fqn, color=r.get_color())
				_edge.add_label(r.label, text_color=r.get_color())
			return graph
		return group


class MBDRelation(TypedElement):
	_TYPE_NAME: str = 'MBDRelation'

	_label: str = None
	_color: str = '#000000'

	source: MBDNode = None
	target: MBDNode = None
	net: MBDNet = None

	def __init__(self, source: MBDNode, target: MBDNode, net: MBDNet = None):
		self.source = source
		self.target = target
		self.net = net or self.source.root.get_node(longest_common_fqn(source.fqn, target.fqn))
		self.create()

	def create(self):
		self.net.relations.append(self)
		self.source.child_relations.append(self)
		self.target.parent_relations.append(self)

	def __repr__(self):
		return f'[{self.__class__.__name__}] {self.source.name} --> {self.target.name}'

	def __hash__(self):
		return hash(f'{self.__class__.__name__}({hash(self.source)},{hash(self.target)})')

	def remove(self):
		self.source.child_relations.remove(self)
		self.target.parent_relations.remove(self)
		self.net.relations.remove(self)

	@classmethod
	def get_color(cls) -> str:
		return cls._color

	@classmethod
	def get_label(cls) -> str:
		return cls._label

	@property
	def label(self) -> str:
		if self._label:
			return self._label
		raise NotImplementedError(f'Label for {self.__class__.__name__} is undefined!')


class PartOfRelation(MBDRelation):
	pass


class MBDNetReasonerView:
	net: MBDNet = None

	def __enter__(self):
		pass

	def __exit__(self, exc_type, exc_value, exc_tb):
		pass


class MBDReasoner(ABC):
	_net: MBDNet = None
	_evidence: dict[str, list[float]] = None
	_targets: set[str] = None

	def __init__(self, net: MBDNet):
		self._net = net
		self._evidence = dict()

	@abstractmethod
	def init(self):
		pass

	@abstractmethod
	def infer(self, *targets: str) -> pd.DataFrame:
		pass

	@abstractmethod
	def conditional_entropy(self, Y: set[str], X: set[str], evidence: dict[str, str] = None) -> float:
		pass

	@abstractmethod
	def entropy(self, Y: set[str], evidence: dict[str, str] = None) -> float:
		pass

	def __repr__(self) -> str:
		return f'Reasoner for {self._net}'

	@property
	def evidence(self) -> dict[str, list[float]]:
		return self._evidence

	def reset(self):
		self.drop_evidence()
		self.drop_targets()

	def prep_evidence(self, node: str, evidence: Union[str, dict[str, float], list[float]]) -> tuple[str, list[float]]:
		n: MBDNode = node if isinstance(node, MBDNode) else self._net.get_node(node)
		if isinstance(evidence, dict):
			return n.fqn, [evidence[s] for s in n.states]
		elif isinstance(evidence, list):
			return n.fqn, evidence
		else:
			return n.fqn, [1. if s == evidence else 0. for s in n.states]

	def add_evidence(self, evidence: dict[str, str | dict[str, float] | list[float]]):
		for _n, _ev in evidence.items():
			n, ev = self.prep_evidence(_n, _ev)
			if any(ev):
				self._evidence[n] = ev

	def drop_evidence(self, *evs: Union[str, MBDNode]):
		if len(evs) > 0:
			for e in evs:
				_e = e if isinstance(e, str) else e.fqn
				if _e in self._evidence:
					del self._evidence[_e]
		else:
			self._evidence = dict()

	def add_targets(self, *targets: str, reset=False):
		if reset:
			self._targets = set()
		self._targets = self._targets.union(targets)

	def drop_targets(self, *targets: str):
		if len(targets) > 0:
			self._targets = self._targets.difference(targets)
		else:
			self._targets = {n.fqn for n in self._net.get_flat_nodes()}


def first_diff_idx(fqns: list[list[str]]) -> tuple[int, int]:
	min_length = min((len(fqn) for fqn in fqns))
	for j in range(min_length):
		if not all(fqns[0][j] == fqns[i][j] for i in range(len(fqns))):
			return j, min_length
	return min_length, min_length


def longest_common_fqn(*fqns: str | list[str] | MBDElement) -> Optional[str]:
	fqns = [fqn if isinstance(fqn, list) else (fqn if isinstance(fqn, str) else fqn.fqn).split('.') for fqn in fqns]
	_first_diff_idx, min_length = first_diff_idx(fqns)
	if min_length == 1:
		return None
	return '.'.join(fqns[0][:_first_diff_idx]) or None

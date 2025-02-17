# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from collections import OrderedDict
from itertools import product
from more_itertools import divide
from pathlib import Path
from typing import Callable, Type
from math import ceil
import pandas as pd

import pyAgrum as gum
import numpy as np

from mbdlyb import MBDElement, MBDNode, MBDNet, MBDRelation, MBDReasoner


MAX_PARENTS = 10


class BNRelation(MBDRelation):
	pass


class BNElement(MBDElement):
	def create_bn_node(self, bn: gum.BayesNet, node_sets: dict[Type[MBDNode], list[int]]):
		raise NotImplementedError(f'Function \'create_bn_node\' is not implemented for {self.__class__.name}!')

	def create_bn_connections(self, bn: gum.BayesNet):
		raise NotImplementedError(f'Function \'create_bn_connections\' is not implemented for {self.__class__.name}!')


class BayesNode(BNElement, MBDNode):
	_cpt_fn: Callable[[dict['MBDNode', str]], list[float]] = None

	@property
	def parents_bn(self) -> list['BayesNode']:
		return [pr.source for pr in self.parent_relations if isinstance(pr, BNRelation)]

	def create_bn_node(self, bn: gum.BayesNet, node_sets: dict[Type[MBDNode], list[int]]):
		if bn is None:
			raise AssertionError('A Bayesnet must be provided to add a node to it!')

		# add this node to the BN
		id: int = bn.add(gum.LabelizedVariable(self.fqn, self.name, self.states))
		if self.__class__ in node_sets:
			node_sets[self.__class__].append(id)
		else:
			node_sets[self.__class__] = [id]

	def create_bn_connections(self, bn: gum.BayesNet):
		bn.addArcs([(p.fqn, self.fqn) for p in self.parents_bn])
		bn.cpt(self.fqn)[:] = self._compute_cpt()

	def set_cpt_fn(self, cpt_fn: Callable[..., list[float]] | None):
		self._cpt_fn = cpt_fn

	def _compute_cpt(self) -> np.ndarray:
		shape = [len(p.states) for p in reversed(self.parents_bn)] + [len(self.states)]
		params = list(product(*[[(p, v) for v in p.states] for p in reversed(self.parents_bn)]))
		table = np.array([self._compute_cpt_line(dict(ps)) for ps in params]).flatten()
		return np.reshape(table, shape)

	def _compute_cpt_line(self, values: dict[MBDNode, str]) -> list[float]:
		if self._cpt_fn is not None:
			return self._cpt_fn(values)
		raise NotImplementedError(f'Method \'_compute_cpt_line\' is not implemented!')


class BayesAutoSplitNode(BayesNode):
	_split_preserve_classes: list[Type[BayesNode]] = []
	_parent_split: OrderedDict[str, list[BayesNode]] = None

	def create_bn_node(self, bn: gum.BayesNet, node_sets: dict[Type[MBDNode], list[int]]):
		super().create_bn_node(bn, node_sets)

		if len(self.parents_bn) > MAX_PARENTS:
			self._parent_split = OrderedDict()
			fixed_parents: list[BayesNode] = []
			eligable_parents_for_split: list[BayesNode] = []
			for p in self.parents_bn:
				if isinstance(p, *self._split_preserve_classes):
					fixed_parents.append(p)
				else:
					eligable_parents_for_split.append(p)

			split_room = max(2, MAX_PARENTS - len(fixed_parents))
			count_groups = ceil(len(eligable_parents_for_split) / split_room)
			for idx, g in enumerate(divide(count_groups, eligable_parents_for_split)):
				self._parent_split[f'{self.fqn}__{idx + 1}'] = fixed_parents + list(g)

			# add group nodes to the BN
			for group_id in self._parent_split.keys():
				id: int = bn.add(gum.LabelizedVariable(group_id, self.name + f'__{group_id}', self.states))
				node_sets[self.__class__].append(id)

	def create_bn_connections(self, bn: gum.BayesNet):
		if self._parent_split is None:
			super().create_bn_connections(bn)
		else:
			bn.addArcs([(p, self.fqn) for p in self._parent_split.keys()])
			bn.cpt(self.fqn)[:] = self._compute_join_cpt(len(self._parent_split))

			for split_node, parents in self._parent_split.items():
				bn.addArcs([(p.fqn, split_node) for p in parents])
				bn.cpt(split_node)[:] = self._compute_limited_cpt(parents)

	def _compute_limited_cpt(self, parents: list[BayesNode]) -> np.ndarray:
		shape = [len(p.states) for p in reversed(parents)] + [len(self.states)]
		params = list(product(*[[(p, v) for v in p.states] for p in reversed(parents)]))
		factor = np.array([self._compute_cpt_line(dict(ps)) for ps in params]).flatten()
		return np.reshape(factor, shape)

	def _compute_join_cpt(self, split_count: int) -> np.ndarray:
		raise NotImplementedError


class BayesNet(BNElement, MBDNet):
	def get_bn_nodes(self) -> list[BNElement]:
		return [n for n in self.nodes.values() if isinstance(n, BNElement)]

	def to_bn(self) -> tuple[gum.BayesNet, dict[Type[MBDNode], list[int]]]:
		bn = gum.BayesNet(self.name)
		node_sets: dict[Type[MBDNode], list[int]] = dict()
		
		with self.create_reasoner_view():
			# add all nodes to the Bayes Net
			self.create_bn_node(bn, node_sets)
			# add connections to the Bayes Net
			self.create_bn_connections(bn)
		return bn, node_sets

	def create_bn_node(self, bn: gum.BayesNet, node_sets: dict[Type[MBDNode], list[int]]):
		for node in self.get_bn_nodes():
			node.create_bn_node(bn, node_sets)

	def create_bn_connections(self, bn: gum.BayesNet):
		for node in self.get_bn_nodes():
			node.create_bn_connections(bn)

	def save_bn(self, path: Path | str, overwrite: bool = False):
		if isinstance(path, str):
			path = Path(path)
			if not path.suffix:
				path = path.with_suffix('.dne')
		if path.exists():
			if overwrite:
				path.unlink()
			else:
				raise FileExistsError(f'File {path} already exists!')
		
		net_spec = self._export_bn()
		with path.open('w') as fp:
			fp.write(net_spec)

	def _export_bn(self) -> str:
		# convert to Bayesian Network using pyAgrum
		bn, node_sets = self.to_bn()
		node_coords, max_x, max_y = self._layout_bn(bn)
		node_specs = [self._save_bn_node(bn, n, node_coords) for n in bn.nodes()]
		
		# format netica file and save to file
		net_spec = '''\
// ~->[DNET-1]->~

bnet knowledge_base {{
AutoCompile = TRUE;
autoupdate = TRUE;
whenchanged = 0;

visual V1 {{
	defdispform = BELIEFBARS;
	nodelabeling = TITLE;
	drawingbounds = ({bound_x}, {bound_y});
	{nodesets_markup}
}};

{nodes}

{nodesets}
}};\
		'''.format(nodesets_markup='\n\t'.join(
			f'NodeSet {ns.get_type_name()} {{Color=0x00{ns.get_color().strip("#")};}};' for ns in node_sets.keys()),
			bound_x=int(max_x), bound_y=int(max_y),
			nodes='\n\n'.join(node_specs),
			nodesets='\n'.join(
				f'NodeSet {ns.get_type_name()} {{Nodes = ({", ".join(_netica_id(n) for n in _nodes)});}};'
				for ns, _nodes in node_sets.items()))
		return net_spec

	@staticmethod
	def _layout_bn(bn: gum.BayesNet) -> tuple[dict[str, tuple[int, int]], int, int]:
		# auto-layout nodes in BN using dot
		import pydot
		graphs = pydot.graph_from_dot_data(bn.toDot())
		dot_bytes = graphs[0].create_dot()  # render dot-diagram to compute node-positions
		graphs = pydot.graph_from_dot_data(str(dot_bytes, encoding='utf-8'))
		node_coords = {}
		for n in bn.nodes():
			nodes = graphs[0].get_node(f'"{bn.variable(n).name()}"')
			xx, yy = nodes[0].get_pos().strip('"').split(',')
			node_coords[bn.variable(n).name()] = (float(xx), float(yy))
		max_x = max(x for x, _ in node_coords.values())
		max_y = max(y for _, y in node_coords.values())
		scale_x = 2. if max_x <= 8191 else 16383. / max_x
		scale_y = 2. if max_y <= 8191 else 16383. / max_y
		node_coords = {fqn: (int(x * scale_x), int(y * scale_y)) for fqn, (x, y) in node_coords.items()}
		return node_coords, max_x * scale_x, max_y * scale_y

	@staticmethod
	def _save_bn_node(bn: gum.BayesNet, n: int, layout: dict[str, tuple[int, int]]) -> str:
		var = bn.variable(n)
		name = var.name()
		cpt = bn.cpt(n).putFirst(name)
		parents = [bn.nodeId(v) for v in cpt.variablesSequence()[1:]]
		shape = [cpt.domainSize() // var.domainSize(), var.domainSize()]
		cpt = np.reshape(cpt.toarray().flatten(), shape)
		x, y = layout.get(name, (0, 0))
		return '''\
node {id} {{
	discrete = TRUE;
	states = ({states});
	kind = NATURE;
	chance = CHANCE;
	parents = ({parents});
	probs = 
		// {states}
		  ({cpt});
	title = "{name}";
	visual V1 {{
		center = ({x}, {y});
	}};
}};\
		'''.format(id=_netica_id(n),
			 	   states=', '.join(var.labels()),
				   parents=', '.join(_netica_id(p) for p in reversed(parents)),
				   cpt=',\n\t\t   '.join(', '.join(str(f) for f in r) for r in cpt),
				   name=name, x=x, y=y)


class BayesNetReasoner(MBDReasoner):
	_bn: gum.BayesNet = None
	_ie: gum.LazyPropagation = None

	def __init__(self, net: BayesNet):
		super().__init__(net)
		self.init()
	
	def init(self):
		self._bn, _ = self._net.to_bn()
		self._ie = gum.LazyPropagation(self._bn)

	def infer(self, *targets: str) -> pd.DataFrame:
		self._ie.setEvidence(self._evidence)
		self._ie.makeInference()
		targets = set(targets) or self._targets
		return pd.concat([
			self._ie.posterior(target).topandas().unstack(level=1) 
			for target in targets
		])

	def is_d_separated(self, Y: str | set[str], X: str | set[str], Z: str | set[str] | dict[str, str] = None):
		"""
		Computes whether Y and X are d-separated given Z.
		Z can either be a list of names of nodes, or a set of evidence.
		"""
		if type(X) == str:
			X = [X]

		if type(Y) == str:
			Y = [Y]

		if type(Z) == str:
			Z = {Z}

		Y = [self._bn.idFromName(name) for name in Y]
		X = [self._bn.idFromName(name) for name in X]

		if Z == None:
			Z = []
		elif type(Z) == dict[str, str]:
			Z = [self._bn.idFromName(name) for name in Z.keys()]
		else:
			Z = [self._bn.idFromName(name) for name in Z]

		return self._bn.dag().dSeparation(Y, X, Z)

	def conditional_entropy(
		self, y: set[str], x: set[str], evidence: dict[str, str] = None
	) -> float:
		"""Computes entropy of y given x."""
		# For some reason, there were bugs when combining with _ie, 
		# so for now create a new ie. Still need to handle evidence!
		ie = gum.LazyPropagation(self._bn)

		evidence = evidence or self._evidence
		if evidence:
			ie.setEvidence(evidence)

		it = gum.InformationTheory(ie, x, y)
		return it.entropyYgivenX()

	def entropy(self, x: set[str], evidence: dict[str, str] = None) -> float:
		"""Computes entropy of a set of nodes."""
		# For some reason, there were bugs when combining with _ie, 
		# so for now create a new ie. Still need to handle evidence!
		ie = gum.LazyPropagation(self._bn)
		ie.addJointTarget(x)

		evidence = evidence or self._evidence
		if evidence:
			ie.setEvidence(evidence)

		ie.makeInference()
		return ie.jointPosterior(x).entropy()


def _netica_id(n: int) -> str:
	return 'N_{:04d}'.format(n)

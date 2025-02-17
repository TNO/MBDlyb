# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from neomodel import (StructuredRel, StructuredNode, UniqueIdProperty, StringProperty, RelationshipTo, ZeroOrOne,
					  RelationshipFrom)
from .base import first_diff_idx


LBL_PARTOF = 'PART_OF'


class MBDRelation(StructuredRel):
	uid = UniqueIdProperty(),

	def __repr__(self):
		return f'[{self.__class__.__name__}] {self.start_node().fqn} --> {self.end_node().fqn}'


class PartOfRelation(MBDRelation):
	pass


class MBDElement(StructuredNode):
	uid = UniqueIdProperty()
	name = StringProperty(required=True)
	net = RelationshipTo('MBDNet', LBL_PARTOF, model=PartOfRelation, cardinality=ZeroOrOne)
	fqn = StringProperty(required=False)

	def __repr__(self):
		return self.fqn or self.name

	def __str__(self):
		return self.fqn or self.name

	def __hash__(self):
		return hash(f'{self.__class__.__name__}({self.fqn or self.name})')

	@property
	def at_root(self) -> bool:
		return len(self.net) == 0

	def get_fqn(self) -> str:
		return self.name if self.at_root else f'{self.get_net().fqn}.{self.name}'

	def set_net(self, net: 'MBDNet'):
		self.net.disconnect_all()
		self.net.connect(net)
		self.fqn = self.get_fqn()

	def get_net(self) -> 'MBDNet':
		return self.net.single()

	def get_root(self) -> 'MBDElement':
		if self.at_root:
			return self
		else:
			return self.get_net().get_root()

	def get_path_to(self, destination: 'MBDElement') -> list['MBDElement']:
		if self.at_root:
			return []
		net = self.get_net()
		if net == destination.get_net():
			return []
		else:
			return net.get_path_to(destination) + [net]

	def rfqn(self, ref: str) -> str:
		ref = ref.split('.')
		_first_diff_idx, _ = first_diff_idx([self.fqn.split('.'), ref])
		return '.'.join(ref[_first_diff_idx:])


class MBDNode(MBDElement):
	pass


class MBDNet(MBDElement):
	elements = RelationshipFrom('MBDElement', LBL_PARTOF, model=PartOfRelation)
	mbdnodes = RelationshipFrom('MBDNode', LBL_PARTOF, model=PartOfRelation)
	subnets = RelationshipFrom('MBDNet', LBL_PARTOF, model=PartOfRelation)

	def get_node(self, fqn: str | list[str], skipped_root: bool = False) -> MBDElement:
		if not fqn:
			return self
		if isinstance(fqn, str):
			fqn = fqn.split('.')
		if not skipped_root and self.at_root and self.name == fqn[0]:
			return self.get_node(fqn[1:], True)
		n = self.elements.get_or_none(name=fqn[0])
		if n is None:
			raise KeyError(f'Node {fqn[0]} not found in {self.fqn}!')
		return n.get_node(fqn[1:]) if fqn[1:] else n

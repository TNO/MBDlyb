# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from typing import Optional

from neomodel import DoesNotExist
from nicegui import ui

from mbdlyb.gdb import MBDElement


def goto(url: str):
	return ui.navigate.to(url)


# HELPER CLASSES

class Relation:
	_attr: str = ''

	_start_cls: type[MBDElement] = None
	_end_cls: type[MBDElement] = None
	start_uid: str = None
	end_uid: str = None
	_original_start_uid: Optional[str] = None
	_original_end_uid: Optional[str] = None
	weight: float

	def __init__(self, start_node: MBDElement = None, end_node: MBDElement = None, node_type: type[MBDElement] = None,
				 weight: float = 1.):
		if start_node is not None:
			self._start_cls = type(start_node)
			self.start_uid = start_node.uid
			self._original_start_uid = start_node.uid
		else:
			self._start_cls = node_type
		if end_node is not None:
			self._end_cls = type(end_node)
			self.end_uid = end_node.uid
			self._original_end_uid = end_node.uid
		else:
			self._end_cls = node_type
		self.weight = weight

	def get_start_node(self) -> MBDElement:
		return self._start_cls.nodes.get(uid=self.start_uid) if self.start_uid else None

	def get_end_node(self) -> MBDElement:
		return self._end_cls.nodes.get(uid=self.end_uid) if self.end_uid else None

	def get_original_start_node(self) -> MBDElement:
		return self._start_cls.nodes.get(uid=self._original_start_uid) if self._original_start_uid else None

	def get_original_end_node(self) -> MBDElement:
		return self._end_cls.nodes.get(uid=self._original_end_uid) if self._original_end_uid else None

	def start_node_changed(self) -> bool:
		return self._original_start_uid is not None and self.start_uid != self._original_start_uid

	def end_node_changed(self) -> bool:
		return self._original_end_uid is not None and self.end_uid != self._original_end_uid

	def save(self, redirect: str = None):
		if self.start_node_changed():
			self.get_original_start_node().__getattribute__(self._attr).disconnect(self.get_original_end_node())
		elif self._original_end_uid is not None:
			self.get_start_node().__getattribute__(self._attr).disconnect(self.get_original_end_node())
		self.get_start_node().__getattribute__(self._attr).connect(self.get_end_node(), self.get_props())
		if redirect is not None:
			ui.navigate.to(redirect)

	def get_props(self) -> dict:
		return {
			'weight': self.weight
		}


# DATABASE HELPERS

def get_object_or_404(cls, **kwargs):
	try:
		return cls.nodes.get(**kwargs)
	except DoesNotExist:
		ui.label(f'Could not find {cls.__name__} with parameters {dict(**kwargs)}.')
		return None


def get_relation_or_404(start_node: MBDElement, end_node: MBDElement, attr: str):
	relation = start_node.__getattribute__(attr).relationship(end_node)
	if relation is None:
		ui.label(f'Could not find \'{attr}\' relationship between {start_node} and {end_node}.')
	return relation

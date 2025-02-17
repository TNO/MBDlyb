# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from typing import Callable, Union

from nicegui import ui

from mbdlyb.gdb import MBDElement
from mbdlyb.ui.helpers import Relation
from mbdlyb.functional.gdb import Cluster, update_fqn


def goto(url: str):
	return ui.navigate.to(url)


# HELPER CLASSES


class SubfunctionOfRelation(Relation):
	_attr = 'subfunctions'


class RealizesRelation(Relation):
	_attr = 'realizes'


class RequiredForRelation(Relation):
	_attr = 'required_for'
	operating_modes: list[str] = None

	def __init__(self, start_node: MBDElement = None, end_node: MBDElement = None, node_type: type[MBDElement] = None,
				 weight: float = 1., operating_modes: list[str] = None):
		super().__init__(start_node, end_node, node_type, weight)
		self.operating_modes = operating_modes or []

	def get_props(self) -> dict:
		return {
			**super().get_props(),
			'operating_modes': self.operating_modes
		}


class AffectsRelation(Relation):
	_attr = 'affects'


class ObservesRelation(Relation):
	_attr = 'observed_by'


class TestsRelation(Relation):
	_attr = 'tested_by'


class IndicatesRelation(Relation):
	_attr = 'indicated_by'


# DATABASE HELPERS

def save_new_object(obj: MBDElement, net: Cluster, disable_redirect: bool = False):
	obj.save()
	obj.set_net(net)
	update_fqn(obj)
	if not disable_redirect:
		goto(f'/cluster/{net.uid}/')


def save_object(obj: MBDElement, succes_url: Union[str, Callable] = None):
	obj.save()
	update_fqn(obj)
	if succes_url is not None:
		goto(succes_url if isinstance(succes_url, str) else succes_url(obj))

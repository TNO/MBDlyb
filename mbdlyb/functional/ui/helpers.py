# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from typing import Callable, Union

from nicegui import ui

from mbdlyb.gdb import MBDElement
from mbdlyb.ui.helpers import Relation, get_object_or_404, CPT
from mbdlyb.functional.gdb import FunctionalNode, Cluster, update_fqn

from .base import page, Button, confirm


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


class State:
	def __init__(self, name: str):
		self.name = name

	def __eq__(self, other):
		if isinstance(other, State):
			return self.name == other.name
		if isinstance(other, str):
			return self.name == other
		return False


class StateTable:
	states: list[State]

	def __init__(self, states: list[State]):
		self.states = states

	def add_state(self):
		name = 'State_'
		for i in range(1, 100):
			if name + str(i) not in self.states:
				name = name + str(i)
				break
		self.states.append(State(name))
		state_table.refresh()


	def drop_state(self, state: State):
		self.states.remove(state)
		state_table.refresh()

	@classmethod
	def from_list(cls, l: list[str]) -> 'StateTable':
		return StateTable([State(s) for s in l or []])

	def to_list(self) -> list[str]:
		return [s.name for s in self.states]


@ui.refreshable
def state_table(state_table: StateTable, default_states: list[str]):
	ui.label(f'Leave empty for default states ({"/".join(default_states)})!')
	with ui.grid(columns=2).classes('vertical-bottom'):
		for col in ('Custom state',):
			ui.label(col).classes('font-bold')
		ui.icon('add', color='primary').on('click', lambda: state_table.add_state()).classes('cursor-pointer')
		for state in state_table.states:
			ui.input('State').bind_value(state, 'name')
			ui.icon('delete', color='negative').on('click', lambda s=state: state_table.drop_state(s)).classes(
				'self-end cursor-pointer')


def node_cpt(klass: type[FunctionalNode], node_id: str, url: str):
	def _save(node: type[FunctionalNode], cpt: CPT):
		node.cpt = cpt.normalize().to_dict()
		save_object(node, url.format(node_id))

	def _drop(node: type[FunctionalNode], node_id: str, url: str):
		node.cpt = dict()
		save_object(node, url.format(node_id))

	node = get_object_or_404(klass, uid=node_id)
	if node is None:
		return

	cpt, message = CPT.from_dict(node, node.cpt)
	buttons = [Button('Discard', None, 'warning', lambda: goto(url.format(node_id))),
			   Button('Save', 'save', None, lambda: _save(node, cpt))]
	if not node.requires_cpt:
		buttons = [Button('Drop', None, 'negative', lambda: confirm('Drop CPT',
																	'Are you sure to drop the CPT? This will revert the behavior of this node to its default behavior?',
																	'Drop CPT', lambda: _drop(node, node_id, url),
																	'negative'))] + buttons
	page(f'CPT of {node}', node.get_net(), buttons)
	if message:
		ui.label(f'{message} The CPT has been regenerated.').classes('text-warning')
	with ui.grid(columns=len(cpt.parents) + len(node.states)).classes('gap-0'):
		fill_all = {s: .0 for s in node.states}
		for _ in cpt.parents:
			ui.label()
		for state in node.states:
			ui.label(state).classes('border text-weight-bold p-1')
		for parent in cpt.parents:
			ui.label(parent.name).classes('border text-weight-bold p-1')
		for state in node.states:
			with ui.row().classes('border p-1'):
				ui.number(min=0., max=1., step=.01, precision=5).bind_value(fill_all, state).props('borderless dense')
				ui.button(icon='content_copy').props('flat borderless dense stretch').on_click(lambda s=state: cpt.set_prob(s, fill_all[s]))
		for line in cpt.lines:
			for state_value in line.conditions:
				ui.label(state_value).classes('border p-1')
			for c in line.cells:
				ui.number(min=0., max=1., step=.01, precision=5).bind_value(c, 'probability').props('borderless dense').classes('border p-1')
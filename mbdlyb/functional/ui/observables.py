# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""
import mbdlyb.functional as fn

from nicegui import ui, APIRouter
from neomodel import db
from mbdlyb.functional.gdb import Cluster, Function, Hardware, DirectObservable
from mbdlyb.ui.helpers import get_object_or_404, get_relation_or_404

from .base import Button, page, TableColumn, Table, confirm_delete, confirm_delete_relation, confirm
from .helpers import goto, save_object, save_new_object, ObservesRelation, node_cpt, StateTable, state_table
from .validation import base_name_validation


router = APIRouter()


def observable_form(observable: DirectObservable, states: StateTable, cluster: Cluster, update: bool = True):
	ui.input('Name', validation=base_name_validation(cluster, observable if update else None)) \
		.bind_value(observable, 'name').props('hide-bottom-space')
	ui.number('False positive rate', min=0., max=1., step=.001).bind_value(observable, 'fp_rate')
	ui.number('False negative rate', min=0., max=1., step=.001).bind_value(observable, 'fn_rate')
	state_table(states, fn.DirectObservable.DEFAULT_STATES)


@router.page('/cluster/{cluster_id}/observable/new/')
def observable_create(cluster_id: str):
	def _save(observable: Hardware, states: StateTable, cluster: Cluster):
		observable.states = states.to_list()
		save_new_object(observable, cluster)

	cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return
	observable = DirectObservable(name='')
	states = StateTable.from_list(observable.states_)
	page(f'New observable in {cluster.name}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save(observable, states, cluster))
	])
	observable_form(observable, states, cluster, False)


@router.page('/observable/{observable_id}/update/')
def observable_update(observable_id: str):
	def _save(observable: Hardware, states: StateTable):
		observable.states = states.to_list()
		save_object(observable, f'/cluster/{cluster.uid}/')

	observable: DirectObservable = get_object_or_404(DirectObservable, uid=observable_id)
	if observable is None:
		return
	cluster = observable.get_net()
	states = StateTable.from_list(observable.states_)
	page(f'Update {observable.fqn}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save(observable, states))
	])
	observable_form(observable, states, cluster)


@router.page('/observable/{observable_id}/')
def observable_details(observable_id: str):
	def _move_up(observable: DirectObservable):
		observable.set_net(observable.get_net().get_net())
		goto(f'/observable/{observable.uid}/')

	observable: DirectObservable = get_object_or_404(DirectObservable, uid=observable_id)
	if observable is None:
		return
	cluster: Cluster = observable.get_net()
	buttons = [
		Button(None, 'border_all', 'secondary', lambda: goto(f'/observable/{observable.uid}/cpt/'), 'Modify CPT'),
		Button(None, 'edit', None, lambda: goto(f'/observable/{observable.uid}/update/'), 'Edit observable'),
		Button(None, 'delete', 'negative', lambda: confirm_delete(observable, f'/cluster/{cluster.uid}/'), 'Delete observable')]
	if not (observable.at_root or cluster.at_root):
		buttons = [Button(None, 'move_up', 'secondary', lambda: confirm('Move up observable',
																  f'Are you sure to move {observable.name} up to {cluster.get_net().fqn}?',
																  'I am sure', lambda: _move_up(observable)), 'Move observable up')] + buttons
	page(f'Observable {observable.name}', cluster, buttons)
	with ui.row():
		ui.label('False positive/negative rate').classes('font-bold')
		ui.label(f'{observable.fp_rate} / {observable.fn_rate}')
	with ui.grid(columns='1fr 1fr').classes('w-full'):
		with ui.card():
			Table('Observed functions', observable.observed_functions.order_by('fqn'), [
				TableColumn('Name', 'name', lambda f: f'/function/{f.uid}/', 'fqn'),
				TableColumn('Weight', lambda f: observable.observed_functions.relationship(f).weight)
			], observable, [
					  Button(icon='add', color='positive', tooltip='Add observed function',
							 handler=f'/observable/{observable.uid}/observes_fn/new/')
				  ], [
					  Button(icon='edit', tooltip='Edit observed function',
							 handler=lambda x: goto(f'/observable/{observable.uid}/observes_fn/{x.uid}/update/')),
					  Button(icon='delete', color='negative', tooltip='Delete observed function',
							 handler=lambda x: confirm_delete_relation('observed_functions', observable, x,
																	   f'/observable/{observable.uid}/'))
				  ]).show()
		with ui.card():
			Table('Observed hardware', observable.observed_hardware.order_by('fqn'), [
				TableColumn('Name', 'name', lambda h: f'/hardware/{h.uid}/', 'fqn'),
				TableColumn('Weight', lambda f: observable.observed_hardware.relationship(f).weight)
			], observable, [
					  Button(icon='add', color='positive', tooltip='Add observed hardware',
							 handler=f'/observable/{observable.uid}/observes_hw/new/')
				  ], [
					  Button(icon='edit', tooltip='Edit observed hardware',
							 handler=lambda x: goto(f'/observable/{observable.uid}/observes_hw/{x.uid}/update/')),
					  Button(icon='delete', color='negative', tooltip='Delete observed hardware',
							 handler=lambda x: confirm_delete_relation('observed_hardware', observable, x,
																	   f'/observable/{observable.uid}/'))
				  ]).show()


# OBSERVES FORM

def observes_relation_form(relation: ObservesRelation, node_type: str):
	options, _ = db.cypher_query(
		f'''MATCH
    (o:DirectObservable {{uid: '{relation.end_uid}'}})-[:PART_OF]->(:Cluster)<-[r:PART_OF]-{{0,99}}(:Cluster)<-[:PART_OF]-(x:{node_type})
WHERE
	NOT (x)-[:OBSERVED_BY]->(o)
RETURN x
ORDER BY size(r), x.fqn''', resolve_objects=True)
	options_dict = dict() if relation.start_uid is None else {relation.start_uid: relation.get_start_node().fqn}
	ui.select({**options_dict, **{sf[0].uid: sf[0].fqn for sf in options}}, label=node_type, with_input=True).bind_value(
		relation, 'start_uid')
	ui.number('Weight', min=0., max=1., precision=2, step=.01).bind_value(relation, 'weight')


# OBSERVES FUNCTION RELATION

@router.page('/observable/{observable_id}/observes_fn/new/')
def observes_fn_relation_create(observable_id: str):
	observable = get_object_or_404(DirectObservable, uid=observable_id)
	if observable is None:
		return

	relation = ObservesRelation(end_node=observable, node_type=Function)
	page(f'Add observed function to {observable}', observable.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/observable/{observable.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/observable/{observable.uid}/'))
	])
	observes_relation_form(relation, 'Function')


@router.page('/observable/{observable_id}/observes_fn/{function_id}/update/')
def observes_fn_relation_update(observable_id: str, function_id: str):
	observable = get_object_or_404(DirectObservable, uid=observable_id)
	function = get_object_or_404(Function, uid=function_id)
	relation = get_relation_or_404(observable, function, 'observed_functions')
	if None in (observable, function, relation):
		return

	relation = ObservesRelation(function, observable, weight=relation.weight)
	page(f'Update {relation}', observable.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/observable/{observable.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/observable/{observable.uid}/'))
	])
	observes_relation_form(relation, 'Function')


# OBSERVES HARDWARE RELATION

@router.page('/observable/{observable_id}/observes_hw/new/')
def observes_hw_relation_create(observable_id: str):
	observable = get_object_or_404(DirectObservable, uid=observable_id)
	if observable is None:
		return

	relation = ObservesRelation(end_node=observable, node_type=Hardware)
	page(f'Add observed hardware to {observable}', observable.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/observable/{observable.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/observable/{observable.uid}/'))
	])
	observes_relation_form(relation, 'Hardware')


@router.page('/observable/{observable_id}/observes_hw/{hardware_id}/update/')
def observes_hw_relation_update(observable_id: str, hardware_id: str):
	observable = get_object_or_404(DirectObservable, uid=observable_id)
	hardware = get_object_or_404(Hardware, uid=hardware_id)
	relation = get_relation_or_404(observable, hardware, 'observed_hardware')
	if None in (observable, hardware, relation):
		return

	relation = ObservesRelation(hardware, observable, weight=relation.weight)
	page(f'Update {relation}', observable.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/observable/{observable.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/observable/{observable.uid}/'))
	])
	observes_relation_form(relation, 'Hardware')


@router.page('/observable/{observable_id}/cpt/')
def observable_cpt(observable_id: str):
	node_cpt(DirectObservable, observable_id, '/observable/{}/')

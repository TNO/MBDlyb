# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from nicegui import ui, APIRouter
from neomodel import db
from mbdlyb.functional.gdb import Cluster, Function, Hardware, DirectObservable
from mbdlyb.ui.helpers import get_object_or_404, get_relation_or_404

from .base import Button, page, build_table, confirm_delete, confirm_delete_relation, confirm
from .helpers import goto, save_object, save_new_object, ObservesRelation
from .validation import base_name_validation


router = APIRouter()


def observable_form(observable: DirectObservable, cluster: Cluster, update: bool = True):
	ui.input('Name', validation=base_name_validation(cluster, observable if update else None)) \
		.bind_value(observable, 'name').props('hide-bottom-space')
	ui.number('False positive rate', min=0., max=1., step=.001).bind_value(observable, 'fp_rate')
	ui.number('False negative rate', min=0., max=1., step=.001).bind_value(observable, 'fn_rate')


@router.page('/cluster/{cluster_id}/observable/new/')
def observable_create(cluster_id: str):
	def _save(observable: Hardware, cluster: Cluster):
		save_new_object(observable, cluster)

	cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return
	observable = DirectObservable(name='')
	page(f'New observable in {cluster.name}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save(observable, cluster))
	])
	observable_form(observable, cluster, False)


@router.page('/observable/{observable_id}/update/')
def observable_update(observable_id: str):
	def _save(observable: Hardware):
		save_object(observable, f'/cluster/{cluster.uid}/')

	observable: DirectObservable = get_object_or_404(DirectObservable, uid=observable_id)
	if observable is None:
		return
	cluster = observable.get_net()
	page(f'Update {observable.fqn}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save(observable))
	])
	observable_form(observable, cluster)


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
			build_table('Observed functions', [
				('FQN', 'fqn'), ('Weight', ('observed_by', lambda sf: sf.relationship(observable).weight))
			], observable.observed_functions.order_by('fqn'),
						detail_url='/function/{}/',
						create_url=f'/observable/{observable.uid}/observes_fn/new/',
						edit_fn=lambda x: goto(f'/observable/{observable.uid}/observes_fn/{x.uid}/update/'),
						delete_fn=lambda x: confirm_delete_relation('observed_functions', observable, x,
																	f'/observable/{observable.uid}/'))
		with ui.card():
			build_table('Observed hardware', [
				('FQN', 'fqn'), ('Weight', ('observed_by', lambda sf: sf.relationship(observable).weight))
			], observable.observed_hardware.order_by('fqn'),
						detail_url='/hardware/{}/',
						create_url=f'/observable/{observable.uid}/observes_hw/new/',
						edit_fn=lambda x: goto(f'/observable/{observable.uid}/observes_hw/{x.uid}/update/'),
						delete_fn=lambda x: confirm_delete_relation('observed_hardware', observable, x,
																	f'/observable/{observable.uid}/'))


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

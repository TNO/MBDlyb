# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from nicegui import ui, APIRouter
from neomodel import db
from mbdlyb.functional.gdb import Cluster, Function, Hardware, DirectObservable, DiagnosticTest
from mbdlyb.ui.helpers import get_object_or_404, get_relation_or_404

from .base import Button, page, TableColumn, Table, confirm_delete, confirm_delete_relation
from .helpers import (goto, save_object, save_new_object, RealizesRelation, AffectsRelation, ObservesRelation,
					  TestsRelation)
from .validation import base_name_validation


router = APIRouter()


class Fault:
	def __init__(self, name: str, prior: float):
		self.name = name
		self.prior = prior

	def __eq__(self, other):
		if isinstance(other, Fault):
			return self.name == other
		return False

	@classmethod
	def to_list(cls, d: dict[str, float]) -> list['Fault']:
		return [Fault(k, v) for k, v in d.items()]

	@classmethod
	def from_list(cls, l: list['Fault']):
		return {f.name: f.prior for f in l}


def add_fault(fault_rates: list[Fault]):
	name = 'Fault_'
	for i in range(1, 100):
		if name + str(i) not in Fault.from_list(fault_rates):
			name = name + str(i)
			break
	fault_rates.append(Fault(name, 0.01))
	fault_table.refresh()


def drop_fault(fault_rates: list[Fault], fault: Fault):
	fault_rates.remove(fault)
	fault_table.refresh()


@ui.refreshable
def fault_table(fault_rates: list[Fault]):
	with ui.grid(columns=3).classes('vertical-bottom'):
		for col in ('Fault', 'Prior'):
			ui.label(col).classes('font-bold')
		ui.icon('add', color='primary').on('click', lambda: add_fault(fault_rates)).classes('cursor-pointer')
		for fault in fault_rates:
			ui.input('Fault').bind_value(fault, 'name')
			ui.number('Prior', min=0., max=1., step=.001).bind_value(fault, 'prior')
			ui.icon('delete', color='negative').on('click', lambda f=fault: drop_fault(fault_rates, f)).classes(
				'self-end cursor-pointer')


def hardware_form(hardware: Hardware, fault_rates: list[Fault], cluster: Cluster, update: bool = True):
	ui.input('Name', validation=base_name_validation(cluster, hardware if update else None)) \
		.bind_value(hardware, 'name').props('hide-bottom-space')
	fault_table(fault_rates)


@router.page('/cluster/{cluster_id}/hardware/new/')
def hardware_create(cluster_id: str):
	def _save(hardware: Hardware, fault_rates: list[Fault], cluster: Cluster):
		hardware.fault_rates = Fault.from_list(fault_rates)
		save_new_object(hardware, cluster)

	cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return
	hardware = Hardware(name='')
	fault_rates = Fault.to_list(hardware.fault_rates)
	page(f'New hardware in {cluster.name}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save(hardware, fault_rates, cluster))
	])
	hardware_form(hardware, fault_rates, cluster, False)


@router.page('/hardware/{hardware_id}/update/')
def hardware_update(hardware_id: str):
	def _save(hardware: Hardware, fault_rates: list[Fault]):
		hardware.fault_rates = Fault.from_list(fault_rates)
		save_object(hardware, f'/cluster/{cluster.uid}/')

	hardware: Hardware = get_object_or_404(Hardware, uid=hardware_id)
	if hardware is None:
		return
	cluster = hardware.get_net()
	fault_rates = Fault.to_list(hardware.fault_rates)
	page(f'Update {hardware.fqn}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save(hardware, fault_rates))
	])
	hardware_form(hardware, fault_rates, cluster)


@router.page('/hardware/{hardware_id}/')
def hardware_details(hardware_id: str):
	hardware: Hardware = get_object_or_404(Hardware, uid=hardware_id)
	if hardware is None:
		return
	cluster: Cluster = hardware.get_net()
	buttons = [
		Button(None, 'edit', None, lambda: goto(f'/hardware/{hardware.uid}/update/'), 'Edit hardware'),
		Button(None, 'delete', 'negative', lambda: confirm_delete(hardware, f'/cluster/{cluster.uid}/'), 'Delete hardware')]
	page(f'Hardware {hardware.name}', cluster, buttons)
	with ui.row():
		ui.label('Indicated by')
		for test_result in hardware.indicated_by.order_by('name'):
			ui.link(test_result.name, f'/test/{test_result.results_from.single().uid}/').tooltip(test_result.fqn)
	with ui.grid(columns='1fr 1fr').classes('w-full'):
		with ui.card():
			Table('Realizes', hardware.realizes.order_by('fqn'), [
				TableColumn('Name', 'name', lambda f: f'/function/{f.uid}/', 'fqn'),
				TableColumn('Weight', lambda f: f.realized_by.relationship(hardware).weight)
			], hardware, [
					  Button(icon='add', tooltip='Add hardware realization',
							 handler=f'/hardware/{hardware.uid}/realizes/new/')
				  ], [
					  Button(icon='edit', tooltip='Edit realization',
							 handler=lambda x: goto(f'/hardware/{hardware.uid}/realizes/{x.uid}/update/')),
					  Button(icon='delete', color='negative', tooltip='Delete realization',
							 handler=lambda x: confirm_delete_relation('realizes', hardware, x,
																	   f'/hardware/{hardware.uid}/'))
				  ]).show()
		with ui.card():
			Table('Affects', hardware.affects.order_by('fqn'), [
				TableColumn('Name', 'name', lambda f: f'/function/{f.uid}/', 'fqn'),
				TableColumn('Weight', lambda f: f.affected_by.relationship(hardware).weight)
			], hardware, [
					  Button(icon='add', tooltip='Add hardware side-effect',
							 handler=f'/hardware/{hardware.uid}/affects/new/')
				  ], [
					  Button(icon='edit', tooltip='Edit side-effect',
							 handler=lambda x: goto(f'/hardware/{hardware.uid}/affects/{x.uid}/update/')),
					  Button(icon='delete', color='negative', tooltip='Delete side-effect',
							 handler=lambda x: confirm_delete_relation('affects', hardware, x,
																	   f'/hardware/{hardware.uid}/'))
				  ]).show()
		with ui.card():
			Table('Observed by', hardware.observed_by.order_by('fqn'), [
				TableColumn('Name', 'name', lambda f: f'/function/{f.uid}/', 'fqn'),
				TableColumn('Weight', lambda o: o.observed_hardware.relationship(hardware).weight)
			], hardware, [
					  Button(icon='add', tooltip='Add observable', handler=f'/hardware/{hardware.uid}/observedby/new/')
				  ], [
					  Button(icon='edit', tooltip='Edit observable',
							 handler=lambda x: goto(f'/hardware/{hardware.uid}/observedby/{x.uid}/update/')),
					  Button(icon='delete', color='negative', tooltip='Delete observable',
							 handler=lambda x: confirm_delete_relation('observed_by', hardware, x,
																	   f'/hardware/{hardware.uid}/'))
				  ]).show()


# REALIZES RELATION

def realizes_relation_form(relation: RealizesRelation):
	options, _ = db.cypher_query(
		f'''MATCH
    (h:Hardware {{uid: '{relation.start_uid}'}})-[:PART_OF]->(:Cluster)<-[:PART_OF]-(f:Function)
WHERE
    NOT (h)-[:REALIZES]->(f)
RETURN f
ORDER BY f.name''', resolve_objects=True)
	options_dict = dict() if relation.end_uid is None else {relation.end_uid: relation.get_end_node().name}
	ui.select({**options_dict, **{sf[0].uid: sf[0].name for sf in options}}, label='Function',
			  with_input=True).bind_value(relation, 'end_uid')
	ui.number('Weight', min=0., max=1., precision=2, step=.01).bind_value(relation, 'weight')


@router.page('/hardware/{hardware_id}/realizes/new/')
def realizes_relation_create(hardware_id: str):
	hardware = get_object_or_404(Hardware, uid=hardware_id)
	if hardware is None:
		return

	relation = RealizesRelation(hardware, node_type=Function)
	page(f'Add realizes to {hardware}', hardware.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/hardware/{hardware.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/hardware/{hardware.uid}/'))
	])
	realizes_relation_form(relation)


@router.page('/hardware/{hardware_id}/realizes/{function_id}/update/')
def realizes_relation_update(hardware_id: str, function_id: str):
	hardware = get_object_or_404(Hardware, uid=hardware_id)
	function = get_object_or_404(Function, uid=function_id)
	relation = get_relation_or_404(hardware, function, 'realizes')
	if None in (hardware, function, relation):
		return

	relation = RealizesRelation(hardware, function, weight=relation.weight)
	page(f'Update {relation}', hardware.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/hardware/{hardware.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/hardware/{hardware.uid}/'))
	])
	realizes_relation_form(relation)


# AFFECTS RELATION

def affects_relation_form(relation: AffectsRelation, root: Cluster):
	options, _ = db.cypher_query(
		f'''MATCH
    (h:Hardware {{uid: '{relation.start_uid}'}})-[:PART_OF]->(c:Cluster),
    (f:Function)-[:PART_OF*]->(:Cluster {{uid: '{root.uid}'}})
WHERE
    NOT (h)-[:AFFECTS]->(f) AND NOT (f)-[:PART_OF]->(c)
 RETURN f''', resolve_objects=True)
	options_dict = dict() if relation.end_uid is None else {relation.end_uid: relation.get_end_node().name}
	ui.select({**options_dict, **{sf[0].uid: sf[0].name for sf in options}}, label='Function').bind_value(
		relation, 'end_uid')
	ui.number('Weight', min=0., max=1., precision=2, step=.01).bind_value(relation, 'weight')


@router.page('/hardware/{hardware_id}/affects/new/')
def affects_relation_create(hardware_id: str):
	hardware = get_object_or_404(Hardware, uid=hardware_id)
	if hardware is None:
		return

	relation = AffectsRelation(hardware, node_type=Function)
	page(f'Add affects to {hardware}', hardware.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/hardware/{hardware.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/hardware/{hardware.uid}/'))
	])
	root = hardware.get_root()
	affects_relation_form(relation, root)


@router.page('/hardware/{hardware_id}/affects/{function_id}/update/')
def affects_relation_update(hardware_id: str, function_id: str):
	hardware = get_object_or_404(Hardware, uid=hardware_id)
	function = get_object_or_404(Function, uid=function_id)
	relation = get_relation_or_404(hardware, function, 'affects')
	if None in (hardware, function, relation):
		return

	relation = AffectsRelation(hardware, function, relation.weight)
	page(f'Update {relation}', hardware.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/hardware/{hardware.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/hardware/{hardware.uid}/'))
	])
	root = hardware.get_root()
	affects_relation_form(relation, root)


# OBSERVED BY RELATION

def observes_relation_form(o_relation: ObservesRelation):
	options, _ = db.cypher_query(
		f'''MATCH
    (h:Hardware {{uid: '{o_relation.start_uid}'}})-[:PART_OF]->(:Cluster)(()-[:PART_OF]->()){{0,99}}(:Cluster)<-[PART_OF]-(o:DirectObservable)
WHERE
    NOT (h)-[:OBSERVED_BY]->(o)
 RETURN o''', resolve_objects=True)
	options_dict = dict() if o_relation.end_uid is None else {o_relation.end_uid: o_relation.get_end_node().fqn}
	ui.select({**options_dict, **{sf[0].uid: sf[0].fqn for sf in options}}, label='Observer',
			  with_input=True).bind_value(o_relation, 'end_uid')
	ui.number('Weight', min=0., max=1., precision=2, step=.01).bind_value(o_relation, 'weight')


@router.page('/hardware/{hardware_id}/observedby/new/')
def observes_relation_create(hardware_id: str):
	hardware = get_object_or_404(Hardware, uid=hardware_id)
	if hardware is None:
		return

	o_relation = ObservesRelation(hardware, node_type=DirectObservable)
	page(f'Add observed_by to {hardware}', hardware.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/hardware/{hardware.uid}/')),
		Button('Save', 'save', None, lambda: o_relation.save(f'/hardware/{hardware.uid}/'))
	])
	observes_relation_form(o_relation)


@router.page('/hardware/{hardware_id}/observedby/{observable_id}/update/')
def observes_relation_update(hardware_id: str, observable_id: str):
	hardware = get_object_or_404(Hardware, uid=hardware_id)
	observable = get_object_or_404(DirectObservable, uid=observable_id)
	relation = get_relation_or_404(hardware, observable, 'observed_by')
	if None in (hardware, observable, relation):
		return

	o_relation = ObservesRelation(hardware, observable, weight=relation.weight)
	page(f'Update {relation}', hardware.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/hardware/{hardware.uid}/')),
		Button('Save', 'save', None, lambda: o_relation.save(f'/hardware/{hardware.uid}/'))
	])
	observes_relation_form(o_relation)


# TESTED BY RELATION

def tests_relation_form(t_relation: TestsRelation):
	options, _ = db.cypher_query(
		f'''MATCH
    (h:Hardware {{uid: '{t_relation.start_uid}'}})-[:PART_OF]->(:Cluster)(()-[:PART_OF]->()){{0,99}}(:Cluster)<-[PART_OF]-(t:DiagnosticTest)
WHERE
    NOT (h)-[:TESTED_BY]->(t)
 RETURN t''', resolve_objects=True)
	options_dict = dict() if t_relation.end_uid is None else {t_relation.end_uid: t_relation.get_end_node().fqn}
	ui.select({**options_dict, **{sf[0].uid: sf[0].fqn for sf in options}}, label='Diagnostic test',
			  with_input=True).bind_value(t_relation, 'end_uid')
	ui.number('Weight', min=0., max=1., precision=2, step=.01).bind_value(t_relation, 'weight')


@router.page('/hardware/{hardware_id}/testedby/new/')
def tests_relation_create(hardware_id: str):
	hardware = get_object_or_404(Hardware, uid=hardware_id)
	if hardware is None:
		return

	t_relation = TestsRelation(hardware, node_type=DiagnosticTest)
	page(f'Add tested_by to {hardware}', hardware.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/hardware/{hardware.uid}/')),
		Button('Save', 'save', None, lambda: t_relation.save(f'/hardware/{hardware.uid}/'))
	])
	tests_relation_form(t_relation)


@router.page('/hardware/{hardware_id}/testedby/{test_id}/update/')
def observes_relation_update(hardware_id: str, test_id: str):
	hardware = get_object_or_404(Hardware, uid=hardware_id)
	test = get_object_or_404(DiagnosticTest, uid=test_id)
	relation = get_relation_or_404(hardware, test, 'tested_by')
	if None in (hardware, test, relation):
		return

	t_relation = TestsRelation(hardware, test, weight=relation.weight)
	page(f'Update {relation}', hardware.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/hardware/{hardware.uid}/')),
		Button('Save', 'save', None, lambda: t_relation.save(f'/hardware/{hardware.uid}/'))
	])
	tests_relation_form(t_relation)

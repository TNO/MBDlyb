# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from typing import Optional

from nicegui import ui, APIRouter

from mbdlyb.functional.gdb import Cluster, OperatingMode
from mbdlyb.ui.helpers import get_object_or_404

from .base import Button, page, confirm_delete
from .helpers import goto, save_object, save_new_object
from .validation import base_name_validation


router = APIRouter()


class FormOperatingMode:
	name: str
	old_name: str
	added: bool
	dropped: bool

	def __init__(self, name: str, new=False):
		self.name = name
		self.old_name = name
		self.added = new
		self.dropped = False

	def __eq__(self, other):
		if isinstance(other, FormOperatingMode):
			return self.name == other.name
		return False

	def drop(self):
		self.dropped = True

	@staticmethod
	def from_list(l: list[str]) -> list['FormOperatingMode']:
		return [FormOperatingMode(i) for i in l]

	@staticmethod
	def to_list(l: list['FormOperatingMode']) -> list[str]:
		return [i.name for i in l if i.dropped == False]

	@staticmethod
	def updates_list(l: list['FormOperatingMode']) -> dict[str,Optional[str]]:
		updates = {}
		for opm in l:
			if opm.added:
				continue
			elif opm.dropped:
				updates[opm.old_name] = None
			elif opm.name != opm.old_name:
				updates[opm.old_name] = opm.name
		return updates


def add_opm(form_operating_modes: list[FormOperatingMode]):
	operating_modes = FormOperatingMode.to_list(form_operating_modes)
	name = 'OperatingMode_'
	for i in range(1, 100):
		if name + str(i) not in operating_modes:
			form_operating_modes.append(FormOperatingMode(name + str(i), new=True))
			break
	operating_mode_list.refresh()


def drop_opm(opm: FormOperatingMode):
	opm.drop()
	operating_mode_list.refresh()


def move_opm(operating_modes: list[FormOperatingMode], opm: FormOperatingMode, direction: int):
	index = operating_modes.index(opm)
	operating_modes.remove(opm)
	operating_modes.insert(index + direction, opm)
	operating_mode_list.refresh()


@ui.refreshable
def operating_mode_list(operating_modes: list[FormOperatingMode]):
	with ui.grid(columns='auto 15px 15px').classes('vertical-top gap-x-0'):
		for col in ('Operating modes',):
			ui.label(col).classes('font-bold w-64')
		ui.icon('add', color='primary').on('click', lambda: add_opm(operating_modes)).classes('cursor-pointer')
		ui.space()
		operating_modes_active = [opm for opm in operating_modes if opm.dropped == False]
		opm_inputs = []
		for i, opm in enumerate(operating_modes_active):
			opm_inputs.append(ui.input('Name', validation=base_name_validation()).bind_value(opm, 'name').classes('w-64').props('hide-bottom-space'))
			ui.icon('delete', color='negative').on('click', lambda f=opm: drop_opm(f)).classes('self-center pt-3 cursor-pointer')
			with ui.column().classes('self-center pt-2 gap-x-0 gap-y-0'):
				if i > 0:
					ui.icon('arrow_upward').classes('cursor-pointer').on('click', lambda f=opm: move_opm(operating_modes, f, -1))
				else:
					ui.icon('arrow_upward').classes('opacity-25')
				if i < len(operating_modes_active) - 1:
					ui.icon('arrow_downward').classes('cursor-pointer').on('click', lambda f=opm: move_opm(operating_modes, f, 1))
				else:
					ui.icon('arrow_downward').classes('opacity-25')
	for input in opm_inputs:
		input.validation['Operating modes must be unique'] = \
			lambda s: sum([i.value == s for i in opm_inputs]) == 1
		for i in opm_inputs:
			if i is not input:
				input.on_value_change(i.validate)


def opm_form(opm: OperatingMode, operating_modes: list[FormOperatingMode], cluster: Cluster, update: bool = True):
	ui.input('Name', validation=base_name_validation(cluster, opm if update else None)).bind_value(opm, 'name').classes('w-64 pb-6').props('hide-bottom-space')
	operating_mode_list(operating_modes)


@router.page('/cluster/{cluster_id}/opm/new/')
def opm_create(cluster_id: str):
	def _save(opm: OperatingMode, operating_modes: list[FormOperatingMode], cluster: Cluster):
		opm.operating_modes = FormOperatingMode.to_list(operating_modes)
		save_new_object(opm, cluster)

	cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return
	opm = OperatingMode(name='', operating_modes=[])
	operating_modes = FormOperatingMode.from_list(['OperatingMode_1', 'OperatingMode_2'])
	page(f'New operating mode in {cluster.name}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save(opm, operating_modes, cluster))
	])
	opm_form(opm, operating_modes, cluster, False)


@router.page('/opm/{opm_id}')
def opm(opm_id: str):
	opm: OperatingMode = get_object_or_404(OperatingMode, uid=opm_id)
	if opm is None:
		return
	cluster = opm.get_net()
	buttons = [
		Button(None, 'edit', None, lambda: goto(f'/opm/{opm.uid}/update/'), 'Edit operating mode'),
		Button(None, 'delete', 'negative', lambda: confirm_delete(opm, f'/cluster/{cluster.uid}/'), 'Delete operating mode')]
	page(f'Operating mode {opm.name}', cluster, buttons)

	with ui.list().props('dense separator'):
		ui.separator()
		ui.item_label('Operating modes').props('header').classes('text-bold')
		ui.separator()
		for m in opm.operating_modes:
			ui.item(m)
		ui.separator()


@router.page('/opm/{opm_id}/update/')
def opm_update(opm_id: str):
	def _save(opm: OperatingMode, operating_modes: list[FormOperatingMode]):
		updates = FormOperatingMode.updates_list(operating_modes)
		OperatingMode.update_relations_with_operating_mode(updates)
		opm.operating_modes = FormOperatingMode.to_list(operating_modes)
		save_object(opm, f'/cluster/{cluster.uid}/')
		operating_modes = [opm for opm in operating_modes if opm.dropped == False]
		for opm in operating_modes:
			opm.old_name = opm.name
			opm.added = False

	opm: OperatingMode = get_object_or_404(OperatingMode, uid=opm_id)
	if opm is None:
		return
	cluster = opm.get_net()
	operating_modes = FormOperatingMode.from_list(opm.operating_modes)
	page(f'Update {opm.fqn}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save(opm, operating_modes))
	])
	opm_form(opm, operating_modes, cluster)

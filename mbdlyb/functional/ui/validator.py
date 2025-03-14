# -*- coding: utf-8 -*-
"""
	Copyright (c) 2025 TNO-ESI
	All rights reserved.
"""
from nicegui import ui, APIRouter, run

from mbdlyb.functional.gdb import Cluster, FunctionalRelation, Error
from mbdlyb.ui.helpers import get_object_or_404, goto
from mbdlyb.ui.utils import Status, status, show_name

from .base import header, footer


router = APIRouter(prefix='/cluster/{cluster_id}')


class UIValidatorData:
	_cluster_id: str = None
	cluster: Cluster = None

	errors: list[Error]

	def __init__(self, cluster_id: str):
		self._cluster_id = cluster_id
		self.errors = []

	def reset(self):
		self.cluster = get_object_or_404(Cluster, uid=self._cluster_id)
		self.errors = []
		show_name.refresh(self.cluster.name)
		status.refresh(Status.READY)

	async def validate(self):
		status.refresh(Status.COMPUTING)
		self.errors = []
		self.cluster.refresh()
		self.errors = await run.io_bound(self.cluster.check_errors)
		show_validation_results.refresh(self.errors)
		status.refresh(Status.READY)


@router.page('/validator/')
def validator_main(cluster_id: str):
	data: UIValidatorData = UIValidatorData(cluster_id)

	header(f'Functional Model Validator')
	footer()

	with ui.row().classes('w-full gap-5 items-center'):
		show_name('Loading...')
		status(Status.COMPUTING)
		btn = ui.button('Validate', icon='refresh', color='primary').on('click', lambda e: data.validate())
		ui.space()
		ui.button('Editor', icon='home', color='primary').on_click(lambda: goto(f'/cluster/{cluster_id}'))

	show_validation_results(data.errors, False)

	data.reset()
	btn.run_method('click')


@ui.refreshable
def show_validation_results(errors: list[Error], validated: bool = True):
	if errors:
		ui.label(
			f'{len(errors)} error{"s were" if len(errors) > 1 else " was"} found:').classes('text-negative')
		with ui.grid(columns=2).classes('gap-0'):
			for error in errors:
				if error.has_fix_url:
					ui.link(error.fix_label, error.fix_url, True).classes('text-negative').classes('border p-1')
				else:
					ui.label('').classes('border p-1')
				ui.label(error.message).classes('text-negative').classes('border p-1')
	elif validated:
		ui.label('No issues were found.').classes('text-positive')

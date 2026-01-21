# -*- coding: utf-8 -*-
"""
	Copyright (c) 2025 - 2026 TNO-ESI
	All rights reserved.
"""
from nicegui import ui, APIRouter, run, app

from mbdlyb.functional.gdb import Cluster
from mbdlyb.ui.messages import ValidationMessage
from mbdlyb.ui.helpers import get_object_or_404, goto
from mbdlyb.ui.utils import Status, status, show_name

from .base import header, footer


router = APIRouter(prefix='/cluster/{cluster_id}')


class UIValidatorData:
	_cluster_id: str = None
	cluster: Cluster = None

	messages: list[ValidationMessage]

	def __init__(self, cluster_id: str):
		self._cluster_id = cluster_id
		self.messages = []

	def reset(self):
		self.cluster = get_object_or_404(Cluster, uid=self._cluster_id)
		self.messages = []
		show_name.refresh(self.cluster.name)
		status.refresh(Status.READY)

	async def validate(self):
		status.refresh(Status.COMPUTING)
		self.messages = []
		self.cluster.refresh()
		self.messages = sorted(await run.io_bound(self.cluster.validate))
		show_validation_results.refresh(self.messages)
		status.refresh(Status.READY)


@router.page('/validator/')
def validator_main(cluster_id: str):
	data: UIValidatorData = UIValidatorData(cluster_id)
	editor_url = f'/cluster/{cluster_id}/'

	header(f'Functional Model Validator')
	footer()

	with ui.row().classes('w-full gap-5 items-center'):
		show_name('Loading...')
		status(Status.COMPUTING)
		btn = ui.button('Validate', icon='refresh', color='primary').on('click', lambda e: data.validate())
		ui.space()
		ui.button(app.storage.general['mode'], icon='home', color='primary').on_click(lambda: goto(editor_url))

	show_validation_results(data.messages, False)

	data.reset()
	btn.run_method('click')


@ui.refreshable
def show_validation_results(messages: list[ValidationMessage], validated: bool = True):
	if messages:
		highest_prio = 9
		text_class = None
		for m in messages:
			if m.priority < highest_prio:
				highest_prio = m.priority
				text_class = m.color
		ui.label(f'{len(messages)} validation rule{"s were" if len(messages) > 1 else " was"} violated:').classes(
			f'text-{text_class}')
		with ui.grid(columns=2).classes('gap-0'):
			for message in messages:
				if message.has_fix_url:
					ui.link(message.fix_label, message.fix_url, True).classes(f'text-{message.color} border p-1')

				else:
					ui.label('').classes('border p-1')
				ui.label(message.message).classes(f'text-{message.color}').classes('border p-1')
	elif validated:
		ui.label('No issues were found.').classes('text-positive')

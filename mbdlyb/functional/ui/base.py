# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from dataclasses import dataclass
from typing import Optional, Callable, Iterable

from nicegui import app, ui

from mbdlyb.functional.gdb import MBDElement, Cluster, FunctionalNode
from mbdlyb.ui.helpers import goto


@dataclass
class Button:
	'''Class for keeping track of button options'''
	label: Optional[str] = None
	icon: Optional[str] = None
	color: Optional[str] = None
	handler: Optional[str] = None
	tooltip: Optional[str] = None


def build_menu_tree(parent: Cluster = None):
	return [
		{'id': cluster.uid, 'label': cluster.name, 'children': build_menu_tree(cluster)} for cluster in
		(Cluster.nodes.has(net=False).order_by('name') if parent is None else parent.subnets.order_by('name'))
	]


def _buttons(buttons: list[list | Button]):
	for button in buttons:
		if isinstance(button, Button):
			with ui.button(button.label or '', icon=button.icon, color=button.color or 'primary', on_click=button.handler).props('outline'):
				if button.tooltip:
					ui.tooltip(button.tooltip).classes('text-xs')
		elif isinstance(button, list) and len(button) >= 2:
			with ui.button_group():
				_buttons(button)
		else:
			import warnings
			warnings.warn('Could not add buttons to page.')


def _settings(dialog: ui.dialog):
	def emit_show_diagram():
		ui.run_javascript('''emitEvent('show_diagram', '');''')

	with dialog, ui.card():
		ui.label('Settings').classes('text-h5')
		ui.switch('Model editor - show diagrams').bind_value(app.storage.general, 'show_diagrams').on_value_change(emit_show_diagram)
		ui.switch('Functional diagnoser - auto-compute').bind_value(app.storage.general, 'auto_compute')


def header(text: str, url: str = '/'):
	ui.page_title(text)
	with ui.header(elevated=True).style('background-color: #3874c8'):
		with ui.link(target=url).classes(remove='nicegui-link'):
			ui.label(text).classes('text-h4')
		ui.space()
		dialog = ui.dialog()
		_settings(dialog)
		ui.icon('settings').classes('text-h4').on('click', dialog.open)


def footer():
	with ui.footer(elevated=False):
		ui.label(
			'This prototype tool has been developed by TNO-ESI. All rights reserved.').classes('fit text-right')


def page(title: str = None, cluster: Cluster = None, buttons=None):
	def _cluster_trace(c: Cluster) -> list[Cluster]:
		return (_cluster_trace(c.get_net()) if not c.at_root else []) + [c]

	cluster_trace = None if cluster is None else _cluster_trace(cluster)

	header('Model Editor')
	with ui.left_drawer(bottom_corner=True).style('background-color: #d7e3f4'):
		menu = ui.tree(build_menu_tree(), on_select=lambda c_id: goto(f'/cluster/{c_id.value}'))
		if cluster_trace is not None:
			menu.expand([c.uid for c in cluster_trace])
	footer()
	if cluster_trace:
		with ui.row():
			for c in cluster_trace:
				ui.link(c.name, target=f'/cluster/{c.uid}/')
		ui.separator()
	if title:
		with ui.row().classes('w-full'):
			ui.label(title).classes('text-h5')
			if buttons:
				ui.space()
				_buttons(buttons)


def build_table(title: str, headers: list[tuple], rows: Iterable, detail_url: Optional[str] = None,
				create_url: Optional[str] = None, edit_fn: Optional[Callable] = None,
				delete_fn: Optional[Callable] = None, actions: list[tuple[str, Callable]] = ()):
	has_actions = actions or edit_fn is not None or delete_fn is not None
	with ui.row().classes('w-full'):
		ui.label(title).classes('text-h6')
		if create_url:
			ui.space()
			with ui.button(icon='add', color='positive', on_click=lambda: goto(create_url)).props('outline'):
				if 'Weight' in [name for (name, _) in headers]:
					tooltip_text = f'Add "{title.lower()}" relation'
					tooltip_title = f'"{title.lower()}" relation with'
				else:
					tooltip_text = f'Add {title.lower().rstrip("s")}'
					tooltip_title = title.lower().rstrip("s")
				ui.tooltip(tooltip_text).classes('text-xs')

	with ui.grid(columns=' '.join(['auto'] * len(headers)) + (' 1fr' if has_actions else '')).classes('w-full gap-0'):
		if not rows:
			ui.label('No entries found.')
			return
		for label, _ in headers:
			ui.markdown(f'**{label}**').classes('border p-3')
		if has_actions:
			ui.label().classes('border p-3')
		for o in rows:
			for idx, (_, attribute) in enumerate(headers):
				label = o.__getattribute__(attribute) if isinstance(attribute, str) else attribute[1](
					o.__getattribute__(attribute[0]))
				(ui.link(label, detail_url.format(o.uid)) if detail_url and idx == 0 else ui.label(label)).classes(
					'border p-3')
			if has_actions:
				name = o.__getattribute__(headers[0][1])
				with ui.element('div').classes('border p-3 w-full text-right'):
					for icon, fn in actions:
						if icon == 'subdirectory_arrow_right':
							tooltip_text = f'Edit subfunctions of "{name}"'
						elif icon == 'table_view':
							tooltip_text = f'Edit tested items of "{name}"'
						else:
							None
						ui.icon(icon, color='primary').on('click', lambda x=o: fn(x)).classes('cursor-pointer').tooltip(tooltip_text)
					if edit_fn is not None:
						tooltip_text = f'Edit {tooltip_title} "{name}"'
						ui.icon('edit').on('click', lambda x=o: edit_fn(x)).classes('cursor-pointer').tooltip(tooltip_text)
					if delete_fn is not None:
						tooltip_text = f'Delete {tooltip_title} "{name}"'
						ui.icon('delete', color='negative').on('click', lambda x=o: delete_fn(x)).classes(
							'cursor-pointer').tooltip(tooltip_text)


# GENERIC DIALOGS

def confirm(title: str, message: str, confirm_text: str, on_confirm: Callable, confirm_color: str = 'primary'):
	def _on_cancel():
		dialog.close()
		dialog.clear()

	def _on_confirm():
		on_confirm()
		_on_cancel()

	with ui.dialog(value=True) as dialog, ui.card():
		ui.label(title).classes('text-h6')
		ui.label(message)
		with ui.row():
			ui.button('Cancel', color='secondary', on_click=_on_cancel).props('outline')
			ui.button(confirm_text, color=confirm_color, on_click=_on_confirm)
		dialog.on('close', dialog.clear)


def confirm_delete(obj, redirect_url: str):
	def _delete():
		if hasattr(obj, 'on_delete') and callable(obj.on_delete):
			obj.on_delete()
		obj.delete()
		goto(redirect_url)

	confirm('Delete confirmation', f'Are you sure you want to delete {type(obj).__name__} {obj}?', 'Yes', _delete,
			'negative')


def confirm_delete_relation(relation: str, start: MBDElement, end: MBDElement, redirect_url: str):
	def _delete():
		start.__getattribute__(relation).disconnect(end)
		goto(redirect_url)

	confirm('Delete confirmation',
			f'Are you sure you want to delete \'{relation}\' relation from  {start.fqn} to {end.fqn}?', 'Yes', _delete,
			'negative')

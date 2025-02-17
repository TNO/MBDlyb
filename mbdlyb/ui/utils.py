# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
import enum

from nicegui import ui


class Status(enum.Enum):
	READY = enum.auto()
	COMPUTING = enum.auto()
	ERROR = enum.auto()

	def print_ui(self):
		if self == Status.READY:
			ui.icon('check', color='positive', size='lg').tooltip('Ready')
		elif self == Status.COMPUTING:
			ui.spinner(color='primary', size='lg').tooltip('Computing')
		else:
			ui.icon('error', color='warning', size='lg').tooltip('Error')


@ui.refreshable
def show_name(name: str):
	ui.label(name).classes('text-h6')


@ui.refreshable
def status(status: Status):
	status.print_ui()

# -*- coding: utf-8 -*-
"""
	Copyright (c) 2025 - 2026 TNO-ESI
	All rights reserved.
"""

class ValidationMessage:
	PRIORITY: int
	COLOR: str

	message: str
	fix_label: str
	fix_url: str

	def __init__(self, message: str, fix_label: str = None, fix_url: str = None):
		self.message = message
		self.fix_label = fix_label
		self.fix_url = fix_url

	def __lt__(self, other):
		if isinstance(other, ValidationMessage):
			return self.priority < other.priority or self.message < other.message
		return self.message < str(other)

	def __repr__(self):
		return f'{self.__class__.__name__}: {self.message}' + (
			f'({self.fix_label}, {self.fix_url}' if self.has_fix_url else '')

	@property
	def priority(self) -> int:
		return self.PRIORITY

	@property
	def color(self) -> str:
		return self.COLOR

	@property
	def has_fix_url(self) -> bool:
		return bool(self.fix_url) and bool(self.fix_label)


class Error(ValidationMessage):
	PRIORITY = 2
	COLOR = 'error'


class Warning(ValidationMessage):
	PRIORITY = 4
	COLOR = 'warning'

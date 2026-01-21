# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""
from __future__ import annotations  # PEP 563 - Postponed Evaluation of Annotations - https://peps.python.org/pep-0563/
from typing import Optional, Union


# Naming convention: 'Operating Mode' has been abbreviated to 'Opm'

class OpmSet:
	_variables: list[OpmVariable] = None

	def __init__(self, variables: list[str | OpmVariable] = None):
		self._variables = []
		if variables:
			self.add_variables(variables)

	def __len__(self) -> int:
		return len(self._variables)

	def __repr__(self) -> str:
		return '{' + ', '.join(self.variable_names) + '}'

	@property
	def variable_names(self) -> list[str]:
		return sorted([v.name for v in self._variables])

	def add_variables(self, opm_vars: list[str | OpmVariable]):
		for v in opm_vars:
			if isinstance(v, str):
				v = OpmVariable(v)
			# assign variable to set and vice-versa
			v.set_opmset(self)
			if v not in self._variables:
				self._variables.append(v)

	def contains_variable(self, var: OpmVariable) -> bool:
		return var in self._variables


class OpmVariable:
	_name: str
	_set: OpmSet | None

	def __init__(self, name, opm_set: OpmSet = None):
		self._name = name
		self._set = opm_set

	def __eq__(self, other: OpmVariable) -> bool:
		return self._name == other._name

	def __repr__(self) -> str:
		return self._name

	@property
	def name(self) -> str:
		return self._name

	@property
	def opmset(self) -> OpmSet:
		return self._set

	def set_opmset(self, opm_set: OpmSet) -> OpmSet:
		self._set = opm_set
		return self._set

	def eval(self, active_modes: list[str]) -> bool:
		# Evaluation of an OpmVariable in three situations:
		#   1. the variable is in the list of active operating modes -> return True
		#   2. another variable of the set this variable belongs to is in the list of active operating modes -> return False
		#   3. none of the variables this set belongs to are in the list of active operating modes -> return True
		if self._name in active_modes:
			return True
		elif any([n in active_modes for n in self._set.variable_names]):
			return False
		else:
			return True


class OpmLogic:
	var_separator: str = ', '
	logic_function = lambda self, _: True

	_vars: list[Union[OpmVariable, OpmLogic]] = None

	@classmethod
	def from_copy(cls, other) -> OpmLogic:
		new_obj = other.__class__()
		new_obj._vars = [v for v in other._vars]
		return new_obj

	# main function to create OpmLogic and OpmVariables during parsing of input files
	@classmethod
	def create(cls, modes: list[str], opm_set: OpmSet = None) -> OpmLogicOr:
		opm_vars = [OpmVariable(m, opm_set) for m in modes]
		return OpmLogicOr(opm_vars)

	@property
	def variables(self) -> list[OpmVariable]:
		vars: list[OpmVariable] = list()
		for v in self._vars:
			if isinstance(v, OpmLogic):
				vars.extend(v.variables)
			else:
				vars.append(v)
		return vars

	@property
	def sets(self) -> set[OpmSet]:
		return set([v.opmset for v in self.variables])

	def __init__(self, opm_vars: Optional[list[OpmVariable]] = None):
		self._vars = opm_vars if opm_vars is not None else []

	def __len__(self) -> int:
		return len(self._vars)

	def __repr__(self) -> str:
		if len(self._vars) == 0:
			return ''
		elif len(self._vars) == 1:
			return str(self._vars[0])
		else:
			return '(' + self.var_separator.join([str(v) for v in self._vars]) + ')'

	def __bool__(self):
		return bool(self._vars)

	def add(self, opm_logic: OpmLogic) -> OpmLogic:
		if len(opm_logic.variables) > 0:
			self._vars.append(opm_logic)
		return self

	def eval(self, active_modes: list[str]) -> bool:
		if len(self._vars) == 0:
			return True
		return self.logic_function([v.eval(active_modes) for v in self._vars])

	def unwrap(self) -> OpmVariable | OpmLogic:
		# remove an unnecessary operation with only one operand
		return self._vars[0] if len(self._vars) == 1 else self


class OpmLogicAnd(OpmLogic):
	var_separator: str = ' & '
	logic_function = all


class OpmLogicOr(OpmLogic):
	var_separator: str = ' | '
	logic_function = any

"""Typed Medallion contracts with stable Python import aliases."""
from importlib import import_module as _import_module
from sys import modules as _modules

# Numbered folders are not valid names in ordinary Python import statements.
for _alias, _folder in (("bronze", "01_bronze"), ("silver", "02_silver"), ("gold", "03_gold")):
    _package = _import_module(f".{_folder}", __name__)
    _modules[f"{__name__}.{_alias}"] = _package
    globals()[_alias] = _package

del _alias, _folder, _package, _import_module, _modules

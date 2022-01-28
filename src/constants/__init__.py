import importlib


def get_constant(name):
    constant = getattr(importlib.import_module(f"constants.{name}"), name)
    return constant

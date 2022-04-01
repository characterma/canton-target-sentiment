import importlib


def get_constant(name):
    constant = getattr(importlib.import_module(f"nlp_pipeline.constants.{name}"), name)
    return constant

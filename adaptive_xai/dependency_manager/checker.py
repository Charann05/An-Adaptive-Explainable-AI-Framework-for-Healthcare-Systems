import importlib.util

def is_installed(import_name):
    return importlib.util.find_spec(import_name) is not None
import os

STORAGE_DIR = os.path.join(os.sep, 'project', 'storage')
RESOURCES_DIR = os.path.join(STORAGE_DIR, 'resources')


def experiment_path(module_path: str, name: str):
    return os.path.join(os.sep, 'project', module_path, name)

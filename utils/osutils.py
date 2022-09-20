import os


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
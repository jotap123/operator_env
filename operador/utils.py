from pathlib import Path


def get_root_dir():
    cur_dir = Path(__file__).resolve().parent

    while cur_dir.name != 'root': # your root here
        cur_dir = cur_dir.parent
    return str(cur_dir)

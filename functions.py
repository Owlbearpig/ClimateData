import os
from pathlib import Path

def find_files(top_dir, search_str='', file_extension=''):
    results = [Path(os.path.join(root, name))
               for root, dirs, files in os.walk(top_dir)
               for name in files if name.endswith(file_extension) and search_str in str(name)]
    return results
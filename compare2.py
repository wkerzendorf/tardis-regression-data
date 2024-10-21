import os
import pathlib
import shutil
import subprocess
import tempfile
from filecmp import dircmp
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode

init_notebook_mode()

# Configuration
CONFIG = {
    'compare_path': 'tardis',
    'temp_dir_prefix': 'ref_compare_',
}

# Utility functions
def highlight_missing(val):
    return 'background-color: #BCF5A9' if val == True else 'background-color: #F5A9A9'

def highlight_relative_difference(val):
    if val is None or val <= 1e-2:
        return 'background-color: #BCF5A9'
    elif val <= 1e-1:
        return 'background-color: #F2F5A9'
    elif val <= 1:
        return 'background-color: #F5D0A9'
    else:
        return 'background-color: #F5A9A9'

def color_print(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def get_relative_path(path, base):
    return str(Path(path).relative_to(base))

class FileManager:
    def __init__(self):
        self.temp_dir = None

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix=CONFIG['temp_dir_prefix'])
        print(f'Created temporary directory at {self.temp_dir}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
            print(f'Removed temporary directory {self.temp_dir}')

    def get_temp_path(self, filename):
        return os.path.join(self.temp_dir, filename)

    def copy_file(self, source, destination):
        shutil.copy2(source, self.get_temp_path(destination))

class FileSetup:
    def __init__(self, file_manager, ref1_hash, ref2_hash):
        self.file_manager = file_manager
        self.ref1_hash = ref1_hash
        self.ref2_hash = ref2_hash

    def setup(self):
        for ref_id, ref_hash in enumerate([self.ref1_hash, self.ref2_hash], 1):
            if ref_hash:
                self._copy_data_from_hash(ref_hash, f'ref{ref_id}_')
            else:
                subprocess.run(f'cp {CONFIG["compare_path"]} {self.file_manager.get_temp_path(f"ref{ref_id}_{CONFIG["compare_path"]}")}', shell=True)

    def _copy_data_from_hash(self, ref_hash, prefix):
        git_cmd = ['git', f'--work-tree={self.file_manager.temp_dir}', 'checkout', ref_hash, CONFIG['compare_path']]
        subprocess.run(git_cmd)
        shutil.move(
            self.file_manager.get_temp_path(CONFIG['compare_path']),
            self.file_manager.get_temp_path(prefix + CONFIG['compare_path'])
        )

class DiffAnalyzer:
    def __init__(self, file_manager):
        self.file_manager = file_manager

    def display_diff_tree(self, dcmp, prefix=''):
        for item in sorted(dcmp.left_only):
            path = Path(dcmp.left) / item
            self._print_item(f'{prefix}−', item, 'red', path.is_dir())

        for item in sorted(dcmp.right_only):
            path = Path(dcmp.right) / item
            self._print_item(f'{prefix}+', item, 'green', path.is_dir())

        for item in sorted(dcmp.diff_files):
            self._print_item(f'{prefix}✱', item, 'yellow')

        for item in sorted(dcmp.common_dirs):
            self._print_item(f'{prefix}├', item, 'blue', True)
            subdir = getattr(dcmp, 'subdirs')[item]
            self.display_diff_tree(subdir, prefix + '│ ')

    def _print_item(self, symbol, item, color, is_dir=False):
        dir_symbol = '/' if is_dir else ''
        color_print(f"{symbol} {item}{dir_symbol}", color)

    def print_diff_files(self, dcmp):
        dcmp.right = str(dcmp.right)
        dcmp.left = str(dcmp.left)
        
        self._print_new_files(dcmp.right_only, dcmp.right, "ref1")
        self._print_new_files(dcmp.left_only, dcmp.left, "ref2")
        self._print_modified_files(dcmp)

        for sub_dcmp in dcmp.subdirs.values():
            self.print_diff_files(sub_dcmp)

    def _print_new_files(self, items, base_path, ref_name):
        for item in items:
            if Path(base_path + "/" + item).is_file():
                print(f"New file detected inside {ref_name}: {item}")
                print(f"Path: {base_path + '/' + item}")
                print()

    def _print_modified_files(self, dcmp):
        for name in dcmp.diff_files:
            print(f"Modified file found {name}")
            left = get_relative_path(dcmp.left, self.file_manager.temp_dir + "/ref1_tardis/")
            right = get_relative_path(dcmp.right, self.file_manager.temp_dir + "/ref2_tardis/")
            if left == right:
                print(f"Path: {left}")
            print()

class HDFComparator:
    def summarise_changes_hdf(self, name, path1, path2):
        ref1 = pd.HDFStore(path1 + "/" + name)
        ref2 = pd.HDFStore(path2 + "/" + name)
        k1, k2 = set(ref1.keys()), set(ref2.keys())
        
        print(f"Total number of keys- in ref1: {len(k1)}, in ref2: {len(k2)}")
        different_keys = len(k1 ^ k2)
        print(f"Number of keys with different names in ref1 and ref2: {different_keys}")

        identical_items = []
        identical_name_different_data = []
        identical_name_different_data_dfs = {}

        for item in k1 & k2:
            try:
                if ref2[item].equals(ref1[item]):
                    identical_items.append(item)
                else:
                    self._compare_and_display_differences(ref1[item], ref2[item], item, name)
                    identical_name_different_data.append(item)
                    identical_name_different_data_dfs[item] = (ref1[item] - ref2[item]) / ref1[item]
            except Exception as e:
                print(f"Error comparing item: {item}")
                print(e)

        print(f"Number of keys with same name but different data in ref1 and ref2: {len(identical_name_different_data)}")
        print(f"Number of totally same keys: {len(identical_items)}")
        print()

        return {
            "different_keys": different_keys,
            "identical_keys": len(identical_items),
            "identical_keys_diff_data": len(identical_name_different_data),
            "identical_name_different_data_dfs": identical_name_different_data_dfs
        }

    def _compare_and_display_differences(self, df1, df2, item, name):
        abs_diff = np.fabs(df1 - df2)
        rel_diff = (abs_diff / np.fabs(df1))[df1 != 0]
        
        print(f"Displaying heatmap for key {item} in file {name}")
        for diff_type, diff in zip(["abs", "rel"], [abs_diff, rel_diff]):
            print(f"Visualising {'Absolute' if diff_type == 'abs' else 'Relative'} Differences")
            self._display_difference(diff)
        print("\n")

    def _display_difference(self, diff):
        with pd.option_context('display.max_rows', 100, 'display.max_columns', 10):
            if isinstance(diff, pd.Series):
                diff = pd.DataFrame([diff.mean(), diff.max()], index=['mean', 'max'])
            elif isinstance(diff.index, pd.core.indexes.multi.MultiIndex):
                diff = diff.reset_index(drop=True)
            
            diff = pd.DataFrame([diff.mean(), diff.max()], index=['mean', 'max'])
            display(diff.style.format('{:.2g}'.format).background_gradient(cmap='Reds'))

class ReferenceComparer:
    def __init__(self, ref1_hash=None, ref2_hash=None):
        assert not ((ref1_hash is None) and (ref2_hash is None)), "One hash can not be None"
        self.ref1_hash = ref1_hash
        self.ref2_hash = ref2_hash
        self.test_table_dict = {}
        self.file_manager = None
        self.file_setup = None
        self.diff_analyzer = None
        self.hdf_comparator = None

    def setup(self):
        self.file_manager = FileManager()
        self.file_setup = FileSetup(self.file_manager, self.ref1_hash, self.ref2_hash)
        self.diff_analyzer = DiffAnalyzer(self.file_manager)
        self.hdf_comparator = HDFComparator()

    def compare(self):
        with self.file_manager:
            self.file_setup.setup()
            self.ref1_path = self.file_manager.get_temp_path(f"ref1_{CONFIG['compare_path']}")
            self.ref2_path = self.file_manager.get_temp_path(f"ref2_{CONFIG['compare_path']}")
            self.dcmp = dircmp(self.ref1_path, self.ref2_path)
            self.diff_analyzer.print_diff_files(self.dcmp)
            self.compare_hdf_files()

    def compare_hdf_files(self):
        for root, dirs, files in os.walk(self.ref1_path):
            for file in files:
                if file.endswith('.h5') or file.endswith('.hdf5'):
                    rel_path = os.path.relpath(root, self.ref1_path)
                    ref2_file_path = os.path.join(self.ref2_path, rel_path, file)
                    if os.path.exists(ref2_file_path):
                        self.summarise_changes_hdf(file, root, os.path.dirname(ref2_file_path))

    def summarise_changes_hdf(self, name, path1, path2):
        self.test_table_dict[name] = {
            "path": get_relative_path(path1, self.file_manager.temp_dir + "/ref1_tardis/")
        }
        self.test_table_dict[name].update(
            self.hdf_comparator.summarise_changes_hdf(name, path1, path2)
        )

    def display_hdf_comparison_results(self):
        for name, results in self.test_table_dict.items():
            print(f"Results for {name}:")
            for key, value in results.items():
                print(f"  {key}: {value}")
            print()

if __name__ == '__main__':
    comparer = ReferenceComparer(ref1_hash="hash1", ref2_hash="hash2")
    comparer.setup()
    comparer.compare()
    comparer.display_hdf_comparison_results()
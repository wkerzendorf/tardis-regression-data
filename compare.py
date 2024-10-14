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

def highlight_missing(val):
    if val == True:
        return 'background-color: #BCF5A9'
    else:
        return 'background-color: #F5A9A9'
    
def highlight_relative_difference(val):
    ret = 'background-color: #BCF5A9'
    if val is None:
        ret = 'background-color: #BCF5A9'
    elif val > 1e-2:
        ret = 'background-color: #F2F5A9'
    elif val > 1e-1:
        ret = 'background-color: #F5D0A9'
    elif val > 1:
        ret = 'background-color: #F5A9A9'
    return ret

class ReferenceComparer(object):
    def __init__(self, ref1_hash=None, ref2_hash=None):
        assert not ((ref1_hash is None) and (ref2_hash is None)), "One hash can not be None"
        self.test_table_dict = {}
        self.ref1_hash = ref1_hash
        self.ref2_hash = ref2_hash
        self.compare_path = "tardis"
        self.tmp_dir = None
        self.setup()
        self.tmp_dir = Path(self.tmp_dir)
        self.ref1_path = self.tmp_dir / f"ref1_{self.compare_path}"
        self.ref2_path = self.tmp_dir / f"ref2_{self.compare_path}"
        self.dcmp = dircmp(self.ref1_path, self.ref2_path)
        # self.print_diff_files(self.dcmp)
    
    def setup(self):
        self.tmp_dir = tempfile.mkdtemp()
        print('Created temporary directory at {0}. Delete after use with .teardown'.format(self.tmp_dir))
        
        for ref_id, ref_hash in enumerate([self.ref1_hash, self.ref2_hash]):
            ref_id += 1
            if ref_hash is not None:
                self._copy_data_from_hash(ref_hash, 'ref{0}_'.format(ref_id))
            else:
                subprocess.Popen('cp {0} {1}'.format(self.compare_path, 
                                                     os.path.join(self.tmp_dir, 
                                                                  'ref{0}_{1}'.format(ref_id, self.compare_path))), 
                                                     shell=True)
            setattr(self, 'ref{0}_fname'.format(ref_id), 
                    os.path.join(self.tmp_dir, 'ref{0}_{1}'.format(ref_id, self.compare_path)))

    def teardown(self):
        shutil.rmtree(self.tmp_dir)

    def _copy_data_from_hash(self, ref_hash, prefix):
        git_cmd = ['git']
        git_cmd.append('--work-tree={0}'.format(self.tmp_dir))
        git_cmd += ['checkout', ref_hash, self.compare_path]
        p = subprocess.Popen(git_cmd)
        p.wait()
        shutil.move(os.path.join(self.tmp_dir, self.compare_path), 
                    os.path.join(self.tmp_dir, prefix + self.compare_path))

    def display_diff_tree(self, dcmp, prefix=''):
        def print_item(symbol, item, is_dir=False):
            print(f"{prefix}{symbol} {item}{'/' if is_dir else ''}")

        for item in sorted(dcmp.left_only):
            path = Path(dcmp.left) / item
            print_item('-', item, path.is_dir())

        for item in sorted(dcmp.right_only):
            path = Path(dcmp.right) / item
            print_item('+', item, path.is_dir())

        for item in sorted(dcmp.diff_files):
            print_item('M', item)

        for item in sorted(dcmp.common_dirs):
            print_item(' ', item, True)
            subdir = getattr(dcmp, f'subdirs')[item]
            self.display_diff_tree(subdir, prefix + '  ')

    def print_diff_files(self, dcmp):
        if isinstance(dcmp.right, pathlib.Path):
            dcmp.right = str(dcmp.right)
        if isinstance(dcmp.left, pathlib.Path):
            dcmp.left = str(dcmp.left)
            
        for item in dcmp.right_only:
            if Path(dcmp.right + "/" + item).is_file:
                print(f"new file detected at: {dcmp.right + '/' +item}")
                print(f"New file detected inside ref1: {item}")
                print(f"Path: {dcmp.right + '/' +item}")
                print()
        for item in dcmp.left_only:
            if Path(dcmp.left + "/" + item).is_file:
                print(f"New file detected inside ref2: {item}")
                print(f"Path: {dcmp.left + '/' +item}")
                print()
    
        for name in dcmp.diff_files:
            print(f"Modified file found {name}")
            left = dcmp.left.removeprefix(str(self.tmp_dir) + "/" + "ref1_tardis/")
            right = dcmp.right.removeprefix(str(self.tmp_dir) + "/" + "ref2_tardis/")
            if left==right:
                print(f"Path: {left}")
            if name.endswith(".h5"):
                self.test_table_dict[name] = {
                    "path": left
                }
                self.summarise_changes_hdf(name, str(dcmp.left), str(dcmp.right))
            print()
        for sub_dcmp in dcmp.subdirs.values():
            self.print_diff_files(sub_dcmp)
            
    def summarise_changes_hdf(self, name, path1, path2):
        ref1 = pd.HDFStore(path1 + "/"+ name)
        ref2 = pd.HDFStore(path2 + "/"+ name)
        k1 = set(ref1.keys())
        k2 = set(ref2.keys())
        print(f"Total number of keys- in ref1: {len(k1)}, in ref2: {len(k2)}")
        different_keys = len(k1^k2)
        print(f"Number of keys with different names in ref1 and ref2: {different_keys}")

        identical_items = []
        identical_name_different_data = []
        identical_name_different_data_dfs = {}
        for item in k1&k2:
            try:
                if ref2[item].equals(ref1[item]):
                    identical_items.append(item)
                else:
                    abs_diff = np.fabs(ref1[item] - ref2[item])
                    rel_diff = (abs_diff / np.fabs(ref1[item]))[ref1[item] != 0]
                    print(f"Displaying heatmap for key {item} in file {name}")
                    for diff_type, diff in zip(["abs", "rel"], [abs_diff, rel_diff]):
                        if "abs" in diff_type:
                            print("Visualising Absolute Differences")
                        else:
                            print("Visualising Relative Differences")
                        if isinstance(diff, pd.Series):
                            diff = pd.DataFrame([diff.mean(), diff.max()], index=['mean', 'max'])
                            display(diff)
                        else:
                            with pd.option_context('display.max_rows', 100, 'display.max_columns', 10):
                                if isinstance(diff.index, pd.core.indexes.multi.MultiIndex):
                                    diff = diff.reset_index(drop=True)
                                    diff = pd.DataFrame([diff.mean(), diff.max()], index=['mean', 'max'])
                                    display(diff.style.format('{:.2g}'.format).background_gradient(cmap='Reds'))
                                    
                                else:
                                    diff = pd.DataFrame([diff.mean(), diff.max()], index=['mean', 'max'])
                                    display(diff.style.format('{:.2g}'.format).background_gradient(cmap='Reds'))

                    identical_name_different_data.append(item)
                    identical_name_different_data_dfs[item] = rel_diff
                    print("\n")

            except Exception as e:
                print("Facing error comparing item: ", item)
                print(e)
                
        print(f"Number of keys with same name but different data in ref1 and ref2: {len(identical_name_different_data)}")
        print(f"Number of totally same keys: {len(identical_items)}")
        print()
        self.test_table_dict[name].update({
            "different_keys": different_keys,
            "identical_keys": len(identical_items),
            "identical_keys_diff_data": len(identical_name_different_data),
            "identical_name_different_data_dfs": identical_name_different_data_dfs
            
        })
        

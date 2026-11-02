"""Where relative paths point.

Relative paths in run configs (output_dir, cache_dir) and the research scripts' fixed folders
(runs_real/...) resolve against WORK: $SLABLET_WORK if set, else the current directory.
"""
import os

WORK = os.path.abspath(os.environ.get('SLABLET_WORK', os.getcwd()))

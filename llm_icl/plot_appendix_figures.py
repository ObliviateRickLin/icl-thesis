#!/usr/bin/env python3
"""Generate all appendix figures for pretrained LLM ICL evaluation.
Reads from results_all_sweep/ directory. Outputs PDF figures.
"""
# This is a symlink/copy of the root plot_appendix_figures.py
# See ../plot_appendix_figures.py for the full implementation.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_appendix_figures import *

if __name__ == '__main__':
    plot_fig1()
    plot_fig2()
    plot_fig3()
    print('All figures done.')

#!/bin/bash

source activate myenv

python3 plot_conclusion_bar_sep.py
python3 plot_mae_vs_pb.py
python3 plot_rarity_vs_acc_dpvi.py
python3 plot_scatter_dpvi.py
python3 table_dm_results.py

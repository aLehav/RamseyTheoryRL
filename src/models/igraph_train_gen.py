import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


MODE = "ALL_N" # can either be ALL_N, PATH_SPECIFIC
DF_NAME = 'all_leq6'
# Not to be touched
DF_PATH = f'{DF_NAME}.csv'
TIME_PATH = f'time_{DF_NAME}.csv'

# Settings if PATH_SPECIFIC
PATHS = ''


if(MODE == "ALL_N"):
    MAX_S = 4
    MAX_T = 9
    MAX_N = 6

    from utils.csv_generation.for_all_n import create_entries_up_to_n
    create_entries_up_to_n(max_n=MAX_N,
                           max_s=MAX_S,
                           max_t=MAX_T,
                           time_path=TIME_PATH,
                           df_path=DF_PATH)
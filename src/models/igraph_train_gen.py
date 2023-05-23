import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

MODE = "RAMSEY_SPECIFIC" # can either be FOR_ALL_N, RAMSEY_SPECIFIC

if MODE == "FOR_ALL_N":
    MAX_S = 4
    MAX_T = 9
    MAX_N = 6
    DF_NAME = f'all_leq{MAX_N}'
    DF_PATH = f'{DF_NAME}.csv'
    TIME_PATH = f'time_{DF_NAME}.csv'

    from utils.csv_generation.for_all_n import create_entries_up_to_n
    create_entries_up_to_n(max_n=MAX_N,
                           max_s=MAX_S,
                           max_t=MAX_T,
                           time_path=TIME_PATH,
                           df_path=DF_PATH)
    
if MODE == "RAMSEY_SPECIFIC":
    S = 3
    T = 6
    PATH = f'data/ramsey_s_t_n/{S}_{T}'
    DF_NAME = f'ramsey_{S}_{T}'
    DF_PATH = f'data/csv/{DF_NAME}.csv'
    TIME_PATH = f'data/time/time_{DF_NAME}.csv'

    from utils.csv_generation.ramsey_specific import ramsey_entries_for_path
    ramsey_entries_for_path(path=PATH,
                            time_path=TIME_PATH,
                            df_path=DF_PATH)
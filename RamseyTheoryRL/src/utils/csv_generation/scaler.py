import pandas as pd
import math

def scale(csv_path, new_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path, index_col=0)

    # Remove rows with n < 4
    df = df[df['n'] >= 4]
    
    # Calculate the scaling factor for each row
    scaling_factors = [math.comb(n, 4) for n in df['n']]
    
    # Scale the values in the first 11 columns of each row
    df.iloc[:, :11] = df.iloc[:, :11].div(scaling_factors, axis=0)
    
    # Write the updated DataFrame to a new CSV file
    df.to_csv(new_path)

def main():
    to_be_scaled = ['all_leq6.csv','all_leq9.csv','ramsey_3_4.csv', 'ramsey_3_5.csv', 'ramsey_3_6.csv', 'ramsey_3_7.csv', 'ramsey_3_9.csv', 'ramsey_4_4.csv']
    for csv_path in to_be_scaled:
        scale(f'data/csv/unscaled/{csv_path}', f'data/csv/scaled/{csv_path}')

if __name__ == '__main__':
    main()
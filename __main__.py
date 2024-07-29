# __main__.py

import pandas as pd 
import csv
import argparse
import Feature_Select

def main(input_file):
    df = pd.read_csv(input_file)

    X, Y = Feature_Select.ExtractColumns(df)
    all_acc, MIF = Feature_Select.SelectFeatures(X, Y)
    Top_Features = Feature_Select.GetTopFeatures(MIF, top_n=3, common_n=5)
    print(f"The top features are {Top_Features}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all functions in functions.py")
    parser.add_argument("-i", "--input", required=True, help="Path to the input file")
    #parser.add_argument("-o", "--output", required=True, help="Path to the output file")

    args = parser.parse_args()

    main(args.input)

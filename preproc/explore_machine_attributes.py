import pandas as pd


if __name__ == "__main__":
    df = pd.read_json("data/machine_attributes-000000000000.json", lines=True)
    print(df)

"""
- Split `sample_instance_events.json` and save events of each day to separate
  file located to directory `daily_events`
- Sort by time each of those files
- `transform_features` (i.e. separate cpu and memory columns)
"""


import pandas as pd


NUM_DAYS = 30
START_OFFSET = 600  # simulation starts at second 600
MICROSEC_PER_SEC = 1000000
SEC_PER_DAY = 24*60*60
SAMPLE_EVENTS_PATH = "sustaingym/data/datacenter/sample_instance_events.json"
SAVE_PATH = "sustaingym/data/datacenter/daily_events/"


def transform_features(df):
    """
    Transform column of dicts 'resource_request' into two separate columns 'cpus' and 'memory'.
    """
    cpus = []
    memory = []
    cpu_error_count = 0
    memory_error_count = 0
    for _, row in df.iterrows():
        try:
            cpus.append(row["resource_request"]["cpus"])
        except TypeError:  # float is not subscriptable, sometimes row["resource_request"] is nan
            cpus.append(None)
            cpu_error_count += 1
        try:
            memory.append(row["resource_request"]["memory"])
        except TypeError:
            memory.append(None)
            memory_error_count += 1
    print("ERROR REPORT: Transform features:")
    print(f"\t - CPU errors: {cpu_error_count}/{len(df)}={100*cpu_error_count/len(df)}%")
    print(f"\t - Memory errors: {memory_error_count}/{len(df)}={100*memory_error_count/len(df)}%")
    return cpus, memory


if __name__ == "__main__":
    print("LOG: begin")
    df = pd.read_json(SAMPLE_EVENTS_PATH, lines=True)
    print("LOG: loaded dataframe, transforming features.")

    cpus, memory = transform_features(df)
    df["cpus"] = cpus  # add new columns
    df["memory"] = memory
    df.drop(columns=["resource_request"], inplace=True)  # drop old column
    print(f"LOG: transformed features")
    
    
    for d in range(NUM_DAYS):
        print(f"LOG: day #{d}")
        start_time = START_OFFSET + d*SEC_PER_DAY*MICROSEC_PER_SEC
        end_time = START_OFFSET + (d+1)*SEC_PER_DAY*MICROSEC_PER_SEC
        df_curr_day = df[(df["time"] >= start_time) & (df["time"] < end_time)]
        df_curr_day.sort_values(by="time", inplace=True)
        df_curr_day.to_csv(f"{SAVE_PATH}day_{d}.txt", index=False)

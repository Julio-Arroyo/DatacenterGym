from src.datacenter.cluster import *
import pandas as pd


TASK_DATA_PATH = "data/task_durations.csv"
HOURS = 24
MICROSEC_PER_HOUR = 60*60*1000000
START_DELAY = 600  # trace period starts at 600 seconds
START_DELAY_H = 600 / 3600  # measured in hours


class DatacenterGym:  #TODO: inherit from Gym
    def __init__(self):
        self.datacenter = Cluster()

        # initialize task data
        self.task_data = pd.read_csv(TASK_DATA_PATH)
        bad_rows = self.task_data[(self.task_data['time'] == 0) | (self.task_data['duration'] == 0) | (self.task_data['cpu'] == 0)].index
        self.task_data.drop(bad_rows, inplace=True)

        self.time_window = MICROSEC_PER_HOUR
        self.episode_len = HOURS
    
    def step(self, VCC):
        """
        Returns 3-tuple (state, reward, terminal)
        """
        self.datacenter.stop_finished_tasks()
        self.datacenter.set_VCC(VCC)

        new_tasks = self.get_new_tasks()
        self.datacenter.enqueue_tasks(new_tasks)
        self.datacenter.schedule_tasks()

        self.datacenter.t += 1
        obs = DatacenterState(self.datacenter)

        return (obs, 0, self.datacenter.t >= self.episode_len)

    def get_new_tasks(self) -> list[Task]:
        curr_t = self.datacenter.t
        start = (curr_t + START_DELAY_H)*MICROSEC_PER_HOUR
        end = (curr_t + START_DELAY_H +1)*MICROSEC_PER_HOUR

        new_task_data = self.task_data[(start <= self.task_data['time']) & (self.task_data['time'] < end)]
        tasks = []
        for _, row in new_task_data.iterrows():
            task_duration = (row['duration'] // self.time_window) + 1  # calculate number of timesteps tasks lasts
            tasks.append(Task(row['task_id'], task_duration, row['cpu']))
        return tasks


class DatacenterState:
    def __init__(self, datacenter: Cluster):
        self.VCC = datacenter.VCC
        self.capacity = datacenter.capacity
        self.n_ready_tasks = len(datacenter.task_q)
    
    def display(self):
        print("Datacenter state:")
        print(f"\t- VCC: {self.VCC}")
        print(f"\t- Used capacity: {self.capacity}")
        print(f"\t- # queued tasks: {self.n_ready_tasks}")

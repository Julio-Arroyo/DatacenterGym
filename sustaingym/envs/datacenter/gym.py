from sustaingym.envs.datacenter.cluster import *
import pandas as pd
import gym


TASK_DATA_PATH = "sustaingym/data/datacenter/task_durations.csv"
SIMULATION_LENGTH = 24  # In hours
HOURS_PER_DAY = 24
MICROSEC_PER_HOUR = 60*60*1000000
START_DELAY = 600  # trace period starts at 600 seconds
START_DELAY_H = 600 / 3600  # measured in hours
SIM_START_TIME = datetime(2019, 5, 1)
SIM_END_TIME = datetime(2019, 6, 1)
BALANCING_AUTHORITY = "SGIP_CAISO_PGE"


class DatacenterGym(gym.Env):
    def __init__(self):
        self.datacenter = Cluster(SIMULATION_LENGTH, SIM_START_TIME,
                                  SIM_END_TIME, BALANCING_AUTHORITY)

        # initialize task data
        self.task_data = pd.read_csv(TASK_DATA_PATH)

        # TODO cleanup file instead of doing it in code
        bad_rows = self.task_data[(self.task_data['time'] == 0) |
                                  (self.task_data['duration'] == 0) |
                                  (self.task_data['cpu'] == 0)].index
        self.task_data.drop(bad_rows, inplace=True)

        self.time_window = MICROSEC_PER_HOUR
        self.episode_len = SIMULATION_LENGTH
    
    def make(self):
        # TODO
        return

    def reset(self):
        # TODO
        return super().reset()
    
    def render(self):
        print(self.datacenter.get_state())

    def step(self, VCC):
        """
        Returns 3-tuple (state, reward, terminal)
        """
        self.datacenter.stop_finished_tasks()

        self.datacenter.set_VCC(VCC)

        new_tasks = self.get_new_tasks()
        self.datacenter.enqueue_tasks(new_tasks)
        self.datacenter.schedule_tasks()

        obs = self.datacenter.get_state()
        reward = self.compute_reward()

        self.datacenter.t += 1

        return (obs, reward, self.datacenter.t >= self.episode_len)

    def close(self):
        # TODO
        return super().close()

    def compute_reward(self) -> float:
        """Times -1 to make it reward"""
        SLO_violation_cost = self.datacenter.compute_SLO_violation_cost()
        carbon_cost = self.datacenter.compute_carbon_cost()

        return -1*(SLO_violation_cost + carbon_cost)

    def update_daily_capacity_req(self, task_start_time : int, task : Task):
        """
        TODO document
        """
        curr_day = task_start_time // HOURS_PER_DAY
        curr_hour_of_day = task_start_time % HOURS_PER_DAY
        hours_remaining_in_day = (HOURS_PER_DAY - curr_hour_of_day)

        if task.duration > hours_remaining_in_day:
            curr_day_capacity_req = task.capacity * hours_remaining_in_day
            self.datacenter.add_daily_capacity_req(curr_day, curr_day_capacity_req)

            # count the number of days in the future with an entire day's full worth of capacity
            task_duration_after_today = task.duration - hours_remaining_in_day
            entire_days = task_duration_after_today // HOURS_PER_DAY
            for future_day in range(curr_day + 1, curr_day + 1 + entire_days):
                self.datacenter.add_daily_capacity_req(future_day, task.capacity * HOURS_PER_DAY)
            # add the leftover capacity in the last day
            task_duration_in_last_day = task_duration_after_today % HOURS_PER_DAY
            self.datacenter.add_daily_capacity_req(curr_day + entire_days + 1, task.capacity * task_duration_in_last_day)
        else:
            self.datacenter.add_daily_capacity_req(curr_day, task.capacity * task.duration)
        
    def get_new_tasks(self) -> list[Task]:
        """
        TODO: document
        """
        curr_t = self.datacenter.t
        start = (curr_t + START_DELAY_H)*MICROSEC_PER_HOUR
        end = (curr_t + START_DELAY_H +1)*MICROSEC_PER_HOUR

        tasks = []
        new_task_data = self.task_data[(start <= self.task_data['time']) & (self.task_data['time'] < end)]
        for _, row in new_task_data.iterrows():
            task_duration = (row['duration'] // self.time_window) + 1  # calculate number of timesteps tasks lasts
            new_task = Task(row['task_id'], task_duration, row['cpu'])
            tasks.append(new_task)
            self.update_daily_capacity_req(curr_t, new_task)

        return tasks

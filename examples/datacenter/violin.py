import ray
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
import sys
sys.path.append(".")
from sustaingym.envs.datacenter import DatacenterGym
import pandas as pd


NUM_TRIALS = 1
NUM_EPISODES = 100


def evaluate(algo, model_name: str, train_carbon_yr: int, test_carbon_yr: int):
    daily_reward = 0
    terminated = truncated = False

    env = DatacenterGym()  # TODO: add test_carbon_yr
    obs, info = env.reset()
    hour = 0

    table = pd.DataFrame(columns=["model", "run", "train_carbon_yr",
                                  "test_carbon_yr", "day", "reward"])

    for trial_num in range(1, NUM_TRIALS+1):
        while not terminated and not truncated:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            hour += 1
            if hour % 24 == 0:
                day = hour // 24
                avg_daily_reward = daily_reward / 24
                table.loc[len(table)] = [model_name, trial_num, train_carbon_yr, test_carbon_yr, day, avg_daily_reward]

            daily_reward += reward

    return table


if __name__ == "__main__":
    ray.init()
    algo = ppo.PPO(env=DatacenterGym,
                   config={"env_config": {},
                           "framework": "torch",
                           "disable_env_checking": True})
    
    untrained_table = evaluate(algo, "untrained_ppo", 2019, 2019)
    untrained_table.to_csv("logs/datacenter/untrained_ppo.csv", index=False)

    while True:
        d = algo.train()
        episodes_total = d["episodes_total"]
        print(f"Episode #{episodes_total} reward: {d['episode_reward_mean']}")
        if episodes_total > NUM_EPISODES:
            break

    trained_table = evaluate(algo, "ppo", 2019, 2019)
    trained_table.to_csv("logs/datacenter/ppo.csv", index=False)

import ray
from ray.rllib.algorithms import ppo, sac, ddpg
import sys
sys.path.append(".")
from sustaingym.envs.datacenter import DatacenterGym
import pandas as pd
import argparse
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--num_trials", type=int, default=5)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--algo", type=str, choices=["PPO", "SAC", "DDPG"])
parser.add_argument("--train_carbon_yr", type=int, default=2019)
parser.add_argument("--test_carbon_yr", type=int, default=2019)
args = parser.parse_args()


def evaluate(algo, model_name: str, train_carbon_yr: int, test_carbon_yr: int):
    daily_reward = 0

    env_config = {"sim_start_time": datetime(test_carbon_yr, 5, 1),
                  "sim_end_time": datetime(test_carbon_yr, 6, 1)}
    env = DatacenterGym(env_config)
    hour = 0

    table = pd.DataFrame(columns=["model", "run", "train_carbon_yr",
                                  "test_carbon_yr", "day", "reward"])

    for trial_num in range(3, args.num_trials+1):
        print(f"Trial #{trial_num}")
        obs, _ = env.reset()
        terminated = truncated = False
        while not terminated and not truncated:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            hour += 1
            if hour % 24 == 0:
                day = hour // 24
                avg_daily_reward = daily_reward
                table.loc[len(table)] = [model_name, trial_num, train_carbon_yr, test_carbon_yr, day, avg_daily_reward]

            daily_reward += reward
        print("")
    return table


if __name__ == "__main__":
    ray.init()
    env_config = {"sim_start_time": datetime(args.train_carbon_yr, 5, 1),
                  "sim_end_time": datetime(args.train_carbon_yr, 6, 1)}
    if args.algo == "PPO":
        algo = ppo.PPO(env=DatacenterGym,
                    config={"env_config": env_config,
                            "framework": "torch",
                            "disable_env_checking": True})
    elif args.algo == "SAC":
        algo = sac.SAC(env=DatacenterGym,
                       config={"env_config": env_config,
                               "framework": "torch",
                               "disable_env_checking": True})
    elif args.algo == "DDPG":
        algo = ddpg.DDPG(env=DatacenterGym,
                         config={"env_config": env_config,
                                 "framework": "torch",
                                 "disable_env_checking": True})
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")

    with open(f"logs/datacenter/{args.algo}_train_log.txt", "w") as f:
        f.write("Episode, Reward\n")
        while True:
            d = algo.train()
            episodes_total = d["episodes_total"]
            f.write(f"{episodes_total},{d['episode_reward_mean']}\n")
            if episodes_total > args.num_episodes:
                break
        print("Training complete!\n\n")

    trained_table = evaluate(algo, args.algo, args.train_carbon_yr, args.test_carbon_yr)
    trained_table.to_csv(f"logs/datacenter/{args.algo}3to5.csv", index=False)

import ray
from ray.rllib.algorithms import ppo, sac, ddpg
import sys
sys.path.append(".")
from sustaingym.envs.datacenter import DatacenterGym
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_trials", type=int, default=5)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--algo", type=str, choices=["PPO", "SAC", "DDPG"])
args = parser.parse_args()


def evaluate(algo, model_name: str, train_carbon_yr: int, test_carbon_yr: int):
    daily_reward = 0
    terminated = truncated = False

    env = DatacenterGym()  # TODO: add test_carbon_yr
    obs, info = env.reset()
    hour = 0

    table = pd.DataFrame(columns=["model", "run", "train_carbon_yr",
                                  "test_carbon_yr", "day", "reward"])

    for trial_num in range(1, args.num_trials+1):
        print(f"Trial #{trial_num}")
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
    if args.algo == "PPO":
        algo = ppo.PPO(env=DatacenterGym,
                    config={"env_config": {},
                            "framework": "torch",
                            "disable_env_checking": True})
    elif args.algo == "SAC":
        algo = sac.SAC(env=DatacenterGym,
                       config={"env_config": {},
                               "framework": "torch",
                               "disable_env_checking": True})
    elif args.algo == "DDPG":
        algo = ddpg.DDPG(env=DatacenterGym,
                         config={"env_config": {},
                                 "framework": "torch",
                                 "disable_env_checking": True})
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")
    
    # untrained_table = evaluate(algo, "untrained_ppo", 2019, 2019)
    # untrained_table.to_csv("logs/datacenter/untrained_ppo.csv", index=False)

    while True:
        d = algo.train()
        episodes_total = d["episodes_total"]
        print(f"Episode #{episodes_total} reward: {d['episode_reward_mean']}")
        if episodes_total > args.num_episodes:
            break
    print("Training complete!\n\n")

    trained_table = evaluate(algo, args.algo, 2019, 2019)
    trained_table.to_csv(f"logs/datacenter/{args.algo}.csv", index=False)

import gymnasium as gym
import ray, sys
from ray.rllib.algorithms import ppo
sys.path.append(".")
from sustaingym.envs.datacenter import DatacenterGym

if __name__ == "__main__":
    ray.init()
    algo = ppo.PPO(env=DatacenterGym, config={ "env_config": {}, "framework": "torch", "disable_env_checking": True})

    while True:
        print(algo.train())

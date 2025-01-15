from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import torch
import numpy as np
import os

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, model_path="dqn_model.zip", vecnorm_path="vecnormalize.pkl"):
        self.path = model_path

        self.model = None
        self.env = VecNormalize(make_vec_env(make_env, n_envs=1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def train(self, total_timesteps=100000):
        
        policy_kwargs = {'net_arch': [256, 256, 64, 64]}
        
        self.model = DQN(
            "MlpPolicy",
            self.env,
            verbose=0,
            learning_rate=lr_scheduler,
            buffer_size=10000,
            batch_size=64,
            exploration_fraction=0.1,
            exploration_initial_eps=1,
            exploration_final_eps=0.05,
            tensorboard_log=f"./{self.path}/tensorboard/",
            device=self.device,
        )

        checkpoint_callback = SaveModelAndVecNormalizeCallback(
            agent = self, 
            save_freq=10000,
            save_path_model=f"./{self.path}/DQN_model_checkpoints",
            save_path_env =f"./{self.path}/DQN_VecNormalize_checkpoints",
            name_prefix="dqn"
        )
        
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        
        self.model.save(f"./{self.path}/model.zip")
        self.env.save(f"./{self.path}/vecnormalize.pkl")

    def act(self, observation, use_random=False):
        if self.model is None:
            self.load()
        
        # Normalize observation if VecNormalize is used
        if self.env and isinstance(self.env, VecNormalize):
            observation = self.env.normalize_obs(np.array(observation))
        
        action, _ = self.model.predict(observation, deterministic=not use_random)
        return action

    def save(self, path):
        if self.model:
            self.model.save(path)
            if isinstance(self.env, VecNormalize):
                self.env.save(self.vecnorm_path)

    def load(self):
        # Path
        self.model_path = "./Model.zip"
        self.vecnorm_path = "./VecNormalizeEnv.pkl"

        # Load VecNormalize first
        env = make_vec_env(make_env, n_envs=1)
        if os.path.exists(self.vecnorm_path):
            self.env = VecNormalize.load(self.vecnorm_path, env)
            self.env.training = False
            self.env.norm_reward = False  # Disable reward normalization for inference
        else:
            self.env = env
        
        # Load the model
        self.model = DQN.load(self.model_path)

class SaveModelAndVecNormalizeCallback(BaseCallback):
    def __init__(self, agent, save_freq, save_path_model, save_path_env, name_prefix="dqn", verbose=0):
        super(SaveModelAndVecNormalizeCallback, self).__init__(verbose)
        self.agent = agent
        self.save_freq = save_freq
        self.save_path_model = save_path_model
        self.save_path_env = save_path_env
        self.name_prefix = name_prefix
        os.makedirs(save_path_model, exist_ok=True)
        os.makedirs(save_path_env, exist_ok=True)
        self.max_score_agent = 0
        self.timestep_max_score_agent = 0 

        self.max_score_agent_dr = 0
        self.timestep_max_score_agent_dr = 0 

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.save_path_model, f"{self.name_prefix}_model_{self.num_timesteps}")
            vecnorm_path = os.path.join(self.save_path_env, f"{self.name_prefix}_vecnormalize_{self.num_timesteps}.pkl")
            
            # Save the model
            self.model.save(model_path)
            
            # Save VecNormalize statistics
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(vecnorm_path) 
            
            print(f"Saved model and VecNormalize at step {self.num_timesteps}")

        if self.num_timesteps % (self.save_freq) == 0:
            score_agent = evaluate_HIV(agent=self.agent, nb_episode=1)
            score_agent_dr = evaluate_HIV_population(agent=self.agent, nb_episode=15)

            if score_agent >= self.max_score_agent:
              self.max_score_agent = score_agent
              self.timestep_max_score_agent = self.num_timesteps

            if score_agent_dr >= self.max_score_agent_dr:
              self.max_score_agent_dr = score_agent_dr
              self.timestep_max_score_agent_dr = self.num_timesteps

            print(f"Best score_agent {self.max_score_agent} for timestep {self.timestep_max_score_agent}")
            print(f"Best score_agent_dr {self.max_score_agent_dr} for timestep {self.timestep_max_score_agent_dr}")

            print(f"score_agent {score_agent}")
            print(f"score_agent_dr {score_agent_dr}")
        return True


def make_env():
    # Create a new instance of the environment with the same configuration
    env = HIVPatient(domain_randomization=False)
    env = TimeLimit(env, max_episode_steps=200)
    return env


def lr_scheduler(progress_remaining):
    time_points = [0, 0.0025, 0.02]
    lr_points = [0.001, 0.0001, 0.0001]
    step = 1 - progress_remaining
    return np.interp(step, time_points, lr_points)

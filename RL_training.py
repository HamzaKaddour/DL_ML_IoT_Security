from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from gym import spaces
import numpy as np

# Define a custom environment for our dataset
class DatasetEnv(gym.Env):
    def __init__(self, df):
        super(DatasetEnv, self).__init__()
        
        # Assume all features are numeric for simplicity and scale to [0, 1]
        # In practice, you would handle categorical features and scaling appropriately
        self.dataset = df.select_dtypes(include=[np.number]).drop(columns=['label']).apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        self.labels = df['label']
        self.current_step = 0
        
        # Define action and observation space
        # Actions: 0 (normal), 1 (alternative)
        self.action_space = spaces.Discrete(2)
        # Observation: features from the dataset
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.dataset.shape[1],), dtype=np.float32)
    
    def reset(self):
        # Reset the environment to the beginning
        self.current_step = 0
        return self.dataset.iloc[self.current_step].values

    def step(self, action):
        # Compare the action with the actual label, reward if correct
        reward = 1 if action == self.labels.iloc[self.current_step] else -1
        
        self.current_step += 1
        done = self.current_step == len(self.labels)
        
        # Set placeholder for info
        info = {}
        
        # Provide the next state
        next_state = self.dataset.iloc[self.current_step % len(self.labels)].values if not done else np.zeros(self.dataset.shape[1])
        
        return next_state, reward, done, info

# Prepare the environment
file_path = 'Processed_datasets/Processed_Windows_dataset/windows10_dataset.csv'
df = pd.read_csv(file_path, sep=None, engine='python')
env = DatasetEnv(df)
check_env(env, warn=True)

# Make environment vectorized for stability
vec_env = make_vec_env(lambda: env, n_envs=1)

# Initialize the agent
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

mean_reward, std_reward

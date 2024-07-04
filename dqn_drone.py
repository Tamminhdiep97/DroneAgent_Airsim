import setup_path
import gymnasium as gym
import airgym
import time

from stable_baselines3 import DQN, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(128, 128, 1),
            )
        )
    ]
)
# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = DQN(
    # "CnnPolicy",
    'MultiInputPolicy',
    env,
    learning_rate=0.0001,
    verbose=2,
    batch_size=32,
    train_freq=1,
    target_update_interval=2,
    learning_starts=40,
    buffer_size=1000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=100,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=15000,
    tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
    log_interval=1,
    progress_bar=True,
    **kwargs
)

# Save policy weights
model.save("dqn_airsim_drone_policy")

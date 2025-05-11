import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
)  # AGGIUNGI VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
import time
from ale_py import ALEInterface
import ale_py

gym.register_envs(ale_py)
ale = ALEInterface()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def make_env(frame_skip=4, terminal_on_life_loss=True):
    env = gym.make("ALE/Pong-v5")
    env = AtariWrapper(
        env,
        frame_skip=frame_skip,
        terminal_on_life_loss=terminal_on_life_loss,
        # grayscale_obs=grayscale_obs,  # Importante per lo spazio di osservazione
        # scale_obs=False,  # Di solito False, CnnPolicy può gestire uint8
    )
    return env


def train_agent(
    env,
    existing_model=None,
    total_timesteps=2_500_000,
    save_path="checkpoints_/pong_a2c.zip",
    checkpoint_interval=100_000,
    use_gpu: bool = True,
):

    log_dir = "checkpoints_prova2/logs/"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_path = "checkpoints_prova2\pong_a2c_ckpt_ckpt_11300000.zip"

    if use_gpu and torch.cuda.is_available():
        current_device = torch.device("cuda")
        print("Training on GPU.")
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU not available, training on CPU.")
        else:
            print("Training on CPU.")
        current_device = torch.device("cpu")

    if existing_model:

        model = A2C.load(model_path, env=env, device=current_device)
        model.num_timesteps = int(model_path.split("_")[-1].split(".")[0])
        print(f"Resuming training from {model.num_timesteps} timesteps.")
        print(f"Modello caricato da: {model_path}")
    else:
        print("Training new model from scratch.")
        model = A2C(
            "CnnPolicy",
            env,
            learning_rate=5e-5,
            n_steps=15,
            gae_lambda=0.95,
            use_rms_prop=True,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            device=current_device,
            # n_stack=n_stack,
            # You might want to ensure other A2C parameters are consistent
            # e.g., learning_rate, n_steps, gamma, etc.
        )

    checkpoint_base_name = os.path.splitext(os.path.basename(save_path))[0]
    checkpoint_dir = os.path.dirname(save_path)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=checkpoint_dir,
        name_prefix=checkpoint_base_name
        + "_ckpt",  # Saves as pong_a2c_ckpt_XXXX_steps.zip
        save_replay_buffer=False,  # Typically False for A2C
        save_vecnormalize=False,  # If you use VecNormalize, set to True
    )

    print(f"Starting/Continuing training up to {total_timesteps} total timesteps.")
    print(
        f"Checkpoints will be saved every {checkpoint_interval} timesteps in '{checkpoint_dir}' with prefix '{checkpoint_base_name}_ckpt'."
    )

    model.learn(
        total_timesteps=total_timesteps,  # This is the target *cumulative* timesteps
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )

    model.save(save_path)
    print(f"Final model saved to {save_path} at {model.num_timesteps} timesteps.")
    return model


def collect_observations(
    env, model, encoder, decoder, num_frames=100_000, save_file="data/pong_frames.pt"
):
    obs_list = []
    latent_list = []
    obs = env.reset()
    for _ in range(num_frames):
        action, _ = model.predict(obs)
        obs_tensor = torch.tensor(obs[0])
        obs_list.append(obs_tensor)

        latent = encoder(obs_tensor.unsqueeze(0).float())
        latent_list.append(latent)

        decoded_obs = decoder(latent).squeeze(0)

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    frames = torch.stack(obs_list)
    latents = torch.cat(latent_list)
    torch.save({"frames": frames, "latents": latents}, save_file)
    print(f"Saved {len(frames)} frames and latents to {save_file}")

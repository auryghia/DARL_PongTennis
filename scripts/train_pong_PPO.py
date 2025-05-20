import os
import sys

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO  # MODIFICATO: da A2C a PPO
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
    total_timesteps=5_500_000,
    save_path="ppo_params2\checkpoints\\",
    checkpoint_interval=100_000,
    use_gpu: bool = True,
):

    log_dir = "ppo_params2\\results\\"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_path = "ppo_params2\_ckpt_400000.zip"

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

        model = PPO.load(
            model_path, env=env, device=current_device
        )  # MODIFICATO: A2C.load -> PPO.load
        model.num_timesteps = int(model_path.split("_")[-1].split(".")[0])
        print(f"Resuming training from {model.num_timesteps} timesteps.")
        print(f"Modello caricato da: {model_path}")
    else:
        print("Training new model from scratch.")
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[128, 128]))

        model = PPO(  # MODIFICATO: A2C -> PPO
            "CnnPolicy",
            env,
            learning_rate=5e-5,  # PPO spesso beneficia di LR più bassi o schedulati
            n_steps=2048,  # PPO tipicamente usa n_steps più grandi (es. 128, 512, 1024, 2048 per env)
            batch_size=128,  # Parametro specifico di PPO (dimensione minibatch)
            n_epochs=18,  # Parametro specifico di PPO (numero di epoche per aggiornamento)
            gae_lambda=0.98,
            gamma=0.99,
            clip_range=0.1,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            device=current_device,
            policy_kwargs=policy_kwargs,
            ent_coef=0.05,
            target_kl=0.015,
            vf_coef=0.5,
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
        n_test_episodes=1,
        deterministic_testing=True,
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

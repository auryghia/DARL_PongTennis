import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import time
from ale_py import ALEInterface
import ale_py

gym.register_envs(ale_py)
ale = ALEInterface()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def make_env():
    env = gym.make("ALE/Pong-v5")
    env = AtariWrapper(env)
    return env


def train_agent(
    env,
    total_timesteps=1_000_000,
    save_path="checkpoints/pong_a2c.zip",
    checkpoint_interval=100_000,
    use_gpu: bool = True,  # Aggiunto parametro per controllare l'uso della GPU
):
    log_dir = "checkpoints/a2c_pong_training_logs/"  # Scegli una directory
    # Determina il device in base al parametro use_gpu e alla disponibilità
    if use_gpu and torch.cuda.is_available():
        current_device = torch.device("cuda")
        print("Training on GPU.")
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU not available, training on CPU.")
        else:
            print("Training on CPU.")
        current_device = torch.device("cpu")

    model = A2C(
        "CnnPolicy", env, verbose=1, tensorboard_log=log_dir, device=current_device
    )  # Imposta verbose=0 per evitare output ridondanti

    # Inizializza la barra di progresso
    timesteps_per_update = checkpoint_interval
    num_updates = total_timesteps // timesteps_per_update
    progress_bar = tqdm(
        total=total_timesteps, desc="Training Progress", unit="timesteps"
    )

    start_time = time.time()

    for update in range(num_updates):
        model.learn(total_timesteps=timesteps_per_update, reset_num_timesteps=False)

        progress_bar.update(timesteps_per_update)

        checkpoint_path = (
            f"{save_path}_checkpoint_{(update + 1) * checkpoint_interval}.zip"
        )
        model.save(checkpoint_path)
        print(f"Checkpoint salvato in: {checkpoint_path}")

    progress_bar.close()  # Chiudi la barra di progresso
    end_time = time.time()  # Fine del timer

    model.save(save_path)
    print(f"Modello finale salvato in: {save_path}")
    print(f"Allenamento completato in {end_time - start_time:.2f} secondi.")
    return model


def collect_observations(
    env, model, encoder, decoder, num_frames=100_000, save_file="data/pong_frames.pt"
):
    obs_list = []
    latent_list = []
    obs = env.reset()
    for _ in range(num_frames):
        action, _ = model.predict(obs)
        obs_tensor = torch.tensor(obs[0])  # assuming 1 env
        obs_list.append(obs_tensor)

        # Encode the observation
        latent = encoder(obs_tensor.unsqueeze(0).float())
        latent_list.append(latent)

        # Decode the latent representation (optional, for validation)
        decoded_obs = decoder(latent).squeeze(0)

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    frames = torch.stack(obs_list)
    latents = torch.cat(latent_list)
    torch.save({"frames": frames, "latents": latents}, save_file)
    print(f"Saved {len(frames)} frames and latents to {save_file}")

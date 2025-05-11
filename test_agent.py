import os
import time
import numpy as np
import torch
from stable_baselines3 import A2C
import torch
import gymnasium as gym
from ale_py import ALEInterface
import ale_py
import sys
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
)  # AGGIUNGI QUESTI

import os
from scripts.train_pong_agent import make_env


def get_env(N_STACK_FRAMES=4):
    env = DummyVecEnv([make_env])
    stacked_env = VecFrameStack(env, n_stack=N_STACK_FRAMES)
    return stacked_env


def test_agent(model_path, env, num_episodes=100, render="human", use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        current_device = torch.device("cuda")
        print("Testing on GPU.")
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU not available, testing on CPU.")
        else:
            print("Testing on CPU.")
        current_device = torch.device("cpu")

    model_path = "checkpoints_prova2\pong_a2c_ckpt_ckpt_16500000_steps.zip"

    try:
        model = A2C.load(model_path, env=env, device=current_device)
        print(f"Modello caricato da: {model_path}")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        return

    total_rewards = []
    total_steps = []

    for episode in range(num_episodes):
        obs = env.reset()
        done_flag = False
        episode_reward = 0
        episode_steps = 0
        max_steps = 10000
        while not done_flag and episode_steps < max_steps:
            action, _states = model.predict(obs, deterministic=True)

            new_obs, rewards, terminated_flags, infos = env.step(action)

            current_obs = new_obs
            truncated_flags = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

            current_obs = new_obs
            current_reward = rewards[0]
            is_terminated = terminated_flags[0]
            is_truncated = truncated_flags[0]

            done_flag = is_terminated or is_truncated
            obs = current_obs
            episode_reward += current_reward
            episode_steps += 1

            if render:
                try:
                    env.render()
                    time.sleep(0.01)
                except Exception as e:
                    print(
                        f"Errore durante il rendering: {e}. Il rendering potrebbe non essere supportato o la finestra è stata chiusa."
                    )
                    render = False

        if episode_steps >= max_steps and not done_flag:
            print(
                f"Raggiunto il limite massimo di passi ({max_steps}) per l'episodio senza terminazione/troncamento."
            )

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        print(
            f"Episodio {episode + 1}: Ricompensa = {episode_reward}, Steps = {episode_steps}"
        )

    env.close()
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(total_steps)
    print(f"\n--- Risultati del Test ---")
    print(f"Numero di episodi: {num_episodes}")
    print(f"Ricompensa media: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"Durata media episodio (steps): {avg_steps:.2f}")
    print(f"--------------------------")
    env.close()
    return avg_reward, std_reward


if __name__ == "__main__":

    print("\nInizio test...")
    model_path = "checkpoints_prova2\pong_a2c_ckpt_ckpt_16500000_steps.zip"

    if not os.path.exists(model_path):
        print(
            f"ERRORE: Il file del modello {model_path} non esiste. Controlla il percorso."
        )

    else:
        stacked_env = get_env(N_STACK_FRAMES=4)
        test_agent(model_path, stacked_env, num_episodes=5, render=True, use_gpu=True)

from scripts.train_pong_agent import make_env, train_agent, collect_observations
from models.encoder import Encoder
from models.decoder import Decoder
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from ale_py import ALEInterface
import ale_py

gym.register_envs(ale_py)

ale = ALEInterface()
env = gym.make("ALE/Pong-v5")

print("Environment loaded successfully!")

N_STACK_FRAMES = 4


def main():
    latent_dim = 128
    total_timesteps = 10_000_000
    num_frames = 100_000
    model_save_path = "checkpoints_prova2\pong_a2c_ckpt.zip"
    data_save_file = "data/pong_frames.pt"

    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    env = DummyVecEnv([make_env])
    stacked_env = VecFrameStack(env, n_stack=N_STACK_FRAMES)

    print("Inizio dell'addestramento dell'agente...")

    model = train_agent(
        stacked_env,
        existing_model=True,
        total_timesteps=total_timesteps,
        save_path=model_save_path,
        use_gpu=True,
    )
    print(f"Modello salvato in: {model_save_path}")

    print("Raccolta delle osservazioni...")
    collect_observations(
        env, model, encoder, decoder, num_frames=num_frames, save_file=data_save_file
    )
    print(f"Dati salvati in: {data_save_file}")


if __name__ == "__main__":
    main()

#
from scripts.train_pong_agent import make_env, train_agent, collect_observations
from models.encoder import Encoder
from models.decoder import Decoder
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from ale_py import ALEInterface
import ale_py

gym.register_envs(ale_py)

ale = ALEInterface()
env = gym.make("ALE/Pong-v5")

print("Environment loaded successfully!")


def main():
    # Definisci i parametri
    latent_dim = 128
    total_timesteps = 1_000_000
    num_frames = 100_000
    model_save_path = "checkpoints/pong_a2c.zip"
    data_save_file = "data/pong_frames.pt"

    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)

    # Sposta i modelli su GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    # Crea l'ambiente
    env = DummyVecEnv([make_env])

    # Allena l'agente
    print("Inizio dell'addestramento dell'agente...")

    model = train_agent(
        env, total_timesteps=total_timesteps, save_path=model_save_path, use_gpu=False
    )
    print(f"Modello salvato in: {model_save_path}")

    print("Raccolta delle osservazioni...")
    collect_observations(
        env, model, encoder, decoder, num_frames=num_frames, save_file=data_save_file
    )
    print(f"Dati salvati in: {data_save_file}")


# if __name__ == "__main__":
#     main()

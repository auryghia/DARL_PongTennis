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

import os

gym.register_envs(ale_py)

ale = ALEInterface()
env = gym.make("ALE/Pong-v5")


def test_agent(model_path, env, num_episodes=10, render=True, use_gpu=True):
    """
    Testa un agente addestrato.

    :param model_path: Percorso del modello salvato (.zip).
    :param env: L'ambiente di test (dovrebbe essere lo stesso tipo di ambiente usato per l'addestramento).
    :param num_episodes: Numero di episodi per cui testare l'agente.
    :param render: Se True, renderizza l'ambiente durante il test.
    :param use_gpu: Se True e la GPU è disponibile, carica il modello sulla GPU.
    """
    # Determina il device
    if use_gpu and torch.cuda.is_available():
        current_device = torch.device("cuda")
        print("Testing on GPU.")
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU not available, testing on CPU.")
        else:
            print("Testing on CPU.")
        current_device = torch.device("cpu")

    # Carica il modello addestrato
    # Assicurati che l'ambiente passato a A2C.load() sia un'istanza dell'ambiente
    # o None se l'ambiente è già wrappato e non vuoi che SB3 lo wrappi di nuovo.
    # Per coerenza, è meglio passare l'ambiente.

    wrap_env = AtariWrapper(env)  # Aggiungi altri parametri se usati in addestramento

    try:
        model = A2C.load(model_path, env=wrap_env, device=current_device)
        print(f"Modello caricato da: {model_path}")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        return

    total_rewards = []
    total_steps = []

    for episode in range(num_episodes):
        obs, _ = wrap_env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        max_steps = 1000  # Limita il numero massimo di passi per episodio
        while not done and episode_steps < max_steps:
            # Usa deterministic=True per ottenere l'azione più probabile (comportamento di exploitation)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = wrap_env.step(action)
            episode_reward += reward
            episode_steps += 1
            time.sleep(0.01)  # Rallenta un po' per la visualizzazione
            if render:
                wrap_env.render()  # Usa il metodo render dell'ambiente wrappato

        if episode_steps >= max_steps:
            print(f"Raggiunto il limite massimo di passi ({max_steps}) per l'episodio.")

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        print(
            f"Episodio {episode + 1}: Ricompensa = {episode_reward}, Steps = {episode_steps}"
        )

    env.close()  # Chiudi l'ambiente dopo il test
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(total_steps)
    print(f"\n--- Risultati del Test ---")
    print(f"Numero di episodi: {num_episodes}")
    print(f"Ricompensa media: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"Durata media episodio (steps): {avg_steps:.2f}")
    print(f"--------------------------")
    return avg_reward, std_reward


# ... (resto del codice, inclusa la funzione collect_observations) ...

if __name__ == "__main__":
    # Esempio di come potresti eseguire l'addestramento e poi il test

    # 1. Crea l'ambiente
    # env_train = make_env()
    # env_train = DummyVecEnv([lambda: env_train]) # SB3 si aspetta un VecEnv

    # 2. Addestra l'agente (o salta se hai già un modello)
    # print("Inizio addestramento...")
    # trained_model_instance = train_agent(env_train, total_timesteps=200000, checkpoint_interval=50000, use_gpu=True)
    # path_to_trained_model = "checkpoints/pong_a2c.zip_checkpoint_200000.zip" # Assicurati che il percorso sia corretto
    # env_train.close()
    # print("Addestramento completato.")

    # 3. Testa l'agente
    print("\nInizio test...")
    model_path = "checkpoints_\pong_a2c_ckpt_1500000.zip"
    # Dovrebbe essere il percorso esatto del tuo file .zip
    # ad esempio, se save_path era "checkpoints/pong_a2c"
    # e checkpoint_interval era 100000,
    # il file per 200000 timesteps potrebbe essere
    # "checkpoints/pong_a2c_checkpoint_200000.zip"
    # o "checkpoints/pong_a2c.zip_checkpoint_200000.zip"
    # a seconda di come hai costruito il nome.
    # Controlla la tua cartella checkpoints!

    if not os.path.exists(model_path):
        print(
            f"ERRORE: Il file del modello {model_path} non esiste. Controlla il percorso."
        )

    else:
        # Non è necessario wrappare in DummyVecEnv per A2C.load se passi l'ambiente singolo
        # Tuttavia, se A2C.load si aspetta un VecEnv, allora wrappalo:
        # env_test = DummyVecEnv([lambda: env_test])

        ale = ALEInterface()
        env = gym.make("ALE/Pong-v5", render_mode="human" if True else None)
        print(
            f"Spazio di osservazione dell'ambiente di test wrappato: {env.observation_space}"
        )
        test_agent(model_path, env, num_episodes=5, render=True, use_gpu=True)
        test_agent(model_path, env, num_episodes=20, render=True, use_gpu=True)
        # env_test.close() # test_agent ora chiude l'ambiente

        env.close()  # Chiudi l'ambiente dopo il test

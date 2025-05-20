import os
import numpy as np
import cv2
import gymnasium as gym
from tqdm import tqdm
import argparse
from stable_baselines3 import PPO


def preprocess_frame(frame, size=(84, 84), rotate=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    if rotate:
        resized = cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return resized.astype(np.uint8)  # shape: (84, 84)


def extract_frames(
    env_name, output_file, num_frames=100000, rotate=False, model_path=None
):
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []

    if model_path and os.path.isfile(model_path):
        print(f"Using trained model from {model_path}")
        model = PPO.load(model_path)
        use_model = True
    else:
        print("No model provided or file not found. Using random actions.")
        model = None
        use_model = False

    print(f"\nExtracting {num_frames} frames from {env_name} (rotate={rotate})")
    pbar = tqdm(total=num_frames)
    while len(frames) < num_frames:
        frame = env.render()
        processed = preprocess_frame(frame, rotate=rotate)
        frames.append(processed)

        if use_model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        pbar.update(1)
    pbar.close()

    frames = np.stack(frames)  # shape: (N, 84, 84)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, frames)
    print(f"Saved {len(frames)} frames to {output_file}")
    env.close()


# python extract_frames_conditional.py \
#   --env ALE/Pong-v5 \
#   --out data/pong_preprocessed.npy \
#   --model path/to/pong_model.zip


# python extract_frames_conditional.py \
#   --env ALE/Tennis-v5 \
#   --out data/tennis_preprocessed.npy \
#   --rotate

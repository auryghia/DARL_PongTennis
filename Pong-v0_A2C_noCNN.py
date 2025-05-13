import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import gymnasium as gym
from ale_py import ALEInterface
import ale_py
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorModel, self).__init__()
        c, h, w = input_shape
        input_dim = c * h * w
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ELU(),
            # opzionale: aggiungi 256, 64 se vuoi che sia più profonda
            # nn.Linear(512, 256),
            # nn.ELU(),
            # nn.Linear(256, 64),
            # nn.ELU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.net(x)
        return F.softmax(x, dim=-1)


class CriticModel(nn.Module):
    def __init__(self, input_shape):
        super(CriticModel, self).__init__()
        c, h, w = input_shape
        input_dim = c * h * w
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ELU(),
            # opzionale: aggiungi 256, 64 se vuoi che sia più profonda
            # nn.Linear(512, 256),
            # nn.ELU(),
            # nn.Linear(256, 64),
            # nn.ELU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x)


class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.lr = 2.5e-5
        self.gamma = 0.99
        self.ROWS, self.COLS, self.REM_STEP = 80, 80, 4
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.image_memory = np.zeros(self.state_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorModel(self.state_size, self.action_size).to(self.device)
        self.critic = CriticModel(self.state_size).to(self.device)

        self.actor_optimizer = optim.RMSprop(
            self.actor.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
            alpha=0.99,
            eps=1e-08,
        )
        self.critic_optimizer = optim.RMSprop(
            self.critic.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
            alpha=0.99,
            eps=1e-08,
        )

        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=100, gamma=0.9
        )
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=100, gamma=0.9
        )

        self.Save_Path = "Results_A2C"
        os.makedirs(self.Save_Path, exist_ok=True)
        self.Model_name = os.path.join(self.Save_Path, "Pong_A2C")

    def PreprocessFrame(self, frame):
        frame = frame[35:195:2, ::2, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.COLS, self.ROWS))
        return gray.astype(np.float32) / 255.0

    def GetImage(self, frame):
        processed = self.PreprocessFrame(frame)
        self.image_memory = np.roll(self.image_memory, 1, axis=0)
        self.image_memory[0, :, :] = processed
        return np.expand_dims(self.image_memory, axis=0)

    def discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted[t] = running_add
        discounted -= np.mean(discounted)  # normalizing the result
        discounted /= np.std(discounted)  # divide by standard deviation
        return discounted

    def act(self, state):
        state_t = torch.from_numpy(state).float().to(self.device)
        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def train_step(self, states, actions, returns):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        values = self.critic(states).squeeze()
        advantages = returns - values.detach()

        # print(f"\nValue Mean: {values.mean().item()}, Std: {values.std().item()}")
        # print(f"Returns Mean: {returns.mean().item()}, Std: {returns.std().item()}")

        advantages = returns - values.detach()
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = (returns - values).pow(2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        print(
            f"Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}",
            flush=True,
        )

        # # Debug gradients
        # for name, param in self.actor.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad mean: {param.grad.abs().mean().item():.5e}")

    def run(self):
        batch_states, batch_actions, batch_returns = [], [], []
        for episode in range(self.EPISODES):
            frame, _ = self.env.reset()
            for i in range(self.REM_STEP):
                self.image_memory[i, :, :] = self.PreprocessFrame(frame)

            done = False
            state_list, action_list, reward_list = [], [], []
            score = 0

            while not done:
                state = np.expand_dims(self.image_memory, axis=0)
                action = self.act(state)

                next_frame, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.GetImage(next_frame)
                reward = np.sign(reward)
                state_list.append(state.squeeze())
                action_list.append(action)
                reward_list.append(reward)

                done = terminated or truncated
                score += reward

            returns = self.discount_rewards(reward_list)
            batch_states.extend(state_list)
            batch_actions.extend(action_list)
            batch_returns.extend(returns)

            print(f"Episode: {episode}, Score: {score}", flush=True)

            # Aggiorna ogni 10 episodi
            if (episode + 1) % 1 == 0:
                self.train_step(batch_states, batch_actions, batch_returns)
                batch_states, batch_actions, batch_returns = [], [], []

                torch.save(self.actor.state_dict(), self.Model_name + "_Actor.h5")
                torch.save(self.critic.state_dict(), self.Model_name + "_Critic.h5")


if __name__ == "__main__":
    # env_name = 'PongDeterministic-v4'
    gym.register_envs(ale_py)

    ale = ALEInterface()
    env = gym.make("ALE/Pong-v5")

    agent = A2CAgent(env)
    agent.run()
    # agent.test('Pong-v0_A2C_2.5e-05_Actor.h5', '')
    # agent.test('PongDeterministic-v4_A2C_1e-05_Actor.h5', '')

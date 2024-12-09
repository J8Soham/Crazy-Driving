import gymnasium
import highway_env
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate

env = gymnasium.make("racetrack-v0", render_mode='rgb_array')

env.unwrapped.config["vehicles_count"] = 0
observation = {
        "type": "Kinematics",
        "vehicles_count": 0,
        "features": ["cos_h"],
        "order": "sorted"
    }
env.unwrapped.config["observation"] = observation
ACTION_SIZE = 3
OBS_SIZE = 3

env.unwrapped.config["other_vehicles"] = 0

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta, spike_grad):
        super().__init__()

        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lin3 = nn.Linear(hidden_size, output_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.reset()

    def reset(self):
        self.mem_1 = self.lif1.init_leaky()
        self.mem_2 = self.lif2.init_leaky()
        self.mem_3 = self.lif3.init_leaky()

    # single step
    def forward(self, x):
        out = self.lin1(x)
        spk_1, self.mem_1 = self.lif1(out, self.mem_1)
        out = self.lin2(spk_1)
        spk_2, self.mem_2 = self.lif2(out, self.mem_2)
        out = self.lin3(spk_2)
        spk_3, self.mem_3 = self.lif2(out, self.mem_3)
        return spk_3

    
def get_racetrack_position(env):
    lane_index = env.unwrapped.road.network.get_closest_lane_index(env.unwrapped.vehicle.position)
    lane = env.unwrapped.road.network.graph[lane_index[0]][lane_index[1]][0]
    position = lane.local_coordinates(env.unwrapped.vehicle.position)[1]
    # -2.6 to 7.6
    position = 2 * (position - (7.6 - 2.6) / 2) / (7.6 + 2.6)
    return position

EPISODES = 100_000
SEEDS_PER_EPISODE = 1

ACTION_SIZE = 3
main_net = SNN(OBS_SIZE, 128, 128, 0.9, surrogate.fast_sigmoid(slope=25))

# loss and optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(main_net.parameters(), lr=learning_rate)

step = 0
for episode in range(EPISODES):
    env.unwrapped.config["duration"] = 100
    print(f'Episode: {episode}')

    for seed_num in range(SEEDS_PER_EPISODE):
        seed = np.random.randint(1_000_000)
        env.reset(seed=seed)
        done = truncated = False
        obs = None
        prev_position = get_racetrack_position(env)
        loss_components = []
        while not (done or truncated):
            action = 0
            if obs is not None:
                obs_tensor = torch.from_numpy(obs)
                action_spikes = main_net(obs_tensor)
                print(action_spikes)

                left_sum = 0
                right_sum = 0
                for i in range(64):
                    left_sum += action_spikes[i]
                    right_sum += action_spikes[i + 64]

                action_tensor = (right_sum - left_sum) / 64
                action = action_tensor.item()

                position = get_racetrack_position(env)
                turn_right_amount = (right_sum - left_sum) / 64
                loss = (position + action_tensor) ** 4

                loss_components.append(loss)
            
            action = np.array([[action]], dtype=np.float32)
            next_obs, reward, done, truncated, info = env.step(action)
            next_obs = next_obs.astype(np.float32)

            # get location from road edges
            position = get_racetrack_position(env)

            # get real next_obs
            cos_h = next_obs[0].item()
            next_obs = np.array([cos_h, position, position - prev_position], dtype=np.float32)
            
            if info["rewards"]["on_road_reward"] == 0:
                done = True

            env.render()
            prev_position = position
            obs = next_obs

        # learn
        optimizer.zero_grad()
        loss = sum(loss_components)
        loss.backward()
        optimizer.step()
        main_net.reset()
        
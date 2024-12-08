import gymnasium
import highway_env
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from time import sleep

env = gymnasium.make("racetrack-v0", render_mode='rgb_array')

# env.unwrapped.config["lane_centering_reward"] = 10
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


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.norm_1 = nn.BatchNorm1d(hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.norm_2 = nn.BatchNorm1d(hidden_size)
        self.layer_3 = nn.Linear(hidden_size, output_size)
        self.norm_3 = nn.BatchNorm1d(output_size)

    def forward(self, obs, batch_size=1):
        
        batch_norm_on = batch_size != 1
        if obs is None:
            retval = torch.zeros(self.output_size)
            return retval
        if not isinstance(obs, torch.Tensor):
            x = torch.tensor(obs)
        else:
            x = obs

        x = x.view(-1, self.input_size)

        x = self.layer_1(x)
        if batch_norm_on:
            x = self.norm_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        if batch_norm_on:
            x = self.norm_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        if batch_norm_on:
            x = self.norm_3(x)
        # x = F.softmax(x, dim=1) # for probabilities
        # x = F.tanh(x)
        # for DQN, don't use a final activation function
        
        return x
    

class ReplayMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity

        self.p_states = []
        self.p_q_values = []
        self.p_action_indices = []

        self.states = []
        self.q_values = []
        self.action_indices = []

    def p_push(self, state, q_value, action_index):
        self.p_states.append(state)
        self.p_q_values.append(q_value)
        self.p_action_indices.append(action_index)
        if len(self.p_states) > 10:
            state = self.p_states.pop(0)
            q_value = self.p_q_values.pop(0)
            action_index = self.p_action_indices.pop(0)
            self.push(state, q_value, action_index)

    def p_flush(self):
        while len(self.p_states) > 0:
            state = self.p_states.pop(0)
            q_value = self.p_q_values.pop(0)
            action_index = self.p_action_indices.pop(0)
            self.push(state, q_value, action_index)

    def p_judge(self, punishment=-2, decay=0.8):
        while len(self.p_states) > 0:
            state = self.p_states.pop(len(self.p_states) - 1)
            q_value = self.p_q_values.pop(len(self.p_states) - 1)
            action_index = self.p_action_indices.pop(len(self.p_states) - 1)
            q_value += punishment
            punishment *= decay
            self.push(state, q_value, action_index)

    def push(self, state, q_value, action_index):
        self.states.append(state)
        self.q_values.append(q_value)
        self.action_indices.append(action_index)
        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.q_values.pop(0)
            self.action_indices.pop(0)

    def sample(self, batch_size):
        indices = random.sample(list(range(len(self.states))), batch_size)
        batch_states = []
        batch_q_values = []
        batch_action_indices = []
        for i in indices:
            batch_states.append(self.states[i])
            batch_q_values.append(self.q_values[i])
            batch_action_indices.append(self.action_indices[i])

        batch_states = torch.stack(batch_states)
        batch_q_values = torch.stack(batch_q_values)
        batch_action_indices = torch.stack(batch_action_indices)

        return batch_states, batch_q_values, batch_action_indices

    def __len__(self):
        return len(self.states)
    
def get_racetrack_position(env):
    lane_index = env.unwrapped.road.network.get_closest_lane_index(env.unwrapped.vehicle.position)
    lane = env.unwrapped.road.network.graph[lane_index[0]][lane_index[1]][0]
    position = lane.local_coordinates(env.unwrapped.vehicle.position)[1]
    position = (position - 2.5) / 2.5
    return position

EPISODES = 100_000
SEEDS_PER_EPISODE = 1
BATCH_SIZE = 100
GAMMA = 0.8
UPDATE_FREQUENCY = 100
EPSILON_START = 0.2
EPSILON_MIN = 0.01
DECAY_RATE = 0.001

# model
main_net = Model(OBS_SIZE, ACTION_SIZE)
target_net = Model(OBS_SIZE, ACTION_SIZE)

# loss and optimizer
learning_rate = 0.001
criterion = nn.SmoothL1Loss()# nn.MSELoss()
optimizer = torch.optim.Adam(main_net.parameters(), lr=learning_rate)

memory = ReplayMemory()
step = 0
epsilon = EPSILON_START
for episode in range(EPISODES):
    env.unwrapped.config["duration"] = 1000
    print(f'Episode: {episode}')

    for seed_num in range(SEEDS_PER_EPISODE):
        seed = np.random.randint(1_000_000)
        env.reset(seed=seed)
        done = truncated = False
        obs = None
        prev_position = get_racetrack_position(env)
        action_index = 1
        while not (done or truncated):
            epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * math.exp(-DECAY_RATE * episode)
            if random.random() < epsilon:
                action_index = random.randint(0, 3)
            if action_index == 0:
                action = -0.5
            elif action_index == 1:
                action = 0
            else: # 2
                action = 0.5
            
            # step according to action (determined in previous iteration, or before loop)
            action = np.array([[action]], dtype=np.float32)

            # position = get_racetrack_position(env)
            # action_index = 1
            # if position < -0.2:
            #     action_index = 2
            # elif position > 0.2:
            #     action_index = 0
            # action = 0
            # if action_index == 0:
            #     action = -0.5
            # elif action_index == 1:
            #     action = 0
            # else: # 2
            #     action = 0.5
            # action = np.array([[action]], dtype=np.float32)

            next_obs, reward, done, truncated, info = env.step(action)
            next_obs = next_obs.astype(np.float32)

            # get location from road edges
            position = get_racetrack_position(env)

            # get real next_obs
            cos_h = next_obs[0].item()
            next_obs = np.array([cos_h, position, position - prev_position], dtype=np.float32)
            
            if info["rewards"]["on_road_reward"] == 0:
                done = True
                # reward -= 5
            reward /= 10

            if abs(position) - abs(prev_position) > 0:
                reward = -1
            else:
                reward = 1
                if abs(prev_position) > 1:
                    reward = 10



            if obs is not None:
                # run target_net on next_obs
                with torch.no_grad():
                    action_values = target_net(next_obs)
                action_index = torch.argmax(action_values, dim=1).item()

                # push to memory replay
                # using obs, not next_obs because formula is reward + (gamma * torch.max(target_network(next_state))
                # obs is already not None because prev_position is not None
                memory.p_push(torch.from_numpy(obs), reward + GAMMA * torch.max(action_values, dim=1).values, torch.tensor([action_index]))
            
            if done:
                memory.p_judge()
            else:
                memory.p_flush()
            

            env.render()
            prev_position = position
            obs = next_obs
            
            # learn
            if len(memory) >= BATCH_SIZE:
                # train main_net
                states, q_values, action_indices = memory.sample(BATCH_SIZE)
                predicted_action_values = main_net(states)
                # predicted_action_values: (BATCH_SIZE, 3)
                # action_indices: (BATCH_SIZE, 1)
                # -> predicted_q_values: (BATCH_SIZE, 1)
                predicted_q_values = torch.gather(predicted_action_values, 1, action_indices)
                loss = criterion(predicted_q_values, q_values)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(main_net.parameters(), 100)
                optimizer.step()

                # update target net
                step += 1
                if step % UPDATE_FREQUENCY == 0:
                    print('Updated')
                    step = 0
                    target_net.load_state_dict(main_net.state_dict())

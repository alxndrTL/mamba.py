import torch
import torch.nn.functional as F

import os
import time

"""
V1 : fully observable, tiny room with a goal generated randomly.
going over the goal gets the agent a reward and spawn a new goal.
the episodes ends after T steps.

it is convenient because:
- all envs end at the same time (both convenient for the env engine, AND for the training of the transformer : no padding needed)
- no obstacles to handle (convenient in the step() func)
- continuing, so its easier for the replay buffer (see buffer.py)
"""
class TinyHomeEngineV1:
    def __init__(self, B, h=10, w=10, max_envs_disp=4):
        self.B = B
        self.h = h
        self.w = w
        self.max_envs_disp = max_envs_disp
    
    def reset(self):
        self.grid = torch.zeros(self.B, self.h, self.w, dtype=torch.int)
        self.grid[:,  0,  :] = 1
        self.grid[:, -1,  :] = 1
        self.grid[:,  :,  0] = 1
        self.grid[:,  :, -1] = 1

        self.pos_player = torch.randint(low=1, high=self.h-1, size=(self.B, 2))
        self.pos_goal = torch.randint(low=1, high=self.h-1, size=(self.B, 2))

        while True:
            overlap = torch.all(self.pos_player == self.pos_goal, dim=1)
            if not overlap.any():
                break
            self.pos_goal[overlap] = torch.randint(low=1, high=self.h-1, size=(overlap.sum(), 2))

        disp_grid = self.grid.clone()
        disp_grid[torch.arange(self.B), self.pos_player[:, 0], self.pos_player[:, 1]] = 2
        disp_grid[torch.arange(self.B), self.pos_goal[:, 0], self.pos_goal[:, 1]] = 3

        """
        x = F.one_hot(self.pos_player[:, 0]-1, num_classes=3)
        y = F.one_hot(self.pos_player[:, 1]-1, num_classes=3)
        u = F.one_hot(self.pos_goal[:, 0]-1, num_classes=3)
        v = F.one_hot(self.pos_goal[:, 1]-1, num_classes=3)

        concatenated = torch.cat([x, y, u, v], dim=1) # (B, 12)
        """

        return disp_grid
    
    def optimal_policy_vectorized(self, moves):
        B, _ = self.pos_player.shape

        # Expand pos_player to (B, 5, 2) to match the moves
        expanded_pos_player = self.pos_player.unsqueeze(1).expand(-1, moves.size(0), -1)

        # Compute new positions for each move
        new_positions = expanded_pos_player + moves
        new_positions = new_positions.clamp(min=1, max=self.h-2)

        # Calculate Manhattan distances for each new position
        distances = torch.sum(torch.abs(new_positions - self.pos_goal.unsqueeze(1)), dim=2)

        # Find the move with the minimum distance for each environment
        actions = torch.argmin(distances, dim=1)

        return actions

    def step(self, a):
        # a : (B,)

        moves = torch.tensor([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]]) # X, N, E, S, W

        #a = self.optimal_policy_vectorized(moves)

        self.pos_player += moves[a]
        self.pos_player = self.pos_player.clamp(min=1, max=self.h-2) # pas du tout généralisable à des murs placés au milieu etc etc

        reached_goal = torch.all(self.pos_player == self.pos_goal, dim=1)
        reward = torch.where(reached_goal, 1., 0.).unsqueeze(1)

        # regen goal (only for "completed" env)
        num_reached = reached_goal.sum()
        if num_reached > 0:
            self.pos_goal[reached_goal] = torch.randint(low=1, high=self.h-1, size=(num_reached, 2))
            
            # make sure that the regenerated goals are at a different place
            while True:
                overlap = torch.all(self.pos_player == self.pos_goal, dim=1)
                if not overlap.any():
                    break
                self.pos_goal[overlap] = torch.randint(low=1, high=self.h-1, size=(overlap.sum(), 2))

        disp_grid = self.grid.clone()
        disp_grid[torch.arange(self.B), self.pos_player[:, 0], self.pos_player[:, 1]] = 2
        disp_grid[torch.arange(self.B), self.pos_goal[:, 0], self.pos_goal[:, 1]] = 3

        """
        x = F.one_hot(self.pos_player[:, 0]-1, num_classes=3)
        y = F.one_hot(self.pos_player[:, 1]-1, num_classes=3)
        u = F.one_hot(self.pos_goal[:, 0]-1, num_classes=3)
        v = F.one_hot(self.pos_goal[:, 1]-1, num_classes=3)

        concatenated = torch.cat([x, y, u, v], dim=1) # (B, 12)
        """

        return disp_grid, reward

    def display(self):
        os.system('cls' if os.name == 'nt' else 'clear')

        disp_grid = self.grid.clone()
        disp_grid[torch.arange(self.B), self.pos_player[:, 0], self.pos_player[:, 1]] = 2
        disp_grid[torch.arange(self.B), self.pos_goal[:, 0], self.pos_goal[:, 1]] = 3

        for b in range(min(self.B, self.max_envs_disp)):
            for row in disp_grid[b]:
                print(''.join(display_mapping.get(value.item(), '?') for value in row))
            
            print("\n")

display_mapping = {
    0: ' ',
    1: '#',
    2: '@',
    3: 'G'
}

def print_grid(grid):
    for b in range(grid.shape[0]):
        for row in grid[b]:
            print(''.join(display_mapping.get(value.item(), '?') for value in row))
            
        print("\n")

actions_to_char = ['X', 'N', 'E', 'S', 'W']
def print_act(act):
  print(actions_to_char[act])

if __name__ == "__main__":
    mode = "display" # "display" or "grind" or "collect"

    if mode == "display":
        nb_instances = 2
        steps = 10

        engine = TinyHomeEngineV1(nb_instances, 5, 5)
        engine.reset()

        engine.display()
        time.sleep(0.5)

        for _ in range(steps):
            obs, rew = engine.step(torch.randint(low=0, high=5, size=(nb_instances,)))
            print(obs)
            #engine.display()
            print(rew.shape)
            time.sleep(0.05)
    
    elif mode == "grind":
        nb_instances = 1000
        steps = 1000

        engine = TinyHomeEngineV1(nb_instances)
        engine.reset()

        start_time = time.perf_counter()

        for _ in range(steps):
            obs, rew = engine.step(torch.randint(low=0, high=5, size=(nb_instances,)))

        end_time = time.perf_counter()

        print(f"The collection of {nb_instances*steps} steps took {end_time-start_time} seconds")

    elif mode == "collect":
        nb_instances = 2
        steps = 1

        embed = torch.nn.Embedding(num_embeddings=6, embedding_dim=2)

        engine = TinyHomeEngineV1(nb_instances, 5, 5)
        engine.reset()

        for _ in range(steps):
            obs, rew = engine.step(torch.randint(low=0, high=5, size=(nb_instances,)))
            # obs: (B, h, w), rew: (B, 1)

            obs = obs.view(nb_instances, 25) # (B, h*w)
            
            e = embed(obs) # (B, h*w, embed_dim)
            print(e.shape)
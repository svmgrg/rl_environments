import torch

# There is one peculiar point about the dynamics used in this environment.
# The agent doesn't bounce off the walls in a way you would expect a 
# realistic elastic collision to occur. It instead undergoes some unusual
# behavior (look at the clamping function in the step() routine below.
# Although, this doesn't really matter for RL agents because it's just a
# different dynamics, it might be helpful for the user to know this.

class DotReacher():
    def __init__(self, target_state=torch.zeros(2), episode_cutoff_length=1000,
                 movement_noise_scaling=0):
        self.num_actions = 9
        self.dim_states = 2
        self.LB = torch.tensor([-1, -1], dtype=torch.float32)
        self.UB = torch.tensor([+1, +1], dtype=torch.float32)
        self.action_values = 0.03 * torch.tensor([[-1, +1], [+0, +1], [+1, +1],
                                                  [-1, +0], [+0, +0], [+1, +0],
                                                  [-1, -1], [+0, -1], [+1, -1]],
                                                 dtype=torch.float32)
        self.target_state = target_state
        self.episode_cutoff_length = episode_cutoff_length
        self.movement_noise_scaling = movement_noise_scaling
        
        self.state = None
        self.t = 0

    def reset(self):
        self.state = self.LB \
            + torch.rand((1, self.dim_states)) * (self.UB - self.LB)
        self.t = 0
        return self.state

    def step(self, action):
        noise = torch.rand(self.dim_states) * 0.06 - 0.03

        self.state = torch.clamp(self.state + self.action_values[action] \
                                 + self.movement_noise_scaling * noise,
                                 self.LB, self.UB)
        reward = - 0.01
        self.t += 1
        
        if torch.allclose(self.state, self.target_state, atol=0.1):
            done = 'terminal'
        elif self.t > self.episode_cutoff_length:
            done = 'cutoff'
        else:
            done = False
            
        return self.state, reward, done

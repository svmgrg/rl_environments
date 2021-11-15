import numpy as np
import pdb

class LinearChain():
    def __init__(self, P, r, start_state, terminal_states, reward_noise=0.3,
                 episode_cutoff_length=1000):
        self.P = P
        self.r = r
        self.reward_noise = reward_noise
        self.n = P.shape[-1]
        self.start_state = start_state
        self.terminal_states = terminal_states

        self.observation_space = self.n
        self.action_space = P.shape[0]
        self.state = None

        self.t = 0
        self.episode_cutoff_length = episode_cutoff_length

    def reset(self):
        self.state = self.start_state
        self.t = 0
        return self.state

    def step(self, action):
        if self.state is None:
            raise Exception('step() used before calling reset()')
        assert action in range(self.P.shape[0])

        reward = self.r[self.state, action] \
            + np.random.normal(loc=0, scale=self.reward_noise)
        self.state = np.random.choice(a=self.n, p=self.P[action, self.state])
        self.t = self.t + 1

        done = False
        if (self.state in self.terminal_states
            or self.t > self.episode_cutoff_length):
            done = True

        return self.state, reward, done, {}

    def calc_v_pi(self, pi, gamma):
        # calculate P_pi from the transition matrix P and the policy pi
        P_pi = np.zeros(self.P[0].shape)
        for a in range(pi.shape[1]):
            P_pi += self.P[a] * pi[:, a].reshape(-1, 1)

        # calculate the vector r_pi
        r_pi = (self.r * pi).sum(1).reshape(-1, 1)

        # calculate v_pi using the equation given above
        v_pi = np.matmul(
            np.linalg.inv(np.eye(self.P[0].shape[0]) - gamma * P_pi),
            r_pi)

        return v_pi

    def calc_q_pi(self, pi, gamma):
        # P_pi_control: SxA -> SxA
        P_pi_control = np.concatenate([pi[:, a] * np.concatenate(self.P)
                                       for a in range(self.action_space)], 1)
        sa_visitation = np.linalg.inv(np.eye(P_pi_control[0].shape[0]) \
                        - gamma * P_pi_control)
        r_sa = self.r.reshape(-1, 1, order='F')
        q_pi = np.matmul(sa_visitation, r_sa).reshape(
            -1, self.action_space, order='F')

        return q_pi

    def calc_d_gamma(self, pi, gamma):
        # calculate P_pi from the transition matrix P and the policy pi
        P_pi = np.zeros(self.P[0].shape)
        for a in range(pi.shape[1]):
            P_pi += self.P[a] * pi[:, a].reshape(-1, 1)

        # calculate d_gamma
        d_gamma = np.linalg.inv(np.eye(self.P[0].shape[0]) - gamma * P_pi)

        return d_gamma

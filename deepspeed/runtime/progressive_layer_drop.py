import numpy as np
from deepspeed.runtime.constants import PLD_THETA, PLD_GAMMA
from deepspeed.utils import log_dist


class ProgressiveLayerDrop(object):
    def __init__(self, pld_params):
        self.theta = pld_params[PLD_THETA]
        self.gamma = pld_params[PLD_GAMMA]
        self.current_theta = 1.0
        log_dist(f'Enabled progressive layer dropping (theta = {self.theta})', ranks=[0])

    def get_state(self):
        kwargs = {'progressive_layer_drop': True, 'pld_theta': self.get_theta()}
        return kwargs

    def get_theta(self):
        return self.current_theta

    def update_state(self, global_step):
        def _prob(x, gamma, p):
            return (1. - p) * np.exp(-gamma * x) + p

        self.current_theta = _prob(global_step, self.gamma, self.theta)

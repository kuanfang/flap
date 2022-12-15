import torch

from rlkit.torch import pytorch_util as ptu
from rlkit.planning.planner import Planner


class RandomPlanner(Planner):

    def __init__(
            self,
            model,

            # Hyperparameters.
            num_samples=1024, 

            debug=False,
            **kwargs):

        super().__init__(
            model=model,
            debug=debug,
            **kwargs)

        self.num_samples = num_samples

    def _plan(self, init_state, goal_state, num_steps, input_info=None):
        # Tile the plan.
        init_states = init_state[None, ...].repeat((self.num_samples, 1, 1, 1))
        goal_states = goal_state[None, ...].repeat((self.num_samples, 1, 1, 1))

        # Initialize z.
        sampled_zs = self._sample_prior(self.num_samples, num_steps)

        # Recursive prediction.
        plans = self._predict(sampled_zs, init_states, goal_states)

        # Compute the cost.
        costs = self._compute_costs(plans, init_states, goal_states)

        # Select.
        top_cost, top_ind = torch.min(costs, 0)
        z = sampled_zs[top_ind]
        plan = plans[top_ind]

        info = {
            'top_step': -1,
            'top_cost': top_cost,
            'top_z': z,
        }

        print('top_cost: %.2f, avg_cost: %.2f, max_cost: %.2f, min_cost: %.2f'
              % (
                  ptu.get_numpy(top_cost),
                  ptu.get_numpy(torch.mean(costs)),
                  ptu.get_numpy(torch.max(costs)),
                  ptu.get_numpy(torch.min(costs))
              ))

        # Debug.
        if self._debug:
            for key, value in info.items():
                print('%s: %r' % (key, ptu.get_numpy(value)))

        return plan, info

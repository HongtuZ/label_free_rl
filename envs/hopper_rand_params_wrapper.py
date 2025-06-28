import numpy as np
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv

RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']

class HopperRandParamsWrappedEnv(HopperEnv):
    def __init__(self, n_tasks=2, randomize_tasks=True):
        super(HopperRandParamsWrappedEnv, self).__init__()
        self.randomize_tasks = randomize_tasks
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
        self.env_step = 0
        self.rand_params = RAND_PARAMS
    
    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        np.random.seed(1234)
        param_sets = []
        for _ in range(n_tasks):
            new_params = {}
            # body mass -> one multiplier for all body parts
            if 'body_mass' in self.rand_params:
                body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_mass.shape)
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

            # body_inertia
            if 'body_inertia' in self.rand_params:
                body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_inertia.shape)
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.dof_damping.shape)
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

        	# friction at the body components
            if 'geom_friction' in self.rand_params:
                dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.geom_friction.shape)
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

            param_sets.append(new_params)
        return param_sets
    
    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset(self):
        self.env_step = 0
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.env_step += 1
        if self.env_step >= self._max_episode_steps:
            done = True
        return obs, reward, done, info

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame


class PointEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, task=0, render_mode=None, **kwargs):
        self.task = task
        self.goal = np.array((np.cos(task), np.sin(task)), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,))
        self.state = np.array((0.0, 0.0), dtype=np.float32)

        # For rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,):
        super().reset(seed=seed)
        task = options.get("task", None) if options is not None else None
        if task is not None:
            self.task = task
            self.goal = np.array((np.cos(task), np.sin(task)), dtype=np.float32)
        self.state = np.array((0.0, 0.0), dtype=np.float32)
        info = {"task": self.task}
        return self.state, info

    def step(self, action):
        self.state += action
        obs = self.state
        reward = -np.linalg.norm(self.state - self.goal)
        info = {"task": self.task}
        return obs, reward, False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        radius = 0.4 * self.window_size

        def xy2pix(xy) -> tuple:
            nonlocal radius
            x, y = xy
            x_pix = x * radius + self.window_size / 2.0
            y_pix = self.window_size / 2.0 - y * radius
            return x_pix, y_pix

        # Darw the circle
        pygame.draw.circle(canvas, (128, 128, 128), xy2pix((0, 0)), radius, 2)

        pygame.draw.circle(
            canvas,
            (128, 128, 128),
            xy2pix((0, 0)),
            self.window_size / 200,
        )

        # First we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            xy2pix(self.goal),
            self.window_size / 100,
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            xy2pix(self.state),
            self.window_size / 100,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

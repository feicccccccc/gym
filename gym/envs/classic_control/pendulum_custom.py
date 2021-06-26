"""
Fix Original Pendulum physics and rendering problem
- Fix integrator
- Fix rendering problem (incorrect representation of angle)
    - Angle at y axis, clockwise as positive
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

from scipy.integrate import solve_ivp


class PendulumCustomEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        # u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = - angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        # # Explicit Euler Integrator
        # thdotdot = (g / l * np.sin(th) + 1. / (2. * m * l ** 2.) * u)
        # newthdot = thdot + thdotdot * dt
        # newth = angle_normalize(th + thdot * dt)

        # Implicit Euler Integrator
        thdotdot = (g / l * np.sin(th) + 1. / (2. * m * l ** 2.) * u)
        newthdot = thdot + thdotdot * dt
        newth = th + newthdot * dt

        # # Second Order RK2 Integrator
        # thdotdot = (g / l * np.sin(th) + 1. / (2. * m * l ** 2.) * u)  # Assume u is uniform
        # k2 = (g / l * np.sin(th) + .5 * thdotdot * dt) + 1. / (2. * m * l ** 2.) * u
        # newthdot = thdot + k2 * dt
        # newth = th + newthdot * dt

        # RK4 Integrator with scipy integrate
        def inverted_pendulum(t, y):
            th = y[0]
            v = y[1]
            return v, g / l * np.sin(th) + 1. / (2. * m * l ** 2.) * u

        t = np.array([0., dt])
        sol = solve_ivp(fun=inverted_pendulum, t_span=[t[0], t[-1]], y0=self.state, t_eval=t)  # RK4 with stable energy

        thdotdot = inverted_pendulum(None, sol.y)[1][-1]
        newthdot = sol.y[1][-1]
        newth = sol.y[0][-1]

        self.state = np.array([newth, newthdot])
        return self._get_obs(), costs, False, {'theta': newth, 'theta_dot': newthdot, 'theta_dotdot': thdotdot}

    def reset(self, state=np.array([np.pi/2., 0])):
        # Modified for custom init

        self.state = state
        self.last_u = 0
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(-self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

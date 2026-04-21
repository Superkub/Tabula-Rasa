# Sisyphus environment — BipedalWalker with a boulder
import numpy as np
import pygame
import gymnasium as gym
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
from Box2D.b2 import polygonShape, fixtureDef

class SisyphusEnv(BipedalWalker):
    """Agent must push a boulder forward."""

    def __init__(self, render_mode=None):
        self.boulder = None
        self.prev_boulder_x = 0
        super().__init__(render_mode=render_mode)

    def _destroy(self):
        if self.boulder:
            self.world.DestroyBody(self.boulder)
            self.boulder = None
        super()._destroy()

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._create_boulder()
        return obs, info

    def _create_boulder(self):
        if self.boulder:
            self.world.DestroyBody(self.boulder)
        x = self.hull.position[0] + 9
        y = self.hull.position[1] + 2
        radius = 1.5
        offsets = [1.0, 0.96, 1.0, 0.94, 1.0, 0.97, 0.98, 1.0]
        vertices = [(radius * offsets[i] * np.cos(2 * np.pi * i / 8),
                     radius * offsets[i] * np.sin(2 * np.pi * i / 8)) for i in range(8)]
        self.boulder = self.world.CreateDynamicBody(
            position=(x, y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=vertices),
                density=1.5,
                friction=0.8,
                restitution=0.05
            )
        )
        self.prev_boulder_x = x

    def _draw_boulder(self):
        if self.render_mode != "human" or not self.boulder:
            return
        screen = pygame.display.get_surface()
        if screen is None:
            return
        scroll = getattr(self, 'scroll', 0)
        for f in self.boulder.fixtures:
            verts = [(self.boulder.transform * v) for v in f.shape.vertices]
            pts = [(int((v[0] - scroll) * 30), int(400 - v[1] * 30)) for v in verts]
            pygame.draw.polygon(screen, (120, 110, 90), pts)
            pygame.draw.polygon(screen, (80, 70, 55), pts, width=3)
        pygame.display.flip()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.boulder:
            # REWARD SYSTEM
            # 1. Progress reward — agent earns points when boulder moves forward
            progress = self.boulder.position[0] - self.prev_boulder_x
            reward += progress * 2
            self.prev_boulder_x = self.boulder.position[0]
            # 2. Proximity bonus — agent earns 0.1 for staying close to boulder
            dist = abs(self.boulder.position[0] - self.hull.position[0])
            if dist < 2:
                reward += 0.1
        self._draw_boulder()
        return obs, reward, terminated, truncated, info

gym.register(id="SisyphusWalker-v0", entry_point="sisyphus_env:SisyphusEnv")
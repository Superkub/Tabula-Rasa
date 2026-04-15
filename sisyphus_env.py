# Sisyphus environment — BipedalWalker with a boulder
import numpy as np                                 # Math for vertices and observations
import pygame                                      # Drawing the boulder visually
import gymnasium as gym                            # Register environment and observation space
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker  # Base environment we extend
from Box2D.b2 import polygonShape, fixtureDef      # Boulder shape and physics

class SisyphusEnv(BipedalWalker):
    """Agent must push a boulder forward."""

    def __init__(self, render_mode=None):
        self.boulder = None
        self.prev_boulder_x = 0
        super().__init__(render_mode=render_mode)
        high = np.array([np.inf] * 28, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)  # 24 original + 4 boulder

    def _destroy(self):                              # Clean up boulder when environment resets
        if self.boulder:
            self.world.DestroyBody(self.boulder)
            self.boulder = None
        super()._destroy()

    def reset(self, **kwargs):                       # Reset environment and create new boulder
        obs, info = super().reset(**kwargs)
        self._create_boulder()
        return self._add_boulder_obs(obs), info

    def _create_boulder(self):                       # Create the boulder as a physics object
        if self.boulder:
            self.world.DestroyBody(self.boulder)
        x = self.hull.position[0] + 9               # Placed ahead of agent
        y = self.hull.position[1] + 2               # Slightly above ground
        radius = 1.5                                 # Size of boulder
        offsets = [1.0, 0.96, 1.0, 0.94, 1.0, 0.97, 0.98, 1.0]  # Subtle bumps for rock look
        vertices = [(radius * offsets[i] * np.cos(2 * np.pi * i / 8),
                     radius * offsets[i] * np.sin(2 * np.pi * i / 8)) for i in range(8)]
        self.boulder = self.world.CreateDynamicBody(
            position=(x, y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=vertices),
                density=1.5,                         # Weight of boulder
                friction=0.8,                        # Grip on ground
                restitution=0.05                     # Almost no bounce
            )
        )
        self.prev_boulder_x = x

    def _add_boulder_obs(self, obs):                 # Add boulder info to agent's observations
        if len(obs) > 24:                            # Already has boulder obs — skip
            return obs.astype(np.float32)
        if self.boulder and self.hull:
            return np.concatenate([obs, [
                (self.boulder.position[0] - self.hull.position[0]) / 10,  # Relative x
                (self.boulder.position[1] - self.hull.position[1]) / 10,  # Relative y
                self.boulder.linearVelocity[0] / 5,                      # Boulder speed x
                self.boulder.linearVelocity[1] / 5,                      # Boulder speed y
            ]]).astype(np.float32)
        return np.concatenate([obs, [0, 0, 0, 0]]).astype(np.float32)

    def _draw_boulder(self):                         # Draw the boulder on screen with Pygame
        if self.render_mode != "human" or not self.boulder:
            return
        screen = pygame.display.get_surface()
        if screen is None:
            return
        scroll = getattr(self, 'scroll', 0)          # Camera scroll position
        for f in self.boulder.fixtures:
            verts = [(self.boulder.transform * v) for v in f.shape.vertices]
            pts = [(int((v[0] - scroll) * 30), int(400 - v[1] * 30)) for v in verts]
            pygame.draw.polygon(screen, (120, 110, 90), pts)        # Stone gray-brown fill
            pygame.draw.polygon(screen, (80, 70, 55), pts, width=3) # Dark edge outline
        pygame.display.flip()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.boulder:
            # REWARD SYSTEM
            # 1. Progress reward — agent earns points when boulder moves forward
            progress = self.boulder.position[0] - self.prev_boulder_x
            reward += progress * 5
            self.prev_boulder_x = self.boulder.position[0]
            # 2. Proximity bonus — agent earns 0.1 for staying close to boulder
            dist = abs(self.boulder.position[0] - self.hull.position[0])
            if dist < 2:
                reward += 0.1
        self._draw_boulder()
        return self._add_boulder_obs(obs), reward, terminated, truncated, info

gym.register(id="SisyphusWalker-v0", entry_point="sisyphus_env:SisyphusEnv")  # Register so gym.make() works
# Sisyphus environment — BipedalWalker with a boulder and slope
import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker, TERRAIN_HEIGHT
from Box2D.b2 import polygonShape, fixtureDef

class SisyphusEnv(BipedalWalker):
    """Agent must push a boulder up a slope."""

    def __init__(self, render_mode=None):
        self.boulder = None
        self.prev_boulder_x = 0
        self.ramp_body = None
        self.ramp_visual = None
        self.ramp_start = 0
        self.ramp_length = 30
        self.ramp_height = 10
        super().__init__(render_mode=render_mode)

    def _destroy(self):
        if self.boulder:
            self.world.DestroyBody(self.boulder)
            self.boulder = None
        if self.ramp_body:
            self.world.DestroyBody(self.ramp_body)
            self.ramp_body = None
        if self.ramp_visual:
            self.world.DestroyBody(self.ramp_visual)
            self.ramp_visual = None
        super()._destroy()

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._create_ramp()
        self._create_boulder()
        visual_bodies = [self.ramp_visual] if self.ramp_visual else []
        self.drawlist = self.terrain + self.legs + [self.hull, self.boulder] + visual_bodies
        return obs, info

    def _get_ramp_profile(self, steps=20):
        """Generate a mountain profile that gets steeper toward the top."""
        points = []
        for i in range(steps + 1):
            t = i / steps
            x = self.ramp_start + self.ramp_length * t
            y = TERRAIN_HEIGHT + self.ramp_height * (t ** 2)
            y += np.sin(t * np.pi * 3) * 0.3
            y += np.sin(t * np.pi * 5) * 0.15
            y += np.cos(t * np.pi * 7) * 0.1
            points.append((x, y))
        return points

    def _create_ramp(self):
        if self.ramp_body:
            self.world.DestroyBody(self.ramp_body)
        if self.ramp_visual:
            self.world.DestroyBody(self.ramp_visual)
        self.ramp_start = self.hull.position[0] + 35
        profile = self._get_ramp_profile(steps=20)
        # Physics — edge fixtures for collision
        self.ramp_body = self.world.CreateStaticBody()
        for i in range(len(profile) - 1):
            self.ramp_body.CreateEdgeFixture(
                vertices=[profile[i], profile[i + 1]],
                friction=2.0
            )
        # Visual — polygon segments for rendering via drawlist
        self.ramp_visual = self.world.CreateStaticBody(position=(0, 0))
        for i in range(len(profile) - 1):
            x1, y1 = profile[i]
            x2, y2 = profile[i + 1]
            verts = [(x1, TERRAIN_HEIGHT), (x1, y1), (x2, y2), (x2, TERRAIN_HEIGHT)]
            try:
                self.ramp_visual.CreateFixture(
                    fixtureDef(
                        shape=polygonShape(vertices=verts),
                        isSensor=True
                    )
                )
            except:
                pass
        self.ramp_visual.color1 = (90, 80, 60)
        self.ramp_visual.color2 = (60, 50, 35)

    def _create_boulder(self):
        if self.boulder:
            self.world.DestroyBody(self.boulder)
        x = self.hull.position[0] + 20
        y = self.hull.position[1] + 2
        radius = 1.5
        offsets = [1.0, 0.96, 1.0, 0.94, 1.0, 0.97, 0.98, 1.0]
        vertices = [(radius * offsets[i] * np.cos(2 * np.pi * i / 8),
                     radius * offsets[i] * np.sin(2 * np.pi * i / 8)) for i in range(8)]
        self.boulder = self.world.CreateDynamicBody(
            position=(x, y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=vertices),
                density=0.5,
                friction=0.8,
                restitution=0.05
            )
        )
        self.boulder.color1 = (120, 110, 90)
        self.boulder.color2 = (80, 70, 55)
        self.prev_boulder_x = x

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
        return obs, reward, terminated, truncated, info

gym.register(id="SisyphusWalker-v0", entry_point="sisyphus_env:SisyphusEnv")
import pygame
import gymnasium as gym
from gymnasium import Wrapper

DARK_GRAY  = (20,  24,  36)
WHITE      = (255, 255, 255)
NEON_CYAN  = (0,   230, 255)
NEON_GREEN = (0,   255, 140)
NEON_RED   = (255, 60,  80)
NEON_GOLD  = (255, 210, 0)
DIM_GRAY   = (80,  90,  110)

SISYPHUS_QUOTES = [
    "The stone slips...",
    "Still climbing.",
    "Each fall is a lesson.",
    "The struggle is enough.",
    "Rise again.",
    "One more attempt.",
    "The hill is steep.",
    "Do not look back.",
    "Keep moving forward.",
    "One must imagine Sisyphus happy.",
]

class TabulaRasaEnv(Wrapper):
    def __init__(self, render_mode="human"):
        env = gym.make("BipedalWalker-v3", render_mode=render_mode)
        super().__init__(env)
        self.episode       = 0
        self.total_reward  = 0.0
        self.status        = ""
        self.quote_index   = 0
        self._render_mode  = render_mode
        self.max_reward    = 300.0
        self._fonts_loaded = False

    def _load_fonts(self):
        if not self._fonts_loaded:
            pygame.font.init()
            self.font_large  = pygame.font.SysFont("consolas", 36, bold=True)
            self.font_medium = pygame.font.SysFont("consolas", 24, bold=True)
            self.font_small  = pygame.font.SysFont("consolas", 16)
            self._fonts_loaded = True

    def reset(self, **kwargs):
        self.episode      += 1
        self.total_reward  = 0.0
        self.status        = ""
        self.quote_index   = (self.episode - 1) % len(SISYPHUS_QUOTES)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        if terminated or truncated:
            self.status = "SUCCESS" if self.total_reward >= 200 else "FAILED"
        return obs, reward, terminated, truncated, info

    def render(self):
        result = self.env.render()
        if self._render_mode != "human":
            return result
        screen = pygame.display.get_surface()
        if screen is None:
            return result
        self._load_fonts()
        W, H = screen.get_size()

        panel = pygame.Surface((260, 130), pygame.SRCALPHA)
        panel.fill((15, 18, 30, 180))
        screen.blit(panel, (16, 16))
        pygame.draw.rect(screen, NEON_CYAN, (16, 16, 260, 130), width=1, border_radius=4)

        screen.blit(self.font_medium.render(f"EPISODE  {self.episode}", True, NEON_CYAN), (28, 24))

        if self.status == "FAILED":
            sc, si = NEON_RED, "✕  FAILED"
        elif self.status == "SUCCESS":
            sc, si = NEON_GREEN, "✓  SUCCESS"
        else:
            sc, si = DIM_GRAY, "●  RUNNING"
        screen.blit(self.font_small.render(si, True, sc), (28, 58))
        screen.blit(self.font_small.render(f'"{SISYPHUS_QUOTES[self.quote_index]}"', True, NEON_GOLD), (28, 82))

        bx, by, bh, bw = W - 70, 20, H - 80, 30
        pygame.draw.rect(screen, DARK_GRAY, (bx, by, bw, bh), border_radius=6)
        fill_ratio  = max(0, min(self.total_reward / self.max_reward, 1.0))
        fill_height = int(bh * fill_ratio)
        if fill_height > 0:
            bc = NEON_GREEN if fill_ratio > 0.66 else NEON_GOLD if fill_ratio > 0.33 else NEON_RED
            pygame.draw.rect(screen, bc, (bx, by + bh - fill_height, bw, fill_height), border_radius=6)
        screen.blit(self.font_small.render("REWARD", True, DIM_GRAY), (bx - 10, by - 20))
        screen.blit(self.font_small.render(f"{int(self.total_reward)}", True, WHITE), (bx + 2, by + bh + 4))

        caption = self.font_small.render("Agent learns through rewards & penalties  •  PPO Algorithm", True, DIM_GRAY)
        screen.blit(caption, (W // 2 - caption.get_width() // 2, H - 24))
        title = self.font_large.render("Trial & Error Learning", True, WHITE)
        screen.blit(title, (W // 2 - title.get_width() // 2, 16))

        pygame.display.flip()
        return result

def make_tabula_rasa_env(render_mode="human"):
    return TabulaRasaEnv(render_mode=render_mode)
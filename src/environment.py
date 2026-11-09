"""
src/environment.py
2D Grid Cellular Automata simulating anisotropic wildfire propagation
governed by biomass/fuel density, dynamic ambient wind vectors, slope, and smoke plumes.
"""

import numpy as np
from scipy.signal import correlate2d


class ForestEnvironment:
    def __init__(self, width=100, height=100, wind_vector=(2.8, 1.8), c_slope=0.1):
        self.W = width
        self.H = height
        self.c_slope = c_slope

        # State: 0 = Unburned, 1 = Burning, 2 = Ash
        self.state = np.zeros((height, width), dtype=int)
        self.fuel = np.ones((height, width), dtype=float)
        self.smoke = np.zeros((height, width), dtype=float)
        self.canopy = np.random.uniform(0.1, 0.65, (height, width))

        # Topography (Elevation elevation gradient for slope calculation)
        X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        self.elevation = 40.0 * np.sin(2.5 * X) + 30.0 * np.cos(2.0 * Y)

        self.wind = np.array(wind_vector, dtype=float)
        norm = np.linalg.norm(self.wind)
        self.wind_dir = self.wind / (norm + 1e-6) if norm > 0 else np.zeros(2)
        self.wind_speed = norm

    def ignite(self, x, y, radius=2):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx_c, ny_c = x + dx, y + dy
                if 0 <= nx_c < self.W and 0 <= ny_c < self.H:
                    self.state[ny_c, nx_c] = 1

    def step(self, base_prob=0.18, fuel_burn_rate=0.35):
        burning_indices = np.argwhere(self.state == 1)
        new_state = self.state.copy()

        for y, x in burning_indices:
            self.fuel[y, x] -= fuel_burn_rate
            if self.fuel[y, x] <= 0.0:
                new_state[y, x] = 2

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx_c, ny_c = x + dx, y + dy
                    if 0 <= nx_c < self.W and 0 <= ny_c < self.H and self.state[ny_c, nx_c] == 0:
                        disp = np.array([dx, dy], dtype=float)
                        disp_dir = disp / np.linalg.norm(disp)

                        wind_alignment = np.dot(self.wind_dir, disp_dir)
                        wind_factor = max(0.0, wind_alignment) * self.wind_speed

                        dz = self.elevation[ny_c, nx_c] - self.elevation[y, x]
                        slope_factor = self.c_slope * (dz / (np.linalg.norm(disp) + 1e-6))

                        p_ignite = base_prob * (1.0 + 1.2 * wind_factor + max(0.0, slope_factor))
                        if np.random.rand() < p_ignite:
                            new_state[ny_c, nx_c] = 1

        self.state = new_state

        # Atmospheric smoke plume drift & decay
        self.smoke *= 0.88
        for y, x in burning_indices:
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    sx = int(x + dx + self.wind_dir[0] * 2.0)
                    sy = int(y + dy + self.wind_dir[1] * 2.0)
                    if 0 <= sx < self.W and 0 <= sy < self.H:
                        dist = np.sqrt(dx**2 + dy**2)
                        self.smoke[sy, sx] = min(1.0, self.smoke[sy, sx] + np.exp(-dist / 2.5) * 0.6)

    def get_fire_centroid(self):
        burning = np.argwhere(self.state == 1)
        if len(burning) > 0:
            return np.array([np.mean(burning[:, 1]), np.mean(burning[:, 0])])
        burned = np.argwhere(self.state == 2)
        if len(burned) > 0:
            return np.array([np.mean(burned[:, 1]), np.mean(burned[:, 0])])
        return np.array([self.W / 2.0, self.H / 2.0])

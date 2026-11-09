import numpy as np
from scipy.signal import correlate2d

class ForestEnvironment:
    """
    2D Grid Cellular Automata simulating anisotropic wildfire propagation
    governed by biomass/fuel density, dynamic ambient wind vectors, and slope.
    Includes dynamic smoke plume diffusion.
    """
    def __init__(self, width=100, height=100, wind_vector=(3.0, 2.0), c_slope=0.1):
        self.W = width
        self.H = height
        self.c_slope = c_slope

        self.state = np.zeros((height, width), dtype=int)  # 0 = Unburned, 1 = Burning, 2 = Ash
        self.fuel = np.ones((height, width), dtype=float)
        self.smoke = np.zeros((height, width), dtype=float)

        # Generate topographical elevation map
        Y, X = np.indices((height, width))
        self.z_elev = 15.0 * np.sin(X / 15.0) + 10.0 * np.cos(Y / 20.0) + 0.5 * Y

        # Tree canopy obstruction field
        self.canopy = np.random.uniform(0.1, 0.65, (height, width))

        # Wind dynamics
        self.wind = np.array(wind_vector, dtype=float)
        norm = np.linalg.norm(self.wind)
        self.wind_dir = self.wind / (norm + 1e-6) if norm > 0 else np.zeros(2)
        self.wind_speed = norm

    def ignite(self, x, y, radius=1):
        """Sparks initial combustion seeds in a local neighborhood."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx_c, ny_c = x + dx, y + dy
                if 0 <= nx_c < self.W and 0 <= ny_c < self.H:
                    self.state[ny_c, nx_c] = 1

    def step(self, base_prob=0.18, fuel_burn_rate=0.4):
        """Executes one discrete cellular automaton step over the grid."""
        burning_mask = (self.state == 1)
        burning_indices = np.argwhere(burning_mask)
        new_state = self.state.copy()

        for y, x in burning_indices:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx_c, ny_c = x + dx, y + dy
                    if 0 <= nx_c < self.W and 0 <= ny_c < self.H and self.state[ny_c, nx_c] == 0:
                        disp = np.array([dx, dy], dtype=float)
                        disp_dir = disp / np.linalg.norm(disp)

                        # Wind alignment 
                        wind_alignment = np.dot(self.wind_dir, disp_dir)
                        wind_factor = max(0.0, wind_alignment) * self.wind_speed

                        # Elevation slope alignment
                        dh = self.z_elev[ny_c, nx_c] - self.z_elev[y, x]
                        run_dist = np.linalg.norm(disp)
                        slope = dh / run_dist
                        slope_factor = self.c_slope * np.arctan(slope)

                        # Calculate ignition probability
                        p_ignite = base_prob * (1.0 + 1.2 * wind_factor) * (1.0 + slope_factor)
                        if np.random.rand() < p_ignite:
                            new_state[ny_c, nx_c] = 1

            # Burn fuel and turn to ash if exhausted
            self.fuel[y, x] = max(0.0, self.fuel[y, x] - fuel_burn_rate)
            if self.fuel[y, x] <= 0.0:
                new_state[y, x] = 2

        self.state = new_state

        # Update smoke field using 2D correlation (diffusion)
        fresh_smoke = np.zeros_like(self.smoke)
        fresh_smoke[self.state == 1] = 0.8

        wx, wy = int(np.round(self.wind_dir[0])), int(np.round(self.wind_dir[1]))
        kernel = np.array([
            [0.05, 0.10, 0.05],
            [0.10, 0.40, 0.10],
            [0.05, 0.10, 0.05]
        ])
        kernel = np.roll(kernel, (wy, wx), axis=(0, 1))

        self.smoke = correlate2d(self.smoke + fresh_smoke, kernel, mode='same', boundary='symm') * 0.82
        self.smoke = np.clip(self.smoke, 0.0, 1.0)

    def get_fire_centroid(self):
        """Calculates center-of-mass of flame front to act as repellent beacon."""
        burning = np.argwhere(self.state == 1)
        if len(burning) > 0:
            return np.array([np.mean(burning[:, 1]), np.mean(burning[:, 0])])
        return np.array([self.W / 2.0, self.H / 2.0])

"""
live_command_center.py (V2 - Precision Slow-Burn & Telemetry Inspector Edition)
PROJECT FIRE-SAR: Interactive Multi-Agent Tactical Command Center (AI3403)
"""

import sys
import numpy as np
import pygame

from src.environment import ForestEnvironment
from src.agents import SurvivorAgent, UAVAgent, RescueVehicle
from src.safety import ActiveSTLController, EpistemicLogicGuard
from src.coordination import GraphManager, ADMMConsensusOptimizer, RescueDispatchManager

# --- Window & Grid Configuration ---
WINDOW_WIDTH = 1420
WINDOW_HEIGHT = 880
MAP_SIZE = 800
MAP_OFFSET_X = 30
MAP_OFFSET_Y = 40
GRID_DIM = 100
CELL_PX = MAP_SIZE // GRID_DIM  # 8x8 pixels per grid cell

# --- Color Palette ---
BG_DARK = (12, 16, 23)
PANEL_BG = (20, 27, 38)
PANEL_BORDER = (42, 58, 78)
TEXT_WHITE = (240, 244, 248)
TEXT_MUTED = (148, 163, 184)
CYAN_ACCENT = (0, 229, 255)
GREEN_SAFE = (0, 230, 118)
RED_ALERT = (255, 23, 68)
AMBER_WARN = (255, 179, 0)
PURPLE_RV = (195, 115, 215)
TRAIL_BROWN = (95, 75, 55)


class TacticalCommandCenterV2:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("FIRE-SAR V2 | Precision Multi-Agent Tactical Command Center")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Typography
        self.font_title = pygame.font.SysFont("consolas", 19, bold=True)
        self.font_hdr = pygame.font.SysFont("consolas", 14, bold=True)
        self.font_body = pygame.font.SysFont("consolas", 12)
        self.font_small = pygame.font.SysFont("consolas", 11)

        # Interactive Presentation Controls
        self.paused = False
        self.show_comms = True
        self.show_sensors = True
        self.show_admm_lines = True
        self.speed_multiplier = 1.0  # 0.5x, 1.0x, 2.0x
        self.frame_counter = 0

        # Inspector Selection: ('UAV', id) or ('SURV', id)
        self.selected_entity = ('UAV', 0)

        self.reset_simulation()

    def reset_simulation(self):
        np.random.seed(42)
        self.sim_time = 0.0
        self.base_dt = 0.12  # Gentle kinematic step for smooth visual tracking
        self.stl_interventions = 0
        self.event_log = ["[SYS] Command Center V2 Ready. Slow-burn physics active."]

        # 1. Initialize Forest Environment
        self.env = ForestEnvironment(width=GRID_DIM, height=GRID_DIM, wind_vector=(1.8, 1.1))
        self.env.ignite(26, 26, radius=2)
        self.env.ignite(64, 22, radius=2)
        self.log_event("Slow-burn wildfire ignited at (26,26) and (64,22).")

        # Hiking Trails toward Safe Zone (88, 88)
        self.trails = [
            [(15, 20), (35, 35), (55, 50), (75, 70), (88, 88)],
            [(70, 15), (68, 35), (62, 55), (75, 70), (88, 88)],
            [(15, 75), (40, 68), (62, 55), (88, 88)]
        ]

        # 2. Spawn 10 Heterogeneous Survivors
        self.survivors = [
            SurvivorAgent(0, [32.0, 34.0], voice_radius=10.0, mobility=0.35, goal=[88.0, 88.0]),
            SurvivorAgent(1, [34.5, 32.5], voice_radius=10.0, mobility=0.40, goal=[88.0, 88.0]),
            SurvivorAgent(2, [42.0, 39.0], voice_radius=10.0, mobility=0.85, goal=[88.0, 88.0]),
            SurvivorAgent(3, [44.0, 41.0], voice_radius=10.0, mobility=0.90, goal=[88.0, 88.0]),
            SurvivorAgent(4, [54.0, 48.0], voice_radius=10.0, mobility=0.80, goal=[88.0, 88.0]),
            SurvivorAgent(5, [57.0, 51.0], voice_radius=10.0, mobility=0.95, goal=[88.0, 88.0]),
            SurvivorAgent(6, [66.0, 29.0], voice_radius=10.0, mobility=0.35, goal=[88.0, 88.0]),
            SurvivorAgent(7, [68.5, 31.0], voice_radius=10.0, mobility=0.45, goal=[88.0, 88.0]),
            SurvivorAgent(8, [26.0, 66.0], voice_radius=10.0, mobility=0.90, goal=[88.0, 88.0]),
            SurvivorAgent(9, [28.5, 68.0], voice_radius=10.0, mobility=1.00, goal=[88.0, 88.0])
        ]
        for s in self.survivors:
            s.health = 75.0 if s.mobility < 0.5 else 100.0

        # 3. Spawn 5 Heterogeneous UAVs
        self.uavs = [
            UAVAgent(0, pos=[15.0, 72.0], home_pos=[6.0, 94.0], role='Fast-Scout', v_max=4.8, comm_radius=36.0, sense_radius=15.0),
            UAVAgent(1, pos=[26.0, 64.0], home_pos=[6.0, 94.0], role='Fast-Scout', v_max=4.6, comm_radius=36.0, sense_radius=14.0),
            UAVAgent(2, pos=[40.0, 74.0], home_pos=[6.0, 94.0], role='Surveyor',   v_max=4.0, comm_radius=34.0, sense_radius=17.0),
            UAVAgent(3, pos=[60.0, 70.0], home_pos=[6.0, 94.0], role='Surveyor',   v_max=4.0, comm_radius=34.0, sense_radius=17.0),
            UAVAgent(4, pos=[74.0, 55.0], home_pos=[6.0, 94.0], role='Relay-Hub',  v_max=3.6, comm_radius=42.0, sense_radius=13.0, battery=40.0)
        ]

        self.patrol_angles = [np.random.uniform(0, 2 * np.pi) for _ in self.uavs]

        # 4. Spawn Ground Rescue Vehicles
        self.rvs = [
            RescueVehicle(0, pos=[14.0, 14.0], base_pos=[5.0, 5.0], max_speed=4.2, capacity=3),
            RescueVehicle(1, pos=[86.0, 14.0], base_pos=[94.0, 5.0], max_speed=4.2, capacity=3)
        ]

        # 5. Formal Safety & Graph Topology
        self.stl_ctrl = ActiveSTLController(d_safe=5.5, d_fire=9.5, h_min=25.0, buffer_zone=7.5, k_repel=8.5)
        self.A_u, self.D_u, self.L_u, self.lambda_2 = GraphManager.compute_uav_matrices(self.uavs)
        self.safety_margins = {'uav_spacing': 10.0, 'fire_clearance': 10.0, 'survivor_health': 50.0}
        self.allocations = {}

    def log_event(self, text):
        timestamp = f"[{self.sim_time:05.1f}s] "
        self.event_log.insert(0, timestamp + text)
        if len(self.event_log) > 8:
            self.event_log.pop()

    def grid_to_screen(self, pos):
        sx = int(MAP_OFFSET_X + pos[0] * CELL_PX)
        sy = int(MAP_OFFSET_Y + (GRID_DIM - pos[1]) * CELL_PX)
        return sx, sy

    def screen_to_grid(self, sx, sy):
        gx = float(sx - MAP_OFFSET_X) / CELL_PX
        gy = float(GRID_DIM - (sy - MAP_OFFSET_Y) / CELL_PX)
        return gx, gy

    def step_physics(self):
        dt = self.base_dt * self.speed_multiplier
        self.sim_time += dt

        # --- SLOW-BURN FIRE PROPAGATION ---
        # Only step the cellular automaton every 30 frames (approx twice per second)
        # with a low ignition probability (0.045) so fire spreads gradually and visibly
        fire_interval = max(8, int(30 / self.speed_multiplier))
        if self.frame_counter % fire_interval == 0:
            self.env.step(base_prob=0.045, fuel_burn_rate=0.04)

        fc = self.env.get_fire_centroid()

        # 1. Step Survivors (Gentle movement so class can watch group splitting)
        for s in self.survivors:
            if not s.rescued and s.health > 0:
                s.step(fc, self.env, dt=dt * 0.65)

        active_survivors = [s for s in self.survivors if not s.rescued and s.health > 0]

        # 2. UAV Sensing
        for u in self.uavs:
            det = u.sense_survivors(active_survivors, self.env, self.sim_time)
            for sid in det:
                if np.random.rand() < 0.04:
                    self.log_event(f"UAV {u.id} sensed S{sid} (P_D={u.belief_map[sid]['conf']:.2f})")

        # 3. Dynamic Graph & Consensus Synchronization
        self.A_u, self.D_u, self.L_u, self.lambda_2 = GraphManager.compute_uav_matrices(self.uavs)
        GraphManager.synchronize_and_track_gradients(self.uavs, self.A_u, self.sim_time)
        GraphManager.handle_tracking_handoff(self.uavs)

        # 4. UAV Kinematics & Active STL Shield
        for idx, u in enumerate(self.uavs):
            if u.pos[0] < 9 or u.pos[0] > 91:
                self.patrol_angles[idx] = np.pi - self.patrol_angles[idx]
            if u.pos[1] < 9 or u.pos[1] > 91:
                self.patrol_angles[idx] = -self.patrol_angles[idx]

            default_cmd = np.array([np.cos(self.patrol_angles[idx]), np.sin(self.patrol_angles[idx])]) * (u.v_max * 0.7)
            has_k = any(EpistemicLogicGuard.satisfies_knowledge_check(u, s.id, self.sim_time) for s in active_survivors)
            f_risk = 1.0 if np.linalg.norm(u.pos - fc) < 15.0 else 0.0

            u_nom = u.update_behavior(self.sim_time, default_cmd, has_knowledge=has_k, fire_risk=f_risk)
            u_safe, intervened = self.stl_ctrl.apply_safety_shield(idx, u_nom, self.uavs, self.env)
            if intervened:
                self.stl_interventions += 1
            u.step(u_safe, self.env, dt=dt)

        self.safety_margins = self.stl_ctrl.evaluate_safety_margins(self.uavs, self.env, self.survivors)

        # 5. ADMM Rescue Allocation & Epistemic Dispatch
        self.allocations = ADMMConsensusOptimizer.run_assignment(self.uavs, self.rvs, active_survivors, max_iters=12)
        prev_rescued = sum(1 for s in self.survivors if s.rescued)
        RescueDispatchManager.step_rescue_vehicles(
            self.rvs, self.survivors, self.uavs, self.allocations, self.env, self.sim_time, dt=dt, require_epistemic=True
        )
        now_rescued = sum(1 for s in self.survivors if s.rescued)
        if now_rescued > prev_rescued:
            self.log_event(f"RESCUE COMPLETE! Survivor evacuated ({now_rescued}/{len(self.survivors)} safe).")

    def draw_map(self):
        # Top Banner over Map
        banner = f"TACTICAL FOREST GRID (100x100m) | Speed: {self.speed_multiplier:.1f}x | Right-Click Any Agent to Inspect"
        self.screen.blit(self.font_hdr.render(banner, True, CYAN_ACCENT), (MAP_OFFSET_X, 12))

        pygame.draw.rect(self.screen, PANEL_BORDER, (MAP_OFFSET_X - 2, MAP_OFFSET_Y - 2, MAP_SIZE + 4, MAP_SIZE + 4), 2)

        # 1. Render Terrain, Elevation Shading, Fire, Ash, and Smoke
        for y in range(GRID_DIM):
            for x in range(GRID_DIM):
                sx = MAP_OFFSET_X + x * CELL_PX
                sy = MAP_OFFSET_Y + (GRID_DIM - 1 - y) * CELL_PX
                state = self.env.state[y, x]
                canopy = self.env.canopy[y, x]
                elev = self.env.elevation[y, x]
                smoke = self.env.smoke[y, x]

                if state == 0:
                    # Topographical contour line highlight
                    is_contour = abs(elev % 15.0) < 1.1
                    g = int(68 + 60 * canopy + (18 if is_contour else 0))
                    col = (20, min(210, g), 34)
                elif state == 1:
                    # Remaining fuel determines flame color (Bright yellow-orange -> deep red)
                    fuel_lvl = self.env.fuel[y, x]
                    g_flame = int(40 + 135 * fuel_lvl)
                    col = (250, max(30, min(220, g_flame)), 15)
                else:
                    col = (44, 48, 54)

                if smoke > 0.06 and state != 1:
                    s_alpha = min(0.72, smoke * 0.75)
                    col = (
                        int(col[0] * (1 - s_alpha) + 185 * s_alpha),
                        int(col[1] * (1 - s_alpha) + 195 * s_alpha),
                        int(col[2] * (1 - s_alpha) + 208 * s_alpha)
                    )

                pygame.draw.rect(self.screen, col, (sx, sy, CELL_PX, CELL_PX))

        # 2. Draw Hiking Trails
        for trail in self.trails:
            pts = [self.grid_to_screen(p) for p in trail]
            pygame.draw.lines(self.screen, TRAIL_BROWN, False, pts, 2)

        # 3. Draw Safe Evacuation Zone & Bases
        gx, gy = self.grid_to_screen([88.0, 88.0])
        pygame.draw.circle(self.screen, GREEN_SAFE, (gx, gy), 24, 2)
        self.screen.blit(self.font_small.render("SAFE ZONE", True, GREEN_SAFE), (gx - 28, gy - 36))

        for rv in self.rvs:
            bx, by = self.grid_to_screen(rv.base_pos)
            pygame.draw.rect(self.screen, PURPLE_RV, (bx - 10, by - 10, 20, 20), 2)
            self.screen.blit(self.font_small.render(f"RV{rv.id} BASE", True, PURPLE_RV), (bx - 22, by + 12))

        if self.uavs:
            hx, hy = self.grid_to_screen(self.uavs[0].home_pos)
            pygame.draw.polygon(self.screen, CYAN_ACCENT, [(hx, hy - 10), (hx - 10, hy + 8), (hx + 10, hy + 8)], 2)
            self.screen.blit(self.font_small.render("UAV BASE", True, CYAN_ACCENT), (hx - 22, hy + 10))

        # 4. Draw ADMM Dispatch Target Vectors
        if self.show_admm_lines:
            surv_map = {s.id: s for s in self.survivors}
            for rv in self.rvs:
                if rv.assigned_survivor_id is not None and rv.assigned_survivor_id in surv_map:
                    st = surv_map[rv.assigned_survivor_id]
                    if not st.rescued and st.health > 0:
                        p_rv = self.grid_to_screen(rv.pos)
                        p_st = self.grid_to_screen(st.pos)
                        pygame.draw.line(self.screen, PURPLE_RV, p_rv, p_st, 2)
                        mid = ((p_rv[0] + p_st[0]) // 2, (p_rv[1] + p_st[1]) // 2)
                        eta = np.linalg.norm(st.pos - rv.pos) / rv.max_speed
                        self.screen.blit(self.font_small.render(f"ADMM ETA:{eta:.1f}s", True, PURPLE_RV), mid)

        # 5. Draw Communication Graph G_U(t) & Voice Mesh G_S(t)
        if self.show_comms:
            for i in range(len(self.survivors)):
                for j in range(i + 1, len(self.survivors)):
                    s1, s2 = self.survivors[i], self.survivors[j]
                    if not s1.rescued and not s2.rescued and s1.health > 0 and s2.health > 0:
                        if np.linalg.norm(s1.pos - s2.pos) <= min(s1.voice_radius, s2.voice_radius):
                            pygame.draw.line(self.screen, AMBER_WARN, self.grid_to_screen(s1.pos), self.grid_to_screen(s2.pos), 1)

            for i in range(len(self.uavs)):
                for j in range(i + 1, len(self.uavs)):
                    if self.A_u[i, j] == 1.0:
                        p1 = self.grid_to_screen(self.uavs[i].pos)
                        p2 = self.grid_to_screen(self.uavs[j].pos)
                        pygame.draw.line(self.screen, GREEN_SAFE, p1, p2, 2)

        # 6. Draw Survivors with Health Bars
        for s in self.survivors:
            if s.rescued:
                continue
            sx, sy = self.grid_to_screen(s.pos)
            if s.health <= 0:
                pygame.draw.line(self.screen, RED_ALERT, (sx - 5, sy - 5), (sx + 5, sy + 5), 2)
                pygame.draw.line(self.screen, RED_ALERT, (sx - 5, sy + 5), (sx + 5, sy - 5), 2)
            else:
                col = AMBER_WARN if s.mobility >= 0.5 else (255, 64, 129)
                pygame.draw.circle(self.screen, col, (sx, sy), 6)
                pygame.draw.circle(self.screen, (0, 0, 0), (sx, sy), 6, 1)
                # Mini Health Bar above survivor
                pygame.draw.rect(self.screen, (30, 30, 30), (sx - 10, sy - 12, 20, 4))
                h_col = GREEN_SAFE if s.health > 50 else RED_ALERT
                pygame.draw.rect(self.screen, h_col, (sx - 10, sy - 12, int(0.2 * s.health), 4))
                self.screen.blit(self.font_small.render(f"S{s.id}", True, TEXT_WHITE), (sx + 8, sy - 6))

            if self.selected_entity == ('SURV', s.id):
                pygame.draw.circle(self.screen, AMBER_WARN, (sx, sy), 14, 2)

        # 7. Draw Rescue Vehicles
        for rv in self.rvs:
            rx, ry = self.grid_to_screen(rv.pos)
            pygame.draw.rect(self.screen, PURPLE_RV, (rx - 8, ry - 8, 16, 16))
            pygame.draw.rect(self.screen, TEXT_WHITE, (rx - 8, ry - 8, 16, 16), 1)
            self.screen.blit(self.font_small.render(f"RV{rv.id}[{len(rv.passengers)}/{rv.capacity}]", True, PURPLE_RV), (rx + 10, ry - 6))

        # 8. Draw UAVs, Flight Trails, & Sensor Cones
        for u in self.uavs:
            ux, uy = self.grid_to_screen(u.pos)
            rad_px = int(u.sense_radius * CELL_PX)

            # Recent flight trail
            if len(u.path_history) > 2:
                pts = [self.grid_to_screen(p) for p in u.path_history[-25:]]
                pygame.draw.lines(self.screen, (70, 130, 180), False, pts, 1)

            if self.show_sensors:
                surf = pygame.Surface((rad_px * 2, rad_px * 2), pygame.SRCALPHA)
                cone_col = (0, 229, 255, 24) if u.task != "TRACK" else (0, 230, 118, 42)
                pygame.draw.circle(surf, cone_col, (rad_px, rad_px), rad_px)
                pygame.draw.circle(surf, (0, 229, 255, 105), (rad_px, rad_px), rad_px, 1)
                self.screen.blit(surf, (ux - rad_px, uy - rad_px))

            u_col = CYAN_ACCENT if u.task == "SEARCH" else (GREEN_SAFE if u.task == "TRACK" else AMBER_WARN)
            pygame.draw.circle(self.screen, u_col, (ux, uy), 8)
            pygame.draw.circle(self.screen, TEXT_WHITE, (ux, uy), 8, 2)
            self.screen.blit(self.font_small.render(f"U{u.id}:{u.task[:4]}", True, TEXT_WHITE), (ux + 10, uy - 8))

            if self.selected_entity == ('UAV', u.id):
                pygame.draw.circle(self.screen, AMBER_WARN, (ux, uy), 16, 2)

        # 9. Wind Compass Overlay
        wx, wy = MAP_OFFSET_X + 42, MAP_OFFSET_Y + 42
        pygame.draw.circle(self.screen, (18, 24, 34), (wx, wy), 30)
        pygame.draw.circle(self.screen, CYAN_ACCENT, (wx, wy), 30, 1)
        end_x = int(wx + self.env.wind_dir[0] * 20)
        end_y = int(wy - self.env.wind_dir[1] * 20)
        pygame.draw.line(self.screen, CYAN_ACCENT, (wx, wy), (end_x, end_y), 3)
        self.screen.blit(self.font_small.render(f"WIND {self.env.wind_speed:.1f}m/s", True, CYAN_ACCENT), (wx - 28, wy + 34))

    def draw_telemetry_sidebar(self):
        panel_x = MAP_OFFSET_X + MAP_SIZE + 25
        panel_w = WINDOW_WIDTH - panel_x - 25

        pygame.draw.rect(self.screen, PANEL_BG, (panel_x, MAP_OFFSET_Y, panel_w, MAP_SIZE), border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, (panel_x, MAP_OFFSET_Y, panel_w, MAP_SIZE), 2, border_radius=8)

        y_cur = MAP_OFFSET_Y + 14
        x_pad = panel_x + 16

        # Header
        self.screen.blit(self.font_title.render("FIRE-SAR TELEMETRY & SPECS", True, CYAN_ACCENT), (x_pad, y_cur))
        y_cur += 24
        st_str = "PAUSED" if self.paused else f"MISSION TIME: {self.sim_time:05.1f}s ({self.speed_multiplier:.1f}x)"
        self.screen.blit(self.font_hdr.render(st_str, True, GREEN_SAFE if not self.paused else AMBER_WARN), (x_pad, y_cur))
        y_cur += 22

        # --- 1. Topology & Formal Safety ---
        pygame.draw.line(self.screen, PANEL_BORDER, (x_pad, y_cur), (panel_x + panel_w - 16, y_cur), 1)
        y_cur += 8
        self.screen.blit(self.font_hdr.render("1. ALGEBRAIC GRAPH & STL SAFETY SHIELD", True, TEXT_WHITE), (x_pad, y_cur))
        y_cur += 19

        l2_col = GREEN_SAFE if self.lambda_2 > 0.01 else RED_ALERT
        conn_str = "CONNECTED" if self.lambda_2 > 0.01 else "PARTITIONED"
        self.screen.blit(self.font_body.render(f"Laplacian lambda_2(L_U): {self.lambda_2:.3f} [{conn_str}]", True, l2_col), (x_pad, y_cur))
        y_cur += 16

        r_uav = self.safety_margins['uav_spacing']
        r_fire = self.safety_margins['fire_clearance']
        self.screen.blit(self.font_body.render(f"STL Pairwise Spacing (rho_uav): +{max(0.0, r_uav):.2f}m (d_safe=5.5m)", True, GREEN_SAFE), (x_pad, y_cur))
        y_cur += 16
        self.screen.blit(self.font_body.render(f"STL Fire Clearance (rho_fire):  +{max(0.0, r_fire):.2f}m (d_fire=9.5m)", True, GREEN_SAFE), (x_pad, y_cur))
        y_cur += 16
        self.screen.blit(self.font_body.render(f"Active STL Barrier Repulsions:  {self.stl_interventions}", True, CYAN_ACCENT), (x_pad, y_cur))
        y_cur += 20

        # --- 2. Live Agent Inspector (Right-Clicked Entity) ---
        pygame.draw.line(self.screen, PANEL_BORDER, (x_pad, y_cur), (panel_x + panel_w - 16, y_cur), 1)
        y_cur += 8
        self.screen.blit(self.font_hdr.render("2. LIVE AGENT INSPECTOR (RIGHT-CLICK MAP)", True, AMBER_WARN), (x_pad, y_cur))
        y_cur += 18

        kind, eid = self.selected_entity
        # Inside draw_telemetry_sidebar() -> if kind == 'UAV':
        if kind == 'UAV':
            u_list = [u for u in self.uavs if u.id == eid]
            if u_list:
                u = u_list[0]
                reg = u.allocator.get_regret_metrics()
                self.screen.blit(self.font_body.render(f"Selected: UAV {u.id} ({u.role}) | Pos: ({u.pos[0]:.1f}, {u.pos[1]:.1f})", True, TEXT_WHITE), (x_pad, y_cur))
                y_cur += 16
                self.screen.blit(self.font_body.render(f"RL Action: {u.task} | Target S_ID: {u.target_survivor_id} | Batt: {u.battery:.1f}%", True, CYAN_ACCENT), (x_pad, y_cur))
                y_cur += 16
                # NEW: Live Regret & Counterfactual Advantage Readout
                self.screen.blit(self.font_body.render(
                    f"External Regret R_T: {reg['R_T']:+.1f} | Avg [R_T]+/T: {reg['avg_regret']:.3f} | Adv: {reg['advantage_sum']:+.1f}",
                    True, GREEN_SAFE), (x_pad, y_cur))
                y_cur += 16
                known_str = ", ".join([f"S{k}({v['conf']:.2f})" for k, v in list(u.belief_map.items())[:4]]) or "None"
                self.screen.blit(self.font_body.render(f"Belief Map K_i: {known_str}", True, AMBER_WARN), (x_pad, y_cur))
                y_cur += 20
        else:
            s_list = [s for s in self.survivors if s.id == eid]
            if s_list:
                s = s_list[0]
                self.screen.blit(self.font_body.render(f"Selected: Survivor S{s.id} | Pos: ({s.pos[0]:.1f}, {s.pos[1]:.1f})", True, TEXT_WHITE), (x_pad, y_cur))
                y_cur += 16
                self.screen.blit(self.font_body.render(f"Health: {s.health:.1f}% | Mobility Factor: {s.mobility:.2f} | Rescued: {s.rescued}", True, AMBER_WARN), (x_pad, y_cur))
                y_cur += 36

        # --- 3. Fleet & ADMM Rescue Summary ---
        pygame.draw.line(self.screen, PANEL_BORDER, (x_pad, y_cur), (panel_x + panel_w - 16, y_cur), 1)
        y_cur += 8
        self.screen.blit(self.font_hdr.render("3. FLEET & ADMM RESCUE DISPATCH STATUS", True, TEXT_WHITE), (x_pad, y_cur))
        y_cur += 18

        for idx, u in enumerate(self.uavs):
            deg = int(self.D_u[idx, idx]) if idx < len(self.D_u) else 0
            line = f"U{u.id}[{u.role[:5]:5s}] Task:{u.task:7s} Deg:{deg} Mem:{len(u.belief_map)}/10"
            self.screen.blit(self.font_body.render(line, True, TEXT_WHITE), (x_pad, y_cur))
            bar_x = panel_x + panel_w - 78
            pygame.draw.rect(self.screen, (45, 55, 72), (bar_x, y_cur + 2, 60, 10))
            b_col = GREEN_SAFE if u.battery > 35 else (AMBER_WARN if u.battery > 15 else RED_ALERT)
            pygame.draw.rect(self.screen, b_col, (bar_x, y_cur + 2, int(0.6 * u.battery), 10))
            y_cur += 17

        y_cur += 6
        rescued_cnt = sum(1 for s in self.survivors if s.rescued)
        for rv in self.rvs:
            ep_status = "EPISTEMIC-OK" if rv.assigned_survivor_id is not None else "WAITING K_i"
            rv_line = f"RV{rv.id} [{rv.state[:12]:12s}] Load:{len(rv.passengers)}/{rv.capacity} -> S{rv.assigned_survivor_id} ({ep_status})"
            self.screen.blit(self.font_body.render(rv_line, True, PURPLE_RV), (x_pad, y_cur))
            y_cur += 16

        self.screen.blit(self.font_body.render(f"Total Survivors Evacuated: {rescued_cnt}/{len(self.survivors)}", True, GREEN_SAFE), (x_pad, y_cur))
        y_cur += 20

        # --- 4. Event Log ---
        pygame.draw.line(self.screen, PANEL_BORDER, (x_pad, y_cur), (panel_x + panel_w - 16, y_cur), 1)
        y_cur += 8
        self.screen.blit(self.font_hdr.render("4. MISSION EVENT LOG", True, TEXT_WHITE), (x_pad, y_cur))
        y_cur += 17
        for log_line in self.event_log[:5]:
            self.screen.blit(self.font_small.render(log_line, True, TEXT_MUTED), (x_pad, y_cur))
            y_cur += 14

        # --- 5. Controls Footer ---
        y_cur = MAP_OFFSET_Y + MAP_SIZE - 110
        pygame.draw.line(self.screen, PANEL_BORDER, (x_pad, y_cur), (panel_x + panel_w - 16, y_cur), 1)
        y_cur += 6
        self.screen.blit(self.font_hdr.render("INTERACTIVE PRESENTATION CONTROLS:", True, AMBER_WARN), (x_pad, y_cur))
        y_cur += 16
        ctrl_lines = [
            "[L-CLICK] Ignite Fire   [R-CLICK] Inspect UAV / Survivor",
            "[1 / 2 / 3] Speed: 0.5x (Slow-Mo) / 1.0x / 2.0x",
            "[K] Kill Random UAV     [W] Rotate Wind +45 deg",
            "[G] Toggle Graphs       [A] Toggle ADMM Vectors  [SPACE] Pause"
        ]
        for c_line in ctrl_lines:
            self.screen.blit(self.font_small.render(c_line, True, TEXT_WHITE), (x_pad, y_cur))
            y_cur += 15

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if MAP_OFFSET_X <= mx <= MAP_OFFSET_X + MAP_SIZE and MAP_OFFSET_Y <= my <= MAP_OFFSET_Y + MAP_SIZE:
                    gx, gy = self.screen_to_grid(mx, my)
                    if event.button == 1:
                        # Left-Click: Ignite Fire
                        self.env.ignite(int(gx), int(gy), radius=2)
                        self.log_event(f"INJECTION: Fire sparked at ({int(gx)}, {int(gy)})!")
                    elif event.button == 3:
                        # Right-Click: Select nearest UAV or Survivor for inspection
                        click_pt = np.array([gx, gy])
                        best_d = 999.0
                        for u in self.uavs:
                            d = np.linalg.norm(u.pos - click_pt)
                            if d < best_d:
                                best_d = d
                                self.selected_entity = ('UAV', u.id)
                        for s in self.survivors:
                            if not s.rescued:
                                d = np.linalg.norm(s.pos - click_pt)
                                if d < best_d:
                                    best_d = d
                                    self.selected_entity = ('SURV', s.id)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_1:
                    self.speed_multiplier = 0.5
                    self.log_event("TIME DILATION: 0.5x Slow-Motion Analysis Mode.")
                elif event.key == pygame.K_2:
                    self.speed_multiplier = 1.0
                    self.log_event("TIME DILATION: 1.0x Normal Real-Time Mode.")
                elif event.key == pygame.K_3:
                    self.speed_multiplier = 2.0
                    self.log_event("TIME DILATION: 2.0x Fast-Forward Mode.")
                elif event.key == pygame.K_g:
                    self.show_comms = not self.show_comms
                elif event.key == pygame.K_s:
                    self.show_sensors = not self.show_sensors
                elif event.key == pygame.K_a:
                    self.show_admm_lines = not self.show_admm_lines
                elif event.key == pygame.K_r:
                    self.reset_simulation()
                elif event.key == pygame.K_k:
                    if len(self.uavs) > 1:
                        lost = self.uavs.pop(0)
                        self.patrol_angles.pop(0)
                        self.log_event(f"FAULT INJECTED: UAV {lost.id} destroyed! Swarm adapting...")
                        self.A_u, self.D_u, self.L_u, self.lambda_2 = GraphManager.compute_uav_matrices(self.uavs)
                elif event.key == pygame.K_w:
                    angle = np.pi / 4.0
                    c, s = np.cos(angle), np.sin(angle)
                    self.env.wind = np.array([[c, -s], [s, c]]) @ self.env.wind
                    norm = np.linalg.norm(self.env.wind)
                    self.env.wind_dir = self.env.wind / (norm + 1e-6)
                    self.log_event(f"WEATHER SHIFT: Wind rotated to {self.env.wind.round(1)}")

    def run(self):
        while True:
            self.handle_events()
            if not self.paused:
                self.frame_counter += 1
                self.step_physics()

            self.screen.fill(BG_DARK)
            self.draw_map()
            self.draw_telemetry_sidebar()

            pygame.display.flip()
            self.clock.tick(30)


if __name__ == "__main__":
    app = TacticalCommandCenterV2()
    app.run()
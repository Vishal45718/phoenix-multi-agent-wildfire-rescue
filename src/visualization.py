import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .environment import ForestEnvironment
from .agents import SurvivorAgent, UAVAgent, RescueVehicle
from .safety import ActiveSTLController, EpistemicLogicGuard
from .coordination import GraphManager, ADMMConsensusOptimizer, RescueDispatchManager

def generate_dashboard(mc_results, save_path="phase4_executive_dashboard.png"):
    """
    Generates a 4-panel executive summary dashboard illustrating:
    1. The physical closed-loop tracking and dispatch
    2. STL Robustness margins over time
    3. Multi-agent ADMM consensus convergence
    4. Monte Carlo benchmark comparisons
    """
    np.random.seed(42)
    env = ForestEnvironment(width=100, height=100, wind_vector=(2.8, 1.8))
    env.ignite(30, 30, radius=3)

    survivors = [
        SurvivorAgent(0, [34.0, 33.0], voice_radius=10.0, mobility=0.25, goal=[85.0, 85.0]),
        SurvivorAgent(1, [36.0, 34.0], voice_radius=10.0, mobility=0.35, goal=[85.0, 85.0]),
        SurvivorAgent(2, [44.0, 40.0], voice_radius=10.0, mobility=0.70, goal=[85.0, 85.0]),
        SurvivorAgent(3, [52.0, 48.0], voice_radius=10.0, mobility=0.85, goal=[85.0, 85.0])
    ]

    uavs = [
        UAVAgent(0, [24.0, 48.0], [10.0, 90.0], v_max=5.5, comm_radius=35.0, battery=100.0),
        UAVAgent(1, [27.0, 46.0], [10.0, 90.0], v_max=5.2, comm_radius=35.0, battery=100.0),
        UAVAgent(2, [38.0, 54.0], [10.0, 90.0], v_max=5.0, comm_radius=32.0, battery=100.0),
        UAVAgent(3, [58.0, 60.0], [10.0, 90.0], v_max=5.0, comm_radius=32.0, battery=100.0)
    ]
    
    rvs = [
        RescueVehicle(0, [20.0, 22.0], max_speed=4.8),
        RescueVehicle(1, [65.0, 62.0], max_speed=4.8)
    ]

    stl_ctrl = ActiveSTLController(d_safe=6.0, d_fire=10.0, h_min=25.0, buffer_zone=7.0, k_repel=9.0)
    default_cmds = [np.array([1.8, -1.5]), np.array([1.6, -1.4]), np.array([1.0, -1.5]), np.array([-1.2, -1.2])]

    rho_spacing_hist = []
    rho_fire_hist = []
    admm_res_curve = None

    # Run demonstration simulation
    for step in range(1, 26):
        t_now = float(step)
        env.step()
        fc = env.get_fire_centroid()
        
        for s in survivors:
            if not getattr(s, 'rescued', False) and s.health > 0.0:
                s.step(fc, env, dt=1.0)

        active_s = [s for s in survivors if not getattr(s, 'rescued', False) and s.health > 0.0]
        for u in uavs:
            u.sense_survivors(active_s, env, t_now)

        A, D, L, l2 = GraphManager.compute_uav_matrices(uavs)
        GraphManager.synchronize_and_track_gradients(uavs, A, t_now)
        GraphManager.handle_tracking_handoff(uavs)

        for idx, u in enumerate(uavs):
            has_k = any(EpistemicLogicGuard.satisfies_knowledge_check(u, s.id, current_time=t_now) for s in active_s)
            u_nom = u.update_behavior(t_now, default_cmds[idx], has_knowledge=has_k)
            u_cmd, _ = stl_ctrl.apply_safety_shield(idx, u_nom, uavs, env)
            u.step(u_cmd, env, dt=1.0)

        margins = stl_ctrl.evaluate_safety_margins(uavs, env, survivors)
        rho_spacing_hist.append(margins['uav_spacing'])
        rho_fire_hist.append(margins['fire_clearance'])

        if len(active_s) > 0:
            allocs, res_hist, _, _ = ADMMConsensusOptimizer.run_assignment(uavs, rvs, active_s, max_iters=25, rho=2.0, return_history=True)
            if (admm_res_curve is None or max(res_hist) > max(admm_res_curve)) and len(res_hist) > 0:
                admm_res_curve = res_hist
            RescueDispatchManager.step_rescue_vehicles(rvs, survivors, uavs, allocs, env, t_now, dt=1.0, pickup_radius=4.5)

    # Initialize plotting canvas
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ========================================================
    # Panel 1: Physical Reality & Aerial/Ground Coordination
    # ========================================================
    ax1 = axes[0, 0]
    cmap = ListedColormap(['#2e7d32', '#d84315', '#424242'])
    ax1.imshow(env.state, cmap=cmap, origin='lower', extent=[0, 100, 0, 100], alpha=0.80)
    for idx, u in enumerate(uavs):
        p_hist = np.array(u.path_history)
        ax1.plot(p_hist[:, 0], p_hist[:, 1], '--', linewidth=1.6, label=f'UAV {u.id} Path')
        ax1.scatter(u.pos[0], u.pos[1], marker='^', s=130, edgecolor='white', zorder=5)
    for rv in rvs:
        rv_hist = np.array(rv.path_history)
        ax1.plot(rv_hist[:, 0], rv_hist[:, 1], '-', linewidth=2.8, color='cyan', label=f'Rescue Truck {rv.id}')
        ax1.scatter(rv.pos[0], rv.pos[1], marker='s', s=160, color='cyan', edgecolor='black', zorder=6)
    for s in survivors:
        col = '#00e676' if getattr(s, 'rescued', False) else 'yellow'
        status = 'RESCUED' if getattr(s, 'rescued', False) else 'ACTIVE'
        ax1.scatter(s.pos[0], s.pos[1], c=col, s=130, edgecolor='black', zorder=7)
        ax1.text(s.pos[0]+1.2, s.pos[1]+1.2, f'S{s.id} ({status})', color='white', fontweight='bold', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
    ax1.set_title("1. Closed-Loop Aerial Tracking & Ground Rescue Dispatch", fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 100); ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.3)

    # ========================================================
    # Panel 2: STL Robustness Curves
    # ========================================================
    ax2 = axes[0, 1]
    steps_x = np.arange(1, len(rho_spacing_hist) + 1)
    ax2.plot(steps_x, rho_spacing_hist, '-o', color='#1e88e5', linewidth=2, label='Inter-UAV Margin (rho_spacing)')
    ax2.plot(steps_x, rho_fire_hist, '-s', color='#fb8c00', linewidth=2, label='Fire Clearance Margin (rho_fire)')
    ax2.axhline(0.0, color='#e53935', linestyle='--', linewidth=2, label='STL Safety Boundary (rho = 0)')
    ax2.fill_between(steps_x, -10, 0, color='#e53935', alpha=0.15)
    ax2.set_title("2. Closed-Loop STL Safety Robustness Margins", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Simulation Step (s)"); ax2.set_ylabel("Robustness Margin (m)")
    ax2.set_ylim(-5, max(max(rho_spacing_hist), max(rho_fire_hist)) + 5)
    ax2.legend(loc='upper right'); ax2.grid(True, linestyle=':', alpha=0.4)

    # ========================================================
    # Panel 3: ADMM Convergence
    # ========================================================
    ax3 = axes[1, 0]
    rvs_eq = [RescueVehicle(0, [20.0, 50.0]), RescueVehicle(1, [80.0, 50.0])]
    surv_eq = [SurvivorAgent(0, [50.0, 50.0]), SurvivorAgent(1, [48.0, 52.0])]
    u0 = UAVAgent(0, [30.0, 50.0], [0.0, 0.0])
    u0.belief_map = {0: {'pos': np.array([47.0, 50.0]), 'conf': 0.9, 'health': 60.0, 'timestamp': 0.0}, 1: {'pos': np.array([53.0, 52.0]), 'conf': 0.9, 'health': 75.0, 'timestamp': 0.0}}
    u1 = UAVAgent(1, [70.0, 50.0], [0.0, 0.0])
    u1.belief_map = {0: {'pos': np.array([52.5, 50.0]), 'conf': 0.9, 'health': 60.0, 'timestamp': 0.0}, 1: {'pos': np.array([46.5, 52.0]), 'conf': 0.9, 'health': 75.0, 'timestamp': 0.0}}
    
    _, admm_res_curve_eq, _, _ = ADMMConsensusOptimizer.run_assignment([u0, u1], rvs_eq, surv_eq, max_iters=25, rho=1.5, return_history=True)
    if admm_res_curve_eq is not None:
        iters_x = np.arange(1, len(admm_res_curve_eq) + 1)
        ax3.semilogy(iters_x, np.maximum(admm_res_curve_eq, 1e-8), '-d', color='#8e24aa', linewidth=2.2)
    ax3.set_title("3. Multi-Agent ADMM Consensus Convergence (Sum ||X_i - Z||_F)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("ADMM Iteration k"); ax3.set_ylabel("Primal Residual (Log Scale)")
    ax3.grid(True, which='both', linestyle=':', alpha=0.4)

    # ========================================================
    # Panel 4: Monte Carlo Results
    # ========================================================
    ax4 = axes[1, 1]
    labels = ["Random Search", "Greedy Uncoord.", "FIRE-SAR (Ours)"]
    keys = ["Random_Search", "Greedy_Uncoordinated", "FIRE_SAR_Proposed"]
    surv_means = [np.mean(mc_results[k]['survival_rate']) * 100 for k in keys]
    resc_means = [np.mean(mc_results[k]['rescued_ratio']) * 100 for k in keys]
    viol_means = [np.mean(mc_results[k]['stl_violations']) for k in keys]

    x_pos = np.arange(len(labels))
    width = 0.26
    ax4.bar(x_pos - width, surv_means, width, label='Survival Rate (%)', color='#43a047')
    ax4.bar(x_pos, resc_means, width, label='Rescued Ratio (%)', color='#1e88e5')
    ax4.bar(x_pos + width, viol_means, width, label='STL Violations (Count)', color='#e53935')
    ax4.set_xticks(x_pos); ax4.set_xticklabels(labels, fontweight='bold')
    ax4.set_title("4. Monte Carlo Benchmark Comparison (12 Seeds, 15% Packet Loss)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Metric Value"); ax4.legend(); ax4.grid(True, axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"SUCCESS: Saved 4-panel executive summary figure to {save_path}")

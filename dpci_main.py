import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from collections import deque
import time
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import csv
import pandas as pd
import datetime
import matplotlib.animation as animation
import airsim
import threading
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize

warnings.filterwarnings('ignore')



pos_offsestx=[0,3,-3,3,-30]
pos_offsesty=[0,3,0,-3,0]
index={
    "PX4":0,
    "our_uav_1":1,
    "our_uav_2":2,
    "our_uav_3":3,
    "EnemyUAV":4


}

class ParallelUAVController:
    def __init__(self, client):
        self.client = client
        self.client_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="UAV")

    def send_velocity_command(self, uav_name, vx, vy, vz, duration):
        with self.client_lock:
            self.client.moveByVelocityAsync(vx, vy, vz, duration, vehicle_name=uav_name)

    def send_all_commands_parallel(self, uav_commands, duration=0.05):
        futures = []
        for uav_name, vx, vy, vz in uav_commands:
            future = self.executor.submit(
                self.send_velocity_command, uav_name, vx, vy, vz, duration
            )
            futures.append(future)

        for future in futures:
            future.result()

class UnifiedCenterLineLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=1024, num_layers=1, output_dim=6, dropout=0.1):
        super(UnifiedCenterLineLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        final_hidden = h_n[-1]
        return self.regressor(final_hidden)


class UnifiedCenterLinePredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.prediction_history = deque(maxlen=10)

        self.baseline_centerline = None
        self.baseline_established_time = None
        self.baseline_update_count = 0

        self.consecutive_stable_predictions = 0
        self.min_stable_predictions = 3
        self.recent_predictions = deque(maxlen=5)

        self.baseline_candidate = None
        self.candidate_stable_count = 0

    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = UnifiedCenterLineLSTM(
                input_dim=3,
                hidden_dim=1024,
                num_layers=1,
                output_dim=6,
                dropout=0.1
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            print(f"Unified centerline LSTM model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load centerline model: {e}")
            return None

    def normalize_trajectory(self, trajectory):
        trajectory_array = np.array(trajectory)
        if len(trajectory_array) == 0:
            return trajectory_array, np.zeros(3)

        first_frame = trajectory_array[0:1].copy()
        normalized_trajectory = trajectory_array - first_frame
        return normalized_trajectory, first_frame.flatten()

    def denormalize_centerline(self, normalized_centerline, first_frame_offset):
        centerline = normalized_centerline.copy()
        centerline[:3] += first_frame_offset
        return centerline

    def predict_centerline(self, trajectory):
        if self.model is None or len(trajectory) < 20:
            print(f"DEBUG: Cannot predict centerline - model={self.model is not None}, traj_len={len(trajectory)}")
            return None

        try:
            normalized_traj, first_frame_offset = self.normalize_trajectory(trajectory)
            print(
                f"DEBUG: Normalized trajectory shape: {normalized_traj.shape}, first_frame_offset: {first_frame_offset}")

            traj_tensor = torch.FloatTensor(normalized_traj).unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction = self.model(traj_tensor)
                normalized_centerline = prediction[0].cpu().numpy()

            centerline = self.denormalize_centerline(normalized_centerline, first_frame_offset)
            print(f"DEBUG: Raw predicted centerline: start={centerline[:3]}, vector={centerline[3:]}")

            self.prediction_history.append(centerline.copy())
            self.recent_predictions.append(centerline.copy())

            return centerline

        except Exception as e:
            print(f"DEBUG: Centerline prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_centerline_deviation(self, new_centerline, reference_centerline):
        #计算误差
        if reference_centerline is None:
            return float('inf')

        start_pos_new = new_centerline[:3]
        start_pos_ref = reference_centerline[:3]
        position_deviation = np.linalg.norm(start_pos_new - start_pos_ref)

        direction_new = new_centerline[3:]
        direction_ref = reference_centerline[3:]

        direction_new_norm = direction_new / (np.linalg.norm(direction_new) + 1e-8)
        direction_ref_norm = direction_ref / (np.linalg.norm(direction_ref) + 1e-8)

        cos_angle = np.clip(np.dot(direction_new_norm, direction_ref_norm), -1.0, 1.0)
        angle_diff = np.arccos(np.abs(cos_angle))

        direction_equivalent_distance = angle_diff * 10.0
        total_deviation = position_deviation + direction_equivalent_distance

        print(f"DEBUG: Centerline deviation analysis:")
        print(f"  Position deviation: {position_deviation:.3f}m")
        print(f"  Angle difference: {angle_diff:.3f} rad ({np.degrees(angle_diff):.1f} deg)")
        print(f"  Direction equivalent distance: {direction_equivalent_distance:.3f}m")
        print(f"  Total deviation: {total_deviation:.3f}")

        return total_deviation

    def check_prediction_stability(self, new_centerline, threshold=1.5):
        if len(self.recent_predictions) < 2:
            return False

        recent_deviations = []
        for prev_prediction in list(self.recent_predictions)[:-1]:
            deviation = self.calculate_centerline_deviation(new_centerline, prev_prediction)
            recent_deviations.append(deviation)

        avg_recent_deviation = np.mean(recent_deviations)
        max_recent_deviation = np.max(recent_deviations)

        is_stable = (avg_recent_deviation < threshold and max_recent_deviation < threshold * 1.5)

        print(f"DEBUG: Prediction stability check:")
        print(f"  Recent deviations: {[f'{d:.2f}' for d in recent_deviations]}")
        print(f"  Average recent deviation: {avg_recent_deviation:.3f}")
        print(f"  Max recent deviation: {max_recent_deviation:.3f}")
        print(f"  Stability threshold: {threshold}")
        print(f"  Is stable: {is_stable}")

        return is_stable

    def is_centerline_changed(self, new_centerline, threshold=2.0):
        is_stable = self.check_prediction_stability(new_centerline, threshold=threshold * 0.75)

        if self.baseline_centerline is None:
            if is_stable:
                print("DEBUG: No baseline centerline, and prediction is stable -> establishing baseline")
                return True, float('inf')
            else:
                print("DEBUG: No baseline centerline, prediction not stable yet -> waiting")
                return False, float('inf')

        deviation_from_baseline = self.calculate_centerline_deviation(new_centerline, self.baseline_centerline)
        baseline_changed = deviation_from_baseline > threshold

        print(f"DEBUG: Centerline change analysis:")
        print(f"  Deviation from baseline: {deviation_from_baseline:.3f}")
        print(f"  Change threshold: {threshold}")
        print(f"  Prediction is stable: {is_stable}")
        print(f"  Baseline would change: {baseline_changed}")

        if baseline_changed:
            if is_stable:
                self.consecutive_stable_predictions += 1
                print(f"  Consecutive stable predictions: {self.consecutive_stable_predictions}")

                if self.consecutive_stable_predictions >= self.min_stable_predictions:
                    print("  -> Sufficient stable predictions, updating baseline")
                    self.consecutive_stable_predictions = 0
                    return True, deviation_from_baseline
                else:
                    print("  -> Not enough consecutive stable predictions yet")
                    return False, deviation_from_baseline
            else:
                self.consecutive_stable_predictions = 0
                print("  -> Large deviation but prediction unstable, waiting")
                return False, deviation_from_baseline
        else:
            self.consecutive_stable_predictions = 0
            print("  -> Small deviation, keeping current baseline")
            return False, deviation_from_baseline

    def update_baseline_centerline(self, new_centerline):
        old_baseline = self.baseline_centerline.copy() if self.baseline_centerline is not None else None

        self.baseline_centerline = new_centerline.copy()
        self.baseline_established_time = time.time()
        self.baseline_update_count += 1
        self.consecutive_stable_predictions = 0

        print(f"DEBUG: *** BASELINE CENTERLINE UPDATED (#{self.baseline_update_count}) ***")
        print(f"  New baseline start: {self.baseline_centerline[:3]}")
        print(f"  New baseline direction: {self.baseline_centerline[3:]}")
        print(f"  Update time: {time.strftime('%H:%M:%S', time.localtime(self.baseline_established_time))}")

        if old_baseline is not None:
            change_magnitude = self.calculate_centerline_deviation(new_centerline, old_baseline)
            print(f"  Change magnitude from previous baseline: {change_magnitude:.3f}")

    def get_current_centerline(self):
        return self.baseline_centerline.copy() if self.baseline_centerline is not None else None

    def get_baseline_info(self):
        if self.baseline_centerline is None:
            return None

        return {
            'centerline': self.baseline_centerline.copy(),
            'established_time': self.baseline_established_time,
            'update_count': self.baseline_update_count,
            'age_seconds': time.time() - self.baseline_established_time if self.baseline_established_time else 0,
            'consecutive_stable_count': self.consecutive_stable_predictions,
            'recent_predictions_count': len(self.recent_predictions)
        }


class SimulatedEnemyUAV:
    def __init__(self, initial_position=[-30.0, 0.0, -15.0], motion_mode='helix'):
        self.initial_position = np.array(initial_position)
        self.position = self.initial_position.copy()
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.motion_mode = motion_mode
        self.current_time = 0.0
        self.setup_pid_controller()
        self.setup_motion()

    def setup_motion(self):
        if self.motion_mode == 'helix':
            self.setup_helix_motion()
        elif self.motion_mode == 'zigzag':
            self.setup_zigzag_motion()
        elif self.motion_mode == 'linear':
            self.setup_linear_motion()
        elif self.motion_mode == 'linear_turn_linear':
            self.setup_linear_turn_linear_motion()
        elif self.motion_mode == 'polyline':
            self.setup_polyline_motion()
        elif self.motion_mode == 'evasive':
            self.setup_evasive_motion()
        else:
            self.setup_helix_motion()

    def setup_evasive_motion(self):
        """Setup evasive motion parameters"""
        print("Setting up evasive motion parameters...")
        # Default parameters - will be overridden by update_motion_parameters
        self.evasive_speed = 3.0
        self.alpha = 0.1  # Mixing factor: escape vs perturbation

        # Perturbation coefficients for x-component
        self.perturbation_x_coeffs = [0.8, 0.6, 0.4]
        self.perturbation_x_freqs = [1.3, 0.7, 2.1]
        self.perturbation_x_funcs = ['sin', 'cos', 'sin']  # Function types

        # Perturbation coefficients for y-component
        self.perturbation_y_coeffs = [0.7, 0.5, 0.4]
        self.perturbation_y_freqs = [1.1, 0.9, 1.9]
        self.perturbation_y_funcs = ['cos', 'sin', 'cos']

        # Perturbation coefficients for z-component
        self.perturbation_z_coeffs = [0.6, 0.4, 0.3]
        self.perturbation_z_freqs = [0.8, 1.2, 1.6]
        self.perturbation_z_funcs = ['sin', 'cos', 'sin']

        # Initial escape direction (fallback when no our_uavs)
        self.init_escape_direction = np.array([-1.0, 0.0, 0.0])

    def setup_helix_motion(self):
        print("Setting up helix motion parameters...")
        self.helix_radius = 1.0
        self.helix_angular_freq = 2.0
        self.helix_pitch = 1.5
        self.helix_initial_phase = 0.0
        self.centerline_velocity = np.array([-2.2, 0.3, -0.1])
        self.radius_variation_amp = 0.8
        self.radius_variation_freq = 0.05
        self.phase_drift_rate = 0.02
        self.velocity_smoothing_alpha = 0.85
        self.smooth_velocity = np.array([0.0, 0.0, 0.0])

    def setup_zigzag_motion(self):
        print("Setting up zigzag motion parameters...")
        self.forward_speed = 2.0
        self.zigzag_amplitude = 4.0
        self.zigzag_frequency = 0.5
        self.z_drift_amplitude = 0.5
        self.z_drift_frequency = 0.1

    def setup_linear_motion(self):
        print("Setting up linear motion parameters...")
        self.linear_speed = 2.5
        self.linear_direction = np.array([-1.0, 0, 0])
        self.linear_direction = self.linear_direction / np.linalg.norm(self.linear_direction)

    def setup_linear_turn_linear_motion(self):
        print("Setting up linear-turn-linear motion parameters...")
        self.linear_speed = 2.0
        self.initial_direction = np.array([-1.0, 0.1, 0.0])
        self.initial_direction = self.initial_direction / np.linalg.norm(self.initial_direction)
        self.detour_radius = 3.0
        self.phase_durations = [3.0, 2.0, 3.0]
        self.total_duration = sum(self.phase_durations)

    def setup_polyline_motion(self):
        print("Setting up polyline motion parameters...")
        self.base_speed = 2.2
        self.num_segments = 4
        self.segment_durations = [2.5, 2.0, 2.5, 3.0]
        base_direction = np.array([-1.0, 0.0, 0.0])
        self.segment_directions = []
        self.turn_points = []

        current_direction = base_direction.copy()
        self.segment_directions.append(current_direction.copy())

        for i in range(1, self.num_segments):
            while True:
                turn_angle = np.random.uniform(-np.pi / 2, np.pi / 2)
                cos_angle = np.cos(turn_angle)
                sin_angle = np.sin(turn_angle)

                new_direction = np.array([
                    current_direction[0] * cos_angle - current_direction[1] * sin_angle,
                    current_direction[0] * sin_angle + current_direction[1] * cos_angle,
                    current_direction[2] + np.random.uniform(-0.05, 0.05)
                ])
                new_direction = new_direction / np.linalg.norm(new_direction)

                if new_direction[0] < 0.2:
                    current_direction = new_direction
                    break

            self.segment_directions.append(current_direction.copy())

    def compute_helix_motion(self, t):
        centerline_pos = self.initial_position + self.centerline_velocity * t
        radius_variation = self.radius_variation_amp * np.sin(2 * np.pi * self.radius_variation_freq * t)
        effective_radius = self.helix_radius + radius_variation
        base_phase = self.helix_angular_freq * t + self.helix_initial_phase
        phase_drift = self.phase_drift_rate * t * np.sin(0.1 * t)
        effective_phase = base_phase + phase_drift

        helix_offset_y = effective_radius * np.cos(effective_phase)
        helix_offset_z = effective_radius * np.sin(effective_phase)
        helix_offset_x = 0.5 * np.sin(effective_phase * 0.5)

        helix_offset = np.array([helix_offset_x, helix_offset_y, helix_offset_z])
        final_position = centerline_pos + helix_offset
        return final_position

    def compute_zigzag_motion(self, t):
        x = self.initial_position[0] - self.forward_speed * t
        y = self.initial_position[1] + self.zigzag_amplitude * np.sin(2 * np.pi * self.zigzag_frequency * t)
        z = self.initial_position[2] + self.z_drift_amplitude * np.sin(2 * np.pi * self.z_drift_frequency * t)
        return np.array([x, y, z])

    def compute_linear_motion(self, t):
        displacement = self.linear_speed * t
        position = self.initial_position + displacement * self.linear_direction
        return position

    def compute_linear_turn_linear_motion(self, t):
        current_pos = self.initial_position.copy()

        if t <= self.phase_durations[0]:
            displacement = self.linear_speed * t
            position = current_pos + displacement * self.initial_direction
        elif t <= self.phase_durations[0] + self.phase_durations[1]:
            t_phase = t - self.phase_durations[0]
            phase_progress = t_phase / self.phase_durations[1]

            start_pos = current_pos + self.linear_speed * self.phase_durations[0] * self.initial_direction
            side_direction = np.array([self.initial_direction[1], -self.initial_direction[0], 0])
            if np.linalg.norm(side_direction) < 0.1:
                side_direction = np.array([0, 1, 0])
            side_direction = side_direction / np.linalg.norm(side_direction)

            A = start_pos
            B = A + 2 * self.detour_radius * self.initial_direction

            angle = np.pi * phase_progress
            circle_center = (A + B) / 2 + self.detour_radius * side_direction

            offset_vector = self.detour_radius * (
                    -side_direction * np.cos(angle) +
                    self.initial_direction * np.sin(angle)
            )
            position = circle_center + offset_vector

        else:
            t_phase = t - self.phase_durations[0] - self.phase_durations[1]
            start_pos = self.initial_position + self.linear_speed * self.phase_durations[0] * self.initial_direction
            B_pos = start_pos + 2 * self.detour_radius * self.initial_direction
            displacement = self.linear_speed * t_phase
            position = B_pos + displacement * self.initial_direction

        return position

    def compute_polyline_motion(self, t):
        current_time = 0
        current_pos = self.initial_position.copy()

        for i, (duration, direction) in enumerate(zip(self.segment_durations, self.segment_directions)):
            if t <= current_time + duration:
                segment_time = t - current_time
                displacement = self.base_speed * segment_time
                position = current_pos + displacement * direction
                return position
            else:
                displacement = self.base_speed * duration
                current_pos += displacement * direction
                current_time += duration

        extra_time = t - current_time
        displacement = self.base_speed * extra_time
        position = current_pos + displacement * self.segment_directions[-1]
        return position

    def compute_evasive_motion(self, t):
        """Evasive motion: escape + perturbation"""

        if self.our_uavs is None:
            # No our UAV info - use pure perturbation motion
            escape_direction = self.init_escape_direction
        else:
            # Calculate escape direction away from our UAVs
            our_positions = [uav.position for uav in self.our_uavs]
            our_center = np.mean(our_positions, axis=0)
            to_us_vector = our_center - self.position
            distance_to_us = np.linalg.norm(to_us_vector)

            if distance_to_us > 0:
                escape_direction = -to_us_vector / distance_to_us
            else:
                escape_direction = self.init_escape_direction

        # Calculate perturbation using parameterized sine/cosine functions
        time_factor = t * 0.1

        # X-component perturbation
        perturbation_x = 0.0
        for i in range(len(self.perturbation_x_coeffs)):
            coeff = self.perturbation_x_coeffs[i]
            freq = self.perturbation_x_freqs[i]
            func_type = self.perturbation_x_funcs[i]

            if func_type == 'sin':
                perturbation_x += coeff * np.sin(time_factor * freq)
            else:  # cos
                perturbation_x += coeff * np.cos(time_factor * freq)

        # Y-component perturbation
        perturbation_y = 0.0
        for i in range(len(self.perturbation_y_coeffs)):
            coeff = self.perturbation_y_coeffs[i]
            freq = self.perturbation_y_freqs[i]
            func_type = self.perturbation_y_funcs[i]

            if func_type == 'sin':
                perturbation_y += coeff * np.sin(time_factor * freq)
            else:  # cos
                perturbation_y += coeff * np.cos(time_factor * freq)

        # Z-component perturbation
        perturbation_z = 0.0
        for i in range(len(self.perturbation_z_coeffs)):
            coeff = self.perturbation_z_coeffs[i]
            freq = self.perturbation_z_freqs[i]
            func_type = self.perturbation_z_funcs[i]

            if func_type == 'sin':
                perturbation_z += coeff * np.sin(time_factor * freq)
            else:  # cos
                perturbation_z += coeff * np.cos(time_factor * freq)

        # Normalize perturbation vector
        perturbation_vector = np.array([perturbation_x, perturbation_y, perturbation_z])
        norm_perturbation = np.linalg.norm(perturbation_vector)
        if norm_perturbation > 0:
            perturbation_unit = perturbation_vector / norm_perturbation
        else:
            perturbation_unit = np.array([1.0, 0.0, 0.0])

        # Combine escape direction and perturbation
        direction_x = self.alpha * escape_direction[0] + (1 - self.alpha) * perturbation_unit[0]
        direction_y = self.alpha * escape_direction[1] + (1 - self.alpha) * perturbation_unit[1]
        direction_z = self.alpha * escape_direction[2] + (1 - self.alpha) * perturbation_unit[2]

        # Normalize final direction
        total_magnitude = np.sqrt(direction_x ** 2 + direction_y ** 2 + direction_z ** 2)
        if total_magnitude > 0:
            displacement = self.evasive_speed * 0.05 * np.array(
                [direction_x, direction_y, direction_z]) / total_magnitude
        else:
            displacement = np.array([self.evasive_speed * 0.05, 0, 0])

        return self.position + displacement

    def step(self, dt=0.05):
        print(f"DEBUG step(): dt={dt}")
        self.current_time += dt
        t = self.current_time

        if self.motion_mode == 'helix':
            target_position = self.compute_helix_motion(t)
        elif self.motion_mode == 'zigzag':
            target_position = self.compute_zigzag_motion(t)
        elif self.motion_mode == 'linear':
            target_position = self.compute_linear_motion(t)
        elif self.motion_mode == 'linear_turn_linear':
            target_position = self.compute_linear_turn_linear_motion(t)
        elif self.motion_mode == 'polyline':
            target_position = self.compute_polyline_motion(t)
        elif self.motion_mode == 'evasive':
            target_position = self.compute_evasive_motion(t)
        else:
            target_position = self.compute_helix_motion(t)

        current_position = self.position  # 由AirSim更新
        target_velocity = self.compute_pid_velocity(target_position, current_position, dt)

        if np.linalg.norm(target_velocity)==0:
            print(f"target=0,current_position={current_position},target_position={target_position},current_time={self.current_time}")
        if target_position[2]>-2:
            print(f"target=0*****,current_position={current_position},target_position={target_position},current_time={self.current_time}")
        return target_velocity

    def reset(self, new_initial_position=None, new_motion_mode=None):
        if new_initial_position is not None:
            self.initial_position = np.array(new_initial_position)
        if new_motion_mode is not None:
            self.motion_mode = new_motion_mode

        self.position = self.initial_position.copy()
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.smooth_velocity = np.array([0.0, 0.0, 0.0])
        self.current_time = 0.0
        self.setup_motion()
    def setup_pid_controller(self):
        """设置PID控制器参数"""
        # PID参数 - 调优后的稳定参数
        self.kp = np.array([2.0, 2.0, 3.0])  # 比例增益
        self.ki = np.array([0.1, 0.1, 0.2])  # 积分增益
        self.kd = np.array([1.0, 1.0, 1.5])  # 微分增益

        # PID状态
        self.reset_pid_state()

        # 安全限制
        self.max_velocity = 500.0  # 最大速度 m/s
        #self.max_acceleration = 3.0  # 最大加速度 m/s²

    def reset_pid_state(self):
        """重置PID状态"""
        self.error_integral = np.zeros(3)
        self.error_previous = np.zeros(3)

    def get_motion_info(self):
        info = {
            'motion_mode': self.motion_mode,
            'current_time': self.current_time,
            'position': self.position.copy(),
            'velocity': self.velocity.copy()
        }

        if self.motion_mode == 'helix':
            info['helix_info'] = {
                'centerline_position': self.initial_position + self.centerline_velocity * self.current_time,
                'centerline_velocity': self.centerline_velocity,
                'radius': self.helix_radius,
                'angular_frequency': self.helix_angular_freq,
                'current_phase': (self.helix_angular_freq * self.current_time + self.helix_initial_phase) % (2 * np.pi)
            }
        elif self.motion_mode == 'polyline':
            current_time = 0
            current_segment = 0
            for i, duration in enumerate(self.segment_durations):
                if self.current_time <= current_time + duration:
                    current_segment = i
                    segment_progress = (self.current_time - current_time) / duration
                    break
                current_time += duration
            else:
                current_segment = len(self.segment_durations) - 1
                segment_progress = 1.0

            info['polyline_info'] = {
                'current_segment': current_segment,
                'segment_progress': segment_progress,
                'total_segments': self.num_segments,
                'segment_directions': self.segment_directions,
                'segment_durations': self.segment_durations,
                'motion_type': 'sharp_corners'
            }

        return info

    def compute_pid_velocity(self, target_position, current_position, dt):
        """
        使用PID控制器计算目标速度

        Args:
            target_position: 目标位置
            current_position: 当前位置
            dt: 时间步长

        Returns:
            目标速度
        """
        # 计算位置误差
        error = target_position - current_position

        # 积分项
        self.error_integral += error * dt
        # 防止积分饱和
        self.error_integral = np.clip(self.error_integral, -10.0, 10.0)

        # 微分项
        error_derivative = (error - self.error_previous) / dt if dt > 0 else np.zeros(3)
        self.error_previous = error.copy()

        # PID计算
        velocity_command = (self.kp * error +
                            self.ki * self.error_integral +
                            self.kd * error_derivative)

        # 速度限制
        velocity_magnitude = np.linalg.norm(velocity_command)
        if velocity_magnitude > self.max_velocity:
            velocity_command = velocity_command / velocity_magnitude * self.max_velocity

        return velocity_command


    def update_motion_parameters(self, motion_mode, params):
        """Update motion parameters during runtime"""
        self.motion_mode = motion_mode

        if motion_mode == 'helix':
            self.helix_radius = params.get('helix_radius', 1.0)
            self.helix_angular_freq = params.get('helix_angular_freq', 2.0)
            centerline_speed = params.get('centerline_speed', 2.2)
            self.centerline_velocity = np.array([-centerline_speed, 0.3, -0.1])
        elif motion_mode == 'zigzag':
            self.forward_speed = params.get('forward_speed', 2.0)
            self.zigzag_amplitude = params.get('zigzag_amplitude', 4.0)
        elif motion_mode == 'linear':
            self.linear_speed = params.get('linear_speed', 2.5)
        elif motion_mode == 'evasive':
            # Update evasive motion parameters
            self.evasive_speed = params.get('evasive_speed', 3.0)
            self.alpha = params.get('alpha', 0.1)

            # X-component parameters
            self.perturbation_x_coeffs = params.get('perturbation_x_coeffs', [0.8, 0.6, 0.4])
            self.perturbation_x_freqs = params.get('perturbation_x_freqs', [1.3, 0.7, 2.1])
            self.perturbation_x_funcs = params.get('perturbation_x_funcs', ['sin', 'cos', 'sin'])

            # Y-component parameters
            self.perturbation_y_coeffs = params.get('perturbation_y_coeffs', [0.7, 0.5, 0.4])
            self.perturbation_y_freqs = params.get('perturbation_y_freqs', [1.1, 0.9, 1.9])
            self.perturbation_y_funcs = params.get('perturbation_y_funcs', ['cos', 'sin', 'cos'])

            # Z-component parameters
            self.perturbation_z_coeffs = params.get('perturbation_z_coeffs', [0.6, 0.4, 0.3])
            self.perturbation_z_freqs = params.get('perturbation_z_freqs', [0.8, 1.2, 1.6])
            self.perturbation_z_funcs = params.get('perturbation_z_funcs', ['sin', 'cos', 'sin'])

            # Initial escape direction
            escape_dir = params.get('init_escape_direction', [-1.0, 0.0, 0.0])
            self.init_escape_direction = np.array(escape_dir) / np.linalg.norm(escape_dir)
        self.setup_motion()


class SimulatedOurUAV:
    def __init__(self, initial_position=[0.0, 0.0, -10.0], uav_id=0):
        self.initial_position = np.array(initial_position)
        self.position = self.initial_position.copy()
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.max_speed = 8.0
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.current_time = 0.0
        self.uav_id = uav_id  # 0 for center, 1,2,3 for outer UAVs
        self.acceleration = np.array([0.0, 0.0, 0.0])
    def set_velocity_command(self, target_velocity):
        self.target_velocity = np.array(target_velocity)


    def step(self, dt=0.05):
        self.current_time += dt
        # self.velocity = self.target_velocity.copy()
        # self.position += self.velocity * dt
        return self.position.copy(), self.velocity.copy()

    def reset(self, new_initial_position=None):
        if new_initial_position is not None:
            self.initial_position = np.array(new_initial_position)

        self.position = self.initial_position.copy()
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.current_time = 0.0


class UnifiedTrajectoryPlanner:
    def __init__(self):
        pass

    def project_point_to_line(self, point, line_start, line_end):
        line_vec = line_end - line_start
        line_length = np.linalg.norm(line_vec)

        if line_length < 0.01:
            return line_start, 0.0

        line_unit = line_vec / line_length
        point_vec = point - line_start
        projection_length = np.dot(point_vec, line_unit)

        projection_length = np.clip(projection_length, 0, line_length)
        projected_point = line_start + projection_length * line_unit

        return projected_point, projection_length

    def plan_intercept_strategy(self, our_pos, enemy_pos, center_line):
        start_point = center_line[:3]
        direction_vector = center_line[3:]
        end_point = start_point + direction_vector

        print(f"DEBUG: Intercept planning - centerline start={start_point}, end={end_point}")

        enemy_projection, _ = self.project_point_to_line(enemy_pos, start_point, end_point)
        print(f"DEBUG: Enemy projection on centerline: {enemy_projection}")

        our_projection, _ = self.project_point_to_line(our_pos, start_point, end_point)
        print(f"DEBUG: Our projection on centerline: {our_projection}")

        distance_to_centerline = np.linalg.norm(our_pos - our_projection)
        distance_to_enemy_projection = np.linalg.norm(our_pos - enemy_projection)

        print(f"DEBUG: Distance to centerline: {distance_to_centerline:.2f}")
        print(f"DEBUG: Distance to enemy projection: {distance_to_enemy_projection:.2f}")

        centerline_threshold = 2.0
        if distance_to_centerline > centerline_threshold:
            target_position = our_projection
            phase = 'APPROACH_CENTERLINE'
            print(f"DEBUG: Phase APPROACH_CENTERLINE, target={target_position}")
        else:
            target_position = enemy_projection
            phase = 'MOVE_TO_PROJECTION'
            print(f"DEBUG: Phase MOVE_TO_PROJECTION, target={target_position}")

        return target_position, phase, distance_to_enemy_projection


class MDNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures, hidden_dim=256):
        super(MDNNetwork, self).__init__()
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.pi_net = nn.Linear(hidden_dim // 2, num_mixtures)
        self.mu_net = nn.Linear(hidden_dim // 2, num_mixtures * output_dim)
        self.sigma_net = nn.Linear(hidden_dim // 2, num_mixtures * output_dim)

    def forward(self, x):
        features = self.feature_net(x)
        pi = torch.softmax(self.pi_net(features), dim=1)
        mu = self.mu_net(features)
        sigma = torch.clamp(torch.exp(self.sigma_net(features)), min=0.01, max=10.0)
        return pi, mu, sigma


class MDNLoss:
    def __init__(self, num_mixtures, output_dim):
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim

    def __call__(self, pi, mu, sigma, target):
        batch_size = target.size(0)
        mu = mu.view(batch_size, self.num_mixtures, self.output_dim)
        sigma = sigma.view(batch_size, self.num_mixtures, self.output_dim)

        target_expanded = target.unsqueeze(1).expand_as(mu)
        normal_dist = D.Normal(mu, sigma)
        log_probs = normal_dist.log_prob(target_expanded).sum(dim=2)
        weighted_log_probs = torch.log(pi + 1e-8) + log_probs
        log_sum = torch.logsumexp(weighted_log_probs, dim=1)
        nll = -log_sum.mean()
        return nll


class OnlineMDNTrainer:
    def __init__(self, input_dim, output_dim, num_mixtures=3, device='cpu', replay_buffer_size=1000):
        self.device = device
        self.model = MDNNetwork(input_dim, output_dim, num_mixtures).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = MDNLoss(num_mixtures, output_dim)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

        self.replay_buffer_X = deque(maxlen=replay_buffer_size)
        self.replay_buffer_y = deque(maxlen=replay_buffer_size)
        self.replay_ratio = 0.7

        # Statistics for sigma threshold calculation
        self.sigma_history = deque(maxlen=500)
        self.sigma_mean = None
        self.sigma_std = None

    def update_model(self, X, y):
        if len(X) < 2:
            return

        X_array = np.array(X)
        y_array = np.array(y)

        if np.any(np.isnan(X_array)) or np.any(np.isnan(y_array)):
            print("WARNING: NaN values detected in training data, skipping update")
            return

        if np.any(np.isinf(X_array)) or np.any(np.isinf(y_array)):
            print("WARNING: Infinite values detected in training data, skipping update")
            return

        y_magnitude = np.linalg.norm(y_array, axis=1)
        max_reasonable_delta = 50.0
        if np.any(y_magnitude > max_reasonable_delta):
            print(f"WARNING: Extreme delta values detected (max: {np.max(y_magnitude):.2f}m), skipping update")
            return

        for i in range(len(X_array)):
            self.replay_buffer_X.append(X_array[i].copy())
            self.replay_buffer_y.append(y_array[i].copy())

        if len(self.replay_buffer_X) >= 50:
            new_data_size = len(X_array)
            replay_data_size = int(new_data_size * self.replay_ratio / (1 - self.replay_ratio))
            replay_data_size = min(replay_data_size, len(self.replay_buffer_X) - new_data_size)

            if replay_data_size > 0:
                available_indices = list(range(len(self.replay_buffer_X) - new_data_size))
                if len(available_indices) >= replay_data_size:
                    replay_indices = np.random.choice(available_indices,
                                                      size=replay_data_size,
                                                      replace=False)

                    replay_X = [self.replay_buffer_X[i] for i in replay_indices]
                    replay_y = [self.replay_buffer_y[i] for i in replay_indices]

                    combined_X = np.vstack([X_array, np.array(replay_X)])
                    combined_y = np.vstack([y_array, np.array(replay_y)])
                else:
                    combined_X = X_array
                    combined_y = y_array
            else:
                combined_X = X_array
                combined_y = y_array
        else:
            combined_X = X_array
            combined_y = y_array

        try:
            if not self.is_fitted:
                X_scaled = self.scaler_X.fit_transform(combined_X)
                y_scaled = self.scaler_y.fit_transform(combined_y)
                self.is_fitted = True
            else:
                X_scaled = self.scaler_X.transform(combined_X)
                y_scaled = self.scaler_y.transform(combined_y)

            if np.any(np.isnan(X_scaled)) or np.any(np.isnan(y_scaled)):
                print("WARNING: NaN values after scaling, skipping update")
                return

            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.FloatTensor(y_scaled).to(self.device)

            self.model.train()
            for epoch in range(3):
                self.optimizer.zero_grad()
                pi, mu, sigma = self.model(X_tensor)

                if torch.any(torch.isnan(pi)) or torch.any(torch.isnan(mu)) or torch.any(torch.isnan(sigma)):
                    print(f"WARNING: NaN in model output at epoch {epoch}, stopping training")
                    break

                loss = self.loss_fn(pi, mu, sigma, y_tensor)

                if torch.isnan(loss):
                    print(f"WARNING: NaN loss at epoch {epoch}, stopping training")
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                has_nan_grad = False
                for param in self.model.parameters():
                    if param.grad is not None and torch.any(torch.isnan(param.grad)):
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    print(f"WARNING: NaN gradients at epoch {epoch}, stopping training")
                    break

                self.optimizer.step()

        except Exception as e:
            print(f"ERROR in model update: {e}")
            self.model = MDNNetwork(self.model.feature_net[0].in_features, 3, self.model.num_mixtures).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def predict(self, X, current_position):
        if not self.is_fitted:
            print("DEBUG: MDN not fitted yet")
            return None, None

        X_array = np.array(X).reshape(1, -1)

        if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
            print("WARNING: Invalid input data for prediction")
            return None, None

        try:
            X_scaled = self.scaler_X.transform(X_array)

            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                print("WARNING: Invalid scaled input data")
                return None, None

            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            self.model.eval()
            with torch.no_grad():
                pi, mu, sigma = self.model(X_tensor)

                if torch.any(torch.isnan(pi)) or torch.any(torch.isnan(mu)) or torch.any(torch.isnan(sigma)):
                    print("WARNING: Model output contains NaN values")
                    return None, None

                batch_size = X_tensor.size(0)
                mu = mu.view(batch_size, self.model.num_mixtures, 3)
                sigma = sigma.view(batch_size, self.model.num_mixtures, 3)

                confidences = pi[0].cpu().numpy()

                if np.any(np.isnan(confidences)) or np.any(np.isinf(confidences)):
                    print("WARNING: Invalid confidence values")
                    return None, None

                print(f"\n=== MDN Multi-Modal Predictions ===")
                print(f"Current Enemy Position: {current_position}")

                all_predictions = []
                valid_predictions = []

                for i in range(self.model.num_mixtures):
                    mu_i = mu[0, i].cpu().numpy()
                    sigma_i = sigma[0, i].cpu().numpy()
                    confidence = confidences[i]

                    if np.any(np.isnan(mu_i)) or np.any(np.isnan(sigma_i)) or np.isnan(confidence):
                        print(f"Mode {i + 1}: INVALID (contains NaN)")
                        continue

                    try:
                        delta_scaled = mu_i.reshape(1, -1)
                        delta = self.scaler_y.inverse_transform(delta_scaled)[0]

                        delta_magnitude = np.linalg.norm(delta)
                        if delta_magnitude > 100.0:
                            print(f"Mode {i + 1}: INVALID (delta too large: {delta_magnitude:.2f}m)")
                            continue

                        predicted_position = current_position + delta
                        uncertainty = np.mean(sigma_i)

                        # Store sigma for threshold calculation
                        self.sigma_history.append(uncertainty)

                        prediction = {
                            'mode': i,
                            'confidence': confidence,
                            'delta': delta,
                            'predicted_position': predicted_position,
                            'uncertainty': uncertainty
                        }

                        all_predictions.append(prediction)
                        valid_predictions.append(prediction)

                        print(f"Mode {i + 1}:")
                        print(f"  Confidence: {confidence:.4f} ({confidence * 100:.1f}%)")
                        print(f"  Delta: [{delta[0]:+.2f}, {delta[1]:+.2f}, {delta[2]:+.2f}]")
                        print(
                            f"  Predicted Position: [{predicted_position[0]:.2f}, {predicted_position[1]:.2f}, {predicted_position[2]:.2f}]")
                        print(f"  Uncertainty: {uncertainty:.4f}")
                        print()

                    except Exception as e:
                        print(f"Mode {i + 1}: ERROR in processing - {e}")
                        continue

                if len(valid_predictions) == 0:
                    print("WARNING: No valid predictions available")
                    return None, None

                valid_confidences = [p['confidence'] for p in valid_predictions]
                best_idx = np.argmax(valid_confidences)
                best_prediction = valid_predictions[best_idx]

                print(
                    f">>> SELECTED: Mode {best_prediction['mode'] + 1} (Confidence: {best_prediction['confidence']:.4f})")
                print(f">>> Target Position: {best_prediction['predicted_position']}")
                print("=" * 40)

                return best_prediction['predicted_position'], best_prediction['uncertainty']

        except Exception as e:
            print(f"ERROR in MDN prediction: {e}")
            return None, None

    def get_sigma_threshold(self, lambda_factor=0.7):
        """Calculate sigma threshold using mu_sigma + lambda * sigma_sigma"""
        if len(self.sigma_history) < 10:
            return 1.0  # Default threshold

        sigma_array = np.array(list(self.sigma_history))
        self.sigma_mean = np.mean(sigma_array)
        self.sigma_std = np.std(sigma_array)

        threshold = self.sigma_mean + lambda_factor * self.sigma_std
        print(
            f"DEBUG: Sigma threshold calculation - mean={self.sigma_mean:.4f}, std={self.sigma_std:.4f}, threshold={threshold:.4f}")
        return threshold


class Drone:
    def __init__(self, position, velocity=np.zeros(3), acceleration=np.zeros(3)):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.array(acceleration, dtype=float)

    def quintic_polynomial_trajectory(self, p0, pf, v0, vf, a0, af, T, t):
        """五次多项式轨迹计算"""
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
            [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
            [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]
        ])

        b = np.array([p0, v0, a0, pf, vf, af])

        try:
            coeffs = np.linalg.solve(A, b)
            trajectory = np.polyval(coeffs[::-1], t)
            return trajectory
        except np.linalg.LinAlgError:
            return p0 + (pf - p0) * (t / T)

    def calc_first_derivative(self, p0, pf, v0, vf, a0, af, T, t):
        """计算速度（一阶导数）"""
        A = np.array([
            [0, 0, 0, 0, 0, 1],
            [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
            [0, 0, 0, 0, 1, 0],
            [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0]
        ])

        B = np.array([p0, pf, v0, vf, a0, af])

        try:
            coefficients = np.linalg.solve(A, B)
            a = coefficients
            return 5 * a[0] * t ** 4 + 4 * a[1] * t ** 3 + 3 * a[2] * t ** 2 + 2 * a[3] * t + a[4]
        except np.linalg.LinAlgError:
            return (pf - p0) / T


class QuinticMotionPlanner:
    def calculate_terminal_conditions(self, current_pos, target_pos, current_vel):
        """计算终端条件"""
        direction_to_target = target_pos - current_pos
        distance_to_target = np.linalg.norm(direction_to_target)

        if distance_to_target > 0.01:
            speed_magnitude = min(4.0, distance_to_target / 0.1)
            terminal_velocity = direction_to_target / distance_to_target * speed_magnitude
        else:
            terminal_velocity = current_vel * 0.8

        terminal_acceleration = np.zeros(3)
        return terminal_velocity, terminal_acceleration

    def plan_trajectory(self, drone, target_position):
        """为单个无人机规划轨迹"""
        p0 = drone.position.copy()
        v0 = drone.velocity.copy()
        a0 = drone.acceleration.copy()

        pf = target_position
        vf, af = self.calculate_terminal_conditions(p0, pf, v0)

        T = 0.05
        t = T

        new_position = np.zeros(3)
        new_velocity = np.zeros(3)

        for dim in range(3):
            new_position[dim] = drone.quintic_polynomial_trajectory(
                p0[dim], pf[dim], v0[dim], vf[dim], a0[dim], af[dim], T, t
            )
            new_velocity[dim] = drone.calc_first_derivative(
                p0[dim], pf[dim], v0[dim], vf[dim], a0[dim], af[dim], T, t
            )

        return new_position, new_velocity
class FormationController:
    """Controls the formation of outer UAVs around the center UAV"""

    def __init__(self, formation_radius=3):
        self.formation_radius = formation_radius
        self.k_d = 0.01  # Distance change gain
        self.k_sigma =0 # Sigma gain
        self.omega_max = 3.0  # Max angular velocity

        # Previous distances for calculating distance change
        self.prev_distances = [None, None, None]
        self.dt = 0.05
        self.time = 0.0
        self.motion_planner = QuinticMotionPlanner()
        # 轨迹规划参数
        self.current_outer_positions = None
        self.last_perpendicular_normal = None
        self.smooth_factor = 0.1

    def check_triangle_quality(self, positions):
        """检查三角形质量，返回最小角度"""
        if len(positions) < 3:
            return 0.0

        a = np.linalg.norm(positions[1] - positions[2])
        b = np.linalg.norm(positions[0] - positions[2])
        c = np.linalg.norm(positions[0] - positions[1])

        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            return 0.0

        cos_A = np.clip((b * b + c * c - a * a) / (2 * b * c), -1.0, 1.0)
        cos_B = np.clip((a * a + c * c - b * b) / (2 * a * c), -1.0, 1.0)
        cos_C = np.clip((a * a + b * b - c * c) / (2 * a * b), -1.0, 1.0)

        angle_A = np.degrees(np.arccos(cos_A))
        angle_B = np.degrees(np.arccos(cos_B))
        angle_C = np.degrees(np.arccos(cos_C))

        return min(angle_A, angle_B, angle_C)

    def find_formation_plane_intersection_point(self, center_pos, target_pos, current_outer_positions):
        """计算当前编队平面与目标-中心连线的交点"""
        if len(current_outer_positions) < 3:
            return center_pos

        v1 = current_outer_positions[1] - current_outer_positions[0]
        v2 = current_outer_positions[2] - current_outer_positions[0]
        plane_normal = np.cross(v1, v2)

        if np.linalg.norm(plane_normal) < 1e-6:
            return center_pos

        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        plane_point = current_outer_positions[0]

        target_direction = target_pos - center_pos
        if np.linalg.norm(target_direction) < 1e-6:
            return center_pos

        target_direction = target_direction / np.linalg.norm(target_direction)

        numerator = np.dot(plane_normal, plane_point - center_pos)
        denominator = np.dot(plane_normal, target_direction)

        if abs(denominator) < 1e-6:
            return center_pos

        t = numerator / denominator
        intersection_point = center_pos + t * target_direction

        return intersection_point

    def compute_perpendicular_plane_through_intersection(self, center_pos, target_pos, current_outer_positions):
        """计算过交点且垂直于目标-中心连线的平面"""
        intersection_point = self.find_formation_plane_intersection_point(
            center_pos, target_pos, current_outer_positions
        )

        target_direction = target_pos - center_pos
        if np.linalg.norm(target_direction) < 1e-6:
            target_direction = np.array([1.0, 0.0, 0.0])
        else:
            target_direction = target_direction / np.linalg.norm(target_direction)

        if abs(target_direction[2]) < 0.9:
            reference = np.array([0.0, 0.0, 1.0])
        else:
            reference = np.array([1.0, 0.0, 0.0])

        u_vec = reference - np.dot(reference, target_direction) * target_direction
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = np.cross(target_direction, u_vec)

        return intersection_point, u_vec, v_vec, target_direction

    def progressive_distance_adjustment(self, positions, center_pos, intersection_point, u_vec, v_vec,
                                        formation_radius, min_distance=0.7, max_move_distance=0.3, max_iterations=10):
        """
        渐进式调整UAV位置，保持最小移动的同时满足间距约束

        Args:
            positions: 当前UAV位置列表
            center_pos: 中心无人机位置
            intersection_point: 垂直平面参考点
            u_vec, v_vec: 垂直平面正交基向量
            formation_radius: 编队半径
            min_distance: 最小UAV间距
            max_move_distance: 单次最大移动距离
            max_iterations: 最大迭代次数

        Returns:
            adjusted_positions: 调整后的位置
            success: 是否成功满足约束
        """

        print(f"\n🔧 Progressive distance adjustment:")
        print(f"  Min distance required: {min_distance}m")
        print(f"  Max move per iteration: {max_move_distance}m")

        current_positions = [pos.copy() for pos in positions]

        for iteration in range(max_iterations):
            # 检查当前间距违反情况
            violations = []
            min_dist_found = float('inf')

            for i in range(3):
                for j in range(i + 1, 3):
                    dist = np.linalg.norm(current_positions[i] - current_positions[j])
                    min_dist_found = min(min_dist_found, dist)
                    if dist < min_distance:
                        violations.append((i, j, dist))

            if not violations:
                print(f"  ✅ Iteration {iteration}: All constraints satisfied! Min distance: {min_dist_found:.3f}m")
                return current_positions, True

            print(f"  📈 Iteration {iteration}: {len(violations)} violations, min_dist: {min_dist_found:.3f}m")

            # 🎯 渐进式调整策略：处理每个违反的UAV对
            adjustment_made = False

            for i, j, current_dist in violations:
                # 计算需要增加的距离
                deficit = min_distance - current_dist + 0.05  # 加buffer

                # 🔧 关键策略：两个UAV相互远离
                # 计算连接向量
                uav_i_to_j = current_positions[j] - current_positions[i]
                distance_3d = np.linalg.norm(uav_i_to_j)

                if distance_3d > 1e-6:
                    # 单位方向向量
                    direction = uav_i_to_j / distance_3d

                    # 每个UAV移动deficit的一半距离
                    move_distance = min(deficit / 2, max_move_distance / 2)

                    # 应用移动：i向后退，j向前移
                    adjustment_i = -move_distance * direction
                    adjustment_j = +move_distance * direction

                    current_positions[i] += adjustment_i
                    current_positions[j] += adjustment_j

                    adjustment_made = True

                    print(f"    UAV{i} ↔ UAV{j}: deficit={deficit:.3f}m, move=±{move_distance:.3f}m")

            if not adjustment_made:
                print(f"  ⚠️ No more adjustments possible, stopping at iteration {iteration}")
                break

            # 🔧 重要：调整后重新满足中心距离约束
            for i in range(3):
                # 将位置重新投影到垂直平面
                relative = current_positions[i] - intersection_point
                x = np.dot(relative, u_vec)
                y = np.dot(relative, v_vec)
                plane_pos = intersection_point + x * u_vec + y * v_vec

                # 调整到正确的中心距离
                center_to_plane = plane_pos - center_pos
                distance = np.linalg.norm(center_to_plane)
                if distance > 1e-6:
                    current_positions[i] = center_pos + center_to_plane / distance * formation_radius

        # 最终检查
        final_violations = []
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(current_positions[i] - current_positions[j])
                if dist < min_distance:
                    final_violations.append((i, j, dist))

        success = len(final_violations) == 0
        status = "✅ SUCCESS" if success else f"⚠️ PARTIAL ({len(final_violations)} violations remain)"
        print(f"  {status} after {max_iterations} iterations")

        return current_positions, success

    def project_to_perpendicular_plane_with_quality_constraint(self, center_pos, target_pos,
                                                               current_outer_positions,
                                                               min_angle_threshold=25.0):
        """将无人机投影到垂直平面，保证三角形质量和距离约束"""
        intersection_point, u_vec, v_vec, target_direction = self.compute_perpendicular_plane_through_intersection(
            center_pos, target_pos, current_outer_positions
        )

        # ========= 原有的投影逻辑保持不变 =========
        projected_positions = []
        for pos in current_outer_positions:
            # 投影到垂直平面
            relative_to_intersection = pos - intersection_point
            u_component = np.dot(relative_to_intersection, u_vec)
            v_component = np.dot(relative_to_intersection, v_vec)

            projected_on_plane = intersection_point + u_component * u_vec + v_component * v_vec

            # 关键修正：保持与中心的距离为formation_radius
            center_to_projected = projected_on_plane - center_pos
            distance = np.linalg.norm(center_to_projected)

            if distance > 1e-6:
                final_position = center_pos + center_to_projected / distance * self.formation_radius
            else:
                # 如果投影点与中心重合，在垂直平面内构建默认位置
                angle = len(projected_positions) * 2 * np.pi / 3
                local_x = self.formation_radius * np.cos(angle)
                local_y = self.formation_radius * np.sin(angle)
                final_position = intersection_point + local_x * u_vec + local_y * v_vec

                # 再次调整距离
                center_to_final = final_position - center_pos
                distance = np.linalg.norm(center_to_final)
                if distance > 1e-6:
                    final_position = center_pos + center_to_final / distance * self.formation_radius

            projected_positions.append(final_position)

            # 验证距离
            actual_distance = np.linalg.norm(final_position - center_pos)
            print(
                f"DEBUG: UAV{len(projected_positions) - 1} distance to center: {actual_distance:.3f}m (target: {self.formation_radius:.3f}m)")

        # ========= 🔥 新增：间距检查和调整 =========
        print(f"\n🔍 Checking inter-UAV distances:")
        min_inter_uav_distance = 0.7
        distance_violations = []

        for i in range(3):
            for j in range(i + 1, 3):
                distance_between = np.linalg.norm(projected_positions[i] - projected_positions[j])
                status = "✅" if distance_between >= min_inter_uav_distance else "❌"
                print(f"  {status} Distance UAV{i} ↔ UAV{j}: {distance_between:.3f}m")

                if distance_between < min_inter_uav_distance:
                    distance_violations.append((i, j, distance_between))

        # 如果有间距违反，使用渐进式调整
        if distance_violations:
            print(f"🔧 Found {len(distance_violations)} distance violations, applying progressive adjustment...")

            adjusted_positions, adjustment_success = self.progressive_distance_adjustment(
                projected_positions, center_pos, intersection_point, u_vec, v_vec,
                self.formation_radius, min_inter_uav_distance, max_move_distance=0.4
            )

            if adjustment_success:
                print(f"✅ Progressive adjustment successful!")
                projected_positions = adjusted_positions
            else:
                print(f"⚠️ Progressive adjustment partial success, using best available positions")
                projected_positions = adjusted_positions
        else:
            print(f"✅ All distance constraints already satisfied!")

        # ========= 原有的三角形质量检查逻辑保持不变 =========
        min_angle = self.check_triangle_quality(projected_positions)

        if min_angle < min_angle_threshold:
            print(f"DEBUG: Poor triangle quality ({min_angle:.1f}°), rebuilding...")

            # 🔧 重建时也要考虑间距约束
            print(f"🔧 Rebuilding standard triangle with distance constraints...")

            adjusted_positions = []
            for i in range(3):
                angle = i * 2 * np.pi / 3
                local_x = self.formation_radius * np.cos(angle)
                local_y = self.formation_radius * np.sin(angle)

                position = intersection_point + local_x * u_vec + local_y * v_vec

                # 确保距离约束
                center_to_pos = position - center_pos
                distance = np.linalg.norm(center_to_pos)
                if distance > 1e-6:
                    position = center_pos + center_to_pos / distance * self.formation_radius

                adjusted_positions.append(position)

            # 验证重建后的间距
            print(f"🔍 Verifying rebuilt triangle distances:")
            rebuild_violations = []
            for i in range(3):
                for j in range(i + 1, 3):
                    dist = np.linalg.norm(adjusted_positions[i] - adjusted_positions[j])
                    status = "✅" if dist >= min_inter_uav_distance else "❌"
                    print(f"  {status} Rebuilt UAV{i} ↔ UAV{j}: {dist:.3f}m")
                    if dist < min_inter_uav_distance:
                        rebuild_violations.append((i, j, dist))

            if not rebuild_violations:
                print(f"✅ Rebuilt triangle satisfies all constraints")
                projected_positions = adjusted_positions
            else:
                print(f"⚠️ Rebuilt triangle still has {len(rebuild_violations)} violations")
                print(f"   This indicates formation_radius ({self.formation_radius}m) may be too small")
                print(f"   Using best available positions anyway")
                projected_positions = adjusted_positions

        return projected_positions, intersection_point, u_vec, v_vec

    def apply_local_rotation(self, uav_id, perpendicular_pos, center_pos, target_pos,
                             intersection_point, sigma, sigma_threshold, all_perpendicular_positions):
        """在局部平面内应用旋转控制，加入安全距离约束"""
        current_distance = np.linalg.norm(target_pos - perpendicular_pos)

        if self.prev_distances[uav_id] is not None:
            e_d = self.prev_distances[uav_id] - current_distance
        else:
            e_d = 0.0
        self.prev_distances[uav_id] = current_distance

        if sigma is not None and sigma_threshold is not None:
            e_sigma = sigma_threshold - sigma
        else:
            e_sigma = 0.0

        # 使用您的原始智能公式
        omega = self.k_d * e_d + self.k_sigma * e_sigma

        # 计算局部旋转平面（无人机-中心-目标）
        center_to_uav = perpendicular_pos - center_pos
        center_to_target = target_pos - center_pos

        local_plane_normal = np.cross(center_to_uav, center_to_target)
        if np.linalg.norm(local_plane_normal) > 0.01:
            local_plane_normal = local_plane_normal / np.linalg.norm(local_plane_normal)

            # 围绕中心点旋转（不是交点）
            rotation_angle = omega * self.dt

            # 使用Rodrigues旋转公式
            relative_pos = perpendicular_pos - center_pos
            cos_theta = np.cos(rotation_angle)
            sin_theta = np.sin(rotation_angle)

            rotated_relative_pos = (relative_pos * cos_theta +
                                    np.cross(local_plane_normal, relative_pos) * sin_theta +
                                    local_plane_normal * np.dot(local_plane_normal, relative_pos) * (1 - cos_theta))

            # 保持与中心距离为formation_radius
            rotated_distance = np.linalg.norm(rotated_relative_pos)
            if rotated_distance > 1e-6:
                rotated_relative_pos = rotated_relative_pos / rotated_distance * self.formation_radius

            proposed_position = center_pos + rotated_relative_pos

            # 🔒 添加安全距离检查
            min_safe_distance = 0.75  # 安全距离阈值
            is_safe = True

            # 构建旋转后的完整位置列表（用于检查）
            proposed_all_positions = all_perpendicular_positions.copy()
            proposed_all_positions[uav_id] = proposed_position

            # 检查所有外围无人机两两之间的距离
            for i in range(len(proposed_all_positions)):
                for j in range(i + 1, len(proposed_all_positions)):
                    distance_between = np.linalg.norm(proposed_all_positions[i] - proposed_all_positions[j])
                    if distance_between < min_safe_distance:
                        is_safe = False
                        print(
                            f"DEBUG UAV{uav_id}: UNSAFE ROTATION! Distance between UAV{i} and UAV{j}: {distance_between:.3f}m < {min_safe_distance}m")
                        break
                if not is_safe:
                    break

            if is_safe:
                # 安全，执行旋转
                final_position = proposed_position
                print(f"DEBUG UAV{uav_id}: SAFE ROTATION - e_d={e_d:.3f}, e_sigma={e_sigma:.3f}, omega={omega:.3f}")
            else:
                # 不安全，保持原位置
                final_position = perpendicular_pos
                omega = 0.0  # 设为0表示没有旋转
                print(f"DEBUG UAV{uav_id}: ROTATION BLOCKED due to safety constraint")

            return final_position, omega

        return perpendicular_pos, 0.0

    def calculate_target_positions(self, center_pos, target_pos, current_outer_positions,
                                   sigma, sigma_threshold):
        """计算所有外围无人机的目标位置"""
        self.time += self.dt

        # 第一步：投影到垂直平面，保证质量和距离约束
        perpendicular_positions, intersection_point, u_vec, v_vec = self.project_to_perpendicular_plane_with_quality_constraint(
            center_pos, target_pos, current_outer_positions
        )

        # 第二步：在每个无人机的局部平面内应用旋转
        final_positions = []
        omega_values = []
        # 先复制一份位置用于安全检查
        working_positions = perpendicular_positions.copy()
        for i in range(3):
            final_pos, omega = self.apply_local_rotation(
                i, perpendicular_positions[i], center_pos, target_pos,
                intersection_point, sigma, sigma_threshold,working_positions
            )
            final_positions.append(final_pos)
            omega_values.append(omega)
            # 更新工作位置，供下一个无人机的安全检查使用
            working_positions[i] = final_pos
        return final_positions, omega_values, intersection_point, u_vec, v_vec

    def check_perpendicularity_error(self, center_pos, target_pos, outer_positions):
        """检查垂直度误差"""
        if len(outer_positions) < 3:
            return float('inf')

        v1 = outer_positions[1] - outer_positions[0]
        v2 = outer_positions[2] - outer_positions[0]
        plane_normal = np.cross(v1, v2)

        if np.linalg.norm(plane_normal) < 1e-6:
            return float('inf')

        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        target_direction = target_pos - center_pos
        if np.linalg.norm(target_direction) < 1e-6:
            return 0.0

        target_direction = target_direction / np.linalg.norm(target_direction)

        # 修正：垂直时法向量应该平行于目标方向
        dot_product = np.dot(plane_normal, target_direction)
        angle = np.arccos(np.clip(abs(dot_product), 0.0, 1.0))
        perpendicularity_error = np.degrees(angle)

        return perpendicularity_error

    def calculate_formation_positions(self, center_pos, center_vel):
        """Calculate ideal positions for outer UAVs in equilateral triangle formation"""

        # Create coordinate system with center velocity as forward direction
        if np.linalg.norm(center_vel) < 0.01:
            # If center velocity is zero, use default forward direction
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward = center_vel / np.linalg.norm(center_vel)

        # Create perpendicular vectors for triangle plane
        if abs(forward[2]) < 0.9:
            right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        else:
            right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # Calculate triangle vertices (120 degrees apart)
        angles = [-np.pi/3, np.pi/2, np.pi*6/7]
        ideal_positions = []

        for angle in angles:
            # Position in triangle plane
            offset = self.formation_radius*np.array([0,np.cos(angle),-np.sin(angle)],dtype=float)
            ideal_pos = center_pos + offset
            ideal_positions.append(ideal_pos)

        for i, pos in enumerate(ideal_positions):
            target_distance = np.linalg.norm(pos - center_pos)
            print(f"UAV{i} ideal_position: {pos}")
            print(f"UAV{i} target_distance: {target_distance:.3f}m (should be {self.formation_radius:.3f}m)")
            if abs(target_distance - self.formation_radius) > 0.1:
                print(f"⚠️ UAV{i} target distance ERROR!")
        return ideal_positions

    def calculate_control_velocity_far(self, uav_id, uav_pos, center_pos, center_vel, ideal_pos):
        """Calculate control velocity for far phase (simple leader following)"""

        # Move towards ideal position
        direction_to_ideal = ideal_pos - uav_pos
        distance_to_ideal = np.linalg.norm(direction_to_ideal)

        # UAV1专门调试
        if uav_id == 0:
            current_distance = np.linalg.norm(uav_pos - center_pos)
            ideal_distance = np.linalg.norm(ideal_pos - center_pos)

            print(f"\n=== UAV1 FAR CONTROL DEBUG ===")
            print(f"  uav_pos: {uav_pos}")
            print(f"  center_pos: {center_pos}")
            print(f"  ideal_pos: {ideal_pos}")
            print(f"  current_distance: {current_distance:.3f}m")
            print(f"  ideal_distance: {ideal_distance:.3f}m")
            print(f"  direction_to_ideal: {direction_to_ideal}")
            print(f"  distance_to_ideal: {distance_to_ideal:.3f}m")

        if distance_to_ideal > 0.01:
            # Proportional control towards ideal position
            kp = 3.0
            velocity_correction = kp * direction_to_ideal

            # Add center velocity for formation following
            target_velocity = center_vel + velocity_correction
            #target_velocity = center_vel
            if uav_id == 0:
                print(f"  kp: {kp}")
                print(f"  velocity_correction: {velocity_correction}")
                print(f"  velocity_correction_magnitude: {np.linalg.norm(velocity_correction):.3f}m/s")
                print(f"  center_vel: {center_vel}")
                print(f"  target_velocity: {target_velocity}")
                print(f"  target_velocity_magnitude: {np.linalg.norm(target_velocity):.3f}m/s")
        else:
            target_velocity = center_vel
            if uav_id == 0:
                print(f"  distance_to_ideal <= 0.01, using center_vel only")

        return target_velocity

    def calculate_control_velocity_near(self, uav_id, uav_pos, center_pos, center_vel,
                                        target_pos, sigma, sigma_threshold):
        """Calculate control velocity for near phase (rotation around center)"""
        # 这个方法现在作为接口兼容性方法，实际控制在 calculate_formation_control 中处理
        # 保持原有接口不变
        # Calculate distance change (e_d)
        current_distance = np.linalg.norm(uav_pos - target_pos)

        if self.prev_distances[uav_id] is not None:
            e_d = self.prev_distances[uav_id] - current_distance
        else:
            e_d = 0.0

        self.prev_distances[uav_id] = current_distance

        # Calculate sigma difference (e_sigma)
        e_sigma = sigma_threshold - sigma if sigma is not None else 0.0

        # Calculate angular velocity
        omega = self.k_d * e_d + 0*self.k_sigma * e_sigma  #越近，sigma越小  omega越大
        #omega = np.clip(omega, -self.omega_max, self.omega_max)

        # Calculate rotation in the plane of (uav, center, target)
        center_to_uav = uav_pos - center_pos
        center_to_target = target_pos - center_pos

        # Create rotation plane normal vector
        if np.linalg.norm(center_to_uav) > 0.01 and np.linalg.norm(center_to_target) > 0.01:
            normal = np.cross(center_to_uav, center_to_target)
            if np.linalg.norm(normal) > 0.01:
                normal = normal / np.linalg.norm(normal)
            else:
                normal = np.array([0.0, 0.0, 1.0])  # Default normal
        else:
            normal = np.array([0.0, 0.0, 1.0])

        # Calculate tangential direction (perpendicular to radius in rotation plane)
        radius_vector = uav_pos - center_pos
        if np.linalg.norm(radius_vector) > 0.01:
            tangent = np.cross(normal, radius_vector)
            tangent = tangent / np.linalg.norm(tangent)
        else:
            tangent = np.array([1.0, 0.0, 0.0])

        # Calculate rotational velocity
        rotational_velocity = omega * self.formation_radius * tangent

        # Add center velocity for formation following
        target_velocity = center_vel+rotational_velocity
        print(f"DEBUG UAV{uav_id}: e_d={e_d:.3f}, e_sigma={e_sigma:.3f}, omega={omega:.3f},rotational_velocity={rotational_velocity},radius_vector={radius_vector},center_vel={center_vel}")
        print(f'target_velocity={target_velocity}')
        return target_velocity


class FourUAVFormationSystem:
    def __init__(self, centerline_model_path, history_frames=10, prediction_frames=5, sample_rate=20,
                 enemy_motion_mode='helix'):
        self.history_frames = history_frames
        self.prediction_frames = prediction_frames
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate

        # Initialize UAVs - center UAV at origin, outer UAVs in triangle formation
        self.center_uav = SimulatedOurUAV(initial_position=[0.0, 0.0, -15.0], uav_id=0)
        #PN
        self.fixed_speed = 6.0
        self.navigation_constant = 3.0


        # 视线角历史记录
        self.prev_los_angle = None
        self.prev_time = None
        # Initialize outer UAVs in equilateral triangle (1.5m from center)
        triangle_radius = 3
        #MPC
        self.prediction_horizon =10
        self.weight_distance = 1.0  # 距离权重
        self.weight_smoothness = 0.1  # 平滑性权重

        # 目标预测相关
        self.target_history = []
        self.max_history = 5

        # 调试信息
        self.debug_info = {}


        #
        # frpn
        self.navigation_constant = 3.0
        # FRPN parameters from paper
        self.frpn_gain = 19.7  # G parameter
        self.frpn_weight = 0.051  # W parameter (5.1 × 10^-2)

        print(f"MPC Controller initialized: speed={self.fixed_speed}m/s, horizon={self.prediction_horizon}, dt={self.dt}s")

        self.outer_uavs = []    #外围无人机
        for i in range(3):
            angle = i * 2 * np.pi / 3  # 120 degrees apart
            x_offset = triangle_radius * np.cos(angle)
            y_offset = triangle_radius * np.sin(angle)
            initial_pos = [x_offset, y_offset, -15.0]
            self.outer_uavs.append(SimulatedOurUAV(initial_position=initial_pos, uav_id=i + 1))

        self.enemy_uav = SimulatedEnemyUAV(initial_position=[-30.0, 0.0, -15.0], motion_mode=enemy_motion_mode)

        # Data storage
        self.center_positions = deque(maxlen=1000)
        self.outer_positions = [deque(maxlen=1000) for _ in range(3)]
        self.enemy_positions = deque(maxlen=1000)
        self.enemy_velocities = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)

        self.trajectory_buffer = deque(maxlen=50)

        # Prediction and control systems
        self.centerline_predictor = UnifiedCenterLinePredictor(centerline_model_path)

        feature_dim = history_frames * 6
        self.mdn_trainer = OnlineMDNTrainer(feature_dim, 3, num_mixtures=3, replay_buffer_size=1000)

        self.training_data_X = deque(maxlen=500)
        self.training_data_y = deque(maxlen=500)

        # Formation controller
        self.formation_controller = FormationController(formation_radius=triangle_radius)

        # Control parameters
        self.distance_threshold = 5.0
        self.control_state = 'COLLECTING_DATA'
        self.frames_collected = 0

        self.current_centerline = None
        self.distance_to_enemy_projection = float('inf')
        self.intercept_phase = None

        self.velocity_buffer = deque(maxlen=5)
        self.mission_complete = False
        self.running = False
        self.interception_time = None

        self.lock = threading.Lock()
        self.simulation_thread = None

        self.last_target_position = None
        self.debug_frame_count = 0

        # CSV logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = f'four_uav_formation_{enemy_motion_mode}_{timestamp}.csv'
        self.setup_csv_file()

        self.trajectory_planner = UnifiedTrajectoryPlanner()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.parallel_controller = ParallelUAVController(self.client)
        self.airsim_lock = threading.RLock()
        # UAV名称映射
        self.uav_names = {
            'center': 'PX4',
            'outer1': 'our_uav_1',
            'outer2': 'our_uav_2',
            'outer3': 'our_uav_3',
            'enemy': 'EnemyUAV'
        }
        self.setup_pid_controller()
        self.formation_size=3


    def reset_to_ground_state(self):
        """Reset entire system to initial ground state"""
        # Stop current mission
        self.running = False
        self.mission_complete = False

        if self.simulation_thread:
            self.simulation_thread.join()

        # Reset AirSim
        try:
            with self.airsim_lock:
                self.client.reset()
                time.sleep(5)  # Wait for reset to complete
        except Exception as e:
            print(f"AirSim reset error: {e}")

        # Clear all historical data
        self.clear_all_system_data()

    def clear_all_system_data(self):
        """Clear all historical tracking data"""
        self.center_positions.clear()
        for positions in self.outer_positions:
            positions.clear()
        self.enemy_positions.clear()
        self.enemy_velocities.clear()
        self.timestamps.clear()
        self.trajectory_buffer.clear()
        self.training_data_X.clear()
        self.training_data_y.clear()

        # Reset AI models
        self.centerline_predictor = UnifiedCenterLinePredictor(
            'unified_centerline_model.pth'
        )
        self.mdn_trainer = OnlineMDNTrainer(60, 3, num_mixtures=3, replay_buffer_size=1000)

        # Reset control state
        self.control_state = 'COLLECTING_DATA'
        self.frames_collected = 0
        self.current_centerline = None
        self.distance_to_enemy_projection = float('inf')
        self.intercept_phase = None

    def update_enemy_parameters(self, motion_mode, params):
        """Update enemy motion parameters"""
        self.enemy_uav.update_motion_parameters(motion_mode, params)
        self.enemy_uav.reset(new_motion_mode=motion_mode)

    def wait_for_completion(self, timeout=120):
        """Wait for mission completion with timeout"""
        start_time = time.time()
        while not self.mission_complete and (time.time() - start_time) < timeout:
            time.sleep(1.0)

        if not self.mission_complete:
            print(f"Mission timeout after {timeout} seconds")
            self.mission_complete = True

        return self.mission_complete

    def get_experiment_result(self):
        """Get experiment result summary"""
        return {
            'success': self.mission_complete and self.interception_time is not None,
            'interception_time': self.interception_time,
            'total_frames': len(self.timestamps),
            'final_distance': np.linalg.norm(self.center_uav.position - self.enemy_uav.position) if hasattr(self,
                                                                                                            'center_uav') else float(
                'inf')
        }


    def predict_target_trajectory(self, current_target_pos, current_target_vel=None):
        """
        预测目标未来轨迹
        Args:
            current_target_pos: 目标当前位置 [x, y, z]
            current_target_vel: 目标当前速度 [x, y, z] (可选)
        Returns:
            predicted_trajectory: 预测的目标轨迹 [N x 3]
        """
        predicted_trajectory = []

        # 更新目标历史
        self.target_history.append(current_target_pos.copy())
        if len(self.target_history) > self.max_history:
            self.target_history.pop(0)

        # 估计目标速度（如果没有提供）
        if current_target_vel is None and len(self.target_history) >= 2:
            estimated_vel = (self.target_history[-1] - self.target_history[-2]) / self.dt
        elif current_target_vel is not None:
            estimated_vel = current_target_vel.copy()
        else:
            estimated_vel = np.zeros(3)  # 假设目标静止

        # 简单的恒速预测（可以根据需要改进为更复杂的模型）
        current_pos = current_target_pos.copy()
        for i in range(self.prediction_horizon):
            # 恒速模型
            next_pos = current_pos + estimated_vel * self.dt
            predicted_trajectory.append(next_pos.copy())
            current_pos = next_pos

        return np.array(predicted_trajectory)

    def dynamics_model(self, pos, vel):
        """
        无人机动力学模型（简单的积分模型）
        Args:
            pos: 当前位置 [x, y, z]
            vel: 当前速度 [x, y, z]
        Returns:
            next_pos: 下一步位置
        """
        return pos + vel * self.dt

    def cost_function(self, control_vars, initial_pos, target_trajectory):
        """
        MPC代价函数
        Args:
            control_vars: 控制变量序列 [N*2] (每步的航向角和俯仰角)
            initial_pos: 初始位置 [x, y, z]
            target_trajectory: 目标预测轨迹 [N x 3]
        Returns:
            total_cost: 总代价
        """
        control_vars = control_vars.reshape(self.prediction_horizon, 2)  # [N x 2]

        total_cost = 0.0
        current_pos = initial_pos.copy()

        # 前一步的控制输入（用于平滑性计算）
        prev_control = None

        for i in range(self.prediction_horizon):
            # 当前步的控制输入：[航向角, 俯仰角]
            yaw = control_vars[i, 0]  # 航向角 (radians)
            pitch = control_vars[i, 1]  # 俯仰角 (radians)

            # 将航向角和俯仰角转换为速度向量（固定模长）
            vx = self.fixed_speed * np.cos(pitch) * np.cos(yaw)
            vy = self.fixed_speed * np.cos(pitch) * np.sin(yaw)
            vz = self.fixed_speed * np.sin(pitch)
            velocity = np.array([vx, vy, vz])

            # 预测下一步位置
            next_pos = self.dynamics_model(current_pos, velocity)

            # 距离代价：与目标的距离
            target_pos = target_trajectory[i]
            distance_cost = np.linalg.norm(next_pos - target_pos) ** 2
            total_cost += self.weight_distance * distance_cost

            # 平滑性代价：控制输入的变化
            if prev_control is not None:
                control_change = np.linalg.norm(control_vars[i] - prev_control) ** 2
                total_cost += self.weight_smoothness * control_change

            # 更新状态
            current_pos = next_pos
            prev_control = control_vars[i].copy()

        return total_cost

    def solve_mpc(self, current_pos, current_vel, target_trajectory):
        """
        求解MPC优化问题
        Args:
            current_pos: 当前位置 [x, y, z]
            current_vel: 当前速度 [x, y, z]
            target_trajectory: 目标预测轨迹 [N x 3]
        Returns:
            optimal_velocity: 最优控制速度 [x, y, z]
        """
        # 初始猜测：朝向第一个目标点的方向
        target_direction = target_trajectory[0] - current_pos
        target_distance = np.linalg.norm(target_direction)

        if target_distance < 1e-6:
            # 目标太近，保持当前速度方向
            if np.linalg.norm(current_vel) > 1e-6:
                return self.fixed_speed * current_vel / np.linalg.norm(current_vel)
            else:
                return np.array([self.fixed_speed, 0.0, 0.0])

        # 计算初始航向角和俯仰角
        initial_yaw = np.arctan2(target_direction[1], target_direction[0])
        horizontal_distance = np.sqrt(target_direction[0] ** 2 + target_direction[1] ** 2)
        initial_pitch = np.arctan2(target_direction[2], horizontal_distance)

        # 初始控制序列：保持朝向目标
        initial_guess = np.tile([initial_yaw, initial_pitch], self.prediction_horizon)

        # 设置边界约束（角度范围）
        bounds = []
        for i in range(self.prediction_horizon):
            bounds.extend([
                (-np.pi, np.pi),  # 航向角范围
                (-np.pi / 2, np.pi / 2)  # 俯仰角范围
            ])

        try:
            # 求解优化问题
            result = minimize(
                fun=self.cost_function,
                x0=initial_guess,
                args=(current_pos, target_trajectory),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )

            if result.success:
                # 提取第一步的最优控制输入
                optimal_controls = result.x.reshape(self.prediction_horizon, 2)
                optimal_yaw = optimal_controls[0, 0]
                optimal_pitch = optimal_controls[0, 1]

                # 转换为速度向量
                vx = self.fixed_speed * np.cos(optimal_pitch) * np.cos(optimal_yaw)
                vy = self.fixed_speed * np.cos(optimal_pitch) * np.sin(optimal_yaw)
                vz = self.fixed_speed * np.sin(optimal_pitch)

                optimal_velocity = np.array([vx, vy, vz])

                # 存储调试信息
                self.debug_info.update({
                    'mpc_success': True,
                    'optimal_yaw': np.degrees(optimal_yaw),
                    'optimal_pitch': np.degrees(optimal_pitch),
                    'cost': result.fun,
                    'iterations': result.nit
                })

                return optimal_velocity

        except Exception as e:
            print(f"MPC optimization failed: {e}")

        # 优化失败时的备用方案：简单比例导航
        self.debug_info['mpc_success'] = False
        target_direction = target_direction / target_distance
        return self.fixed_speed * target_direction

    def calculate_mpc_control_velocity(self, center_pos, center_vel, target_pos, target_vel, sim_time):
        """
        计算MPC控制速度
        Args:
            center_pos: 中心无人机当前位置 [x, y, z]
            center_vel: 中心无人机当前速度 [x, y, z]
            target_pos: 目标位置 [x, y, z]
            target_vel: 目标速度 [x, y, z] (可选)
            sim_time: 当前仿真时间
        Returns:
            control_velocity: MPC控制速度 [x, y, z]
        """
        # 预测目标轨迹
        target_trajectory = self.predict_target_trajectory(target_pos, target_vel)

        # 求解MPC优化问题
        optimal_velocity = self.solve_mpc(center_pos, center_vel, target_trajectory)

        # 验证速度模长
        actual_speed = np.linalg.norm(optimal_velocity)
        distance_to_target = np.linalg.norm(target_pos - center_pos)

        # 打印调试信息
        success_str = "✓" if self.debug_info.get('mpc_success', False) else "✗"
        print(f"DEBUG MPC [{success_str}]: distance={distance_to_target:.2f}m, "
              f"speed={actual_speed:.2f}m/s (target={self.fixed_speed:.2f}m/s)")

        if self.debug_info.get('mpc_success', False):
            print(f"    Optimal: yaw={self.debug_info.get('optimal_yaw', 0):.1f}°, "
                  f"pitch={self.debug_info.get('optimal_pitch', 0):.1f}°, "
                  f"cost={self.debug_info.get('cost', 0):.4f}")

        return optimal_velocity

    def calculate_frpn_control_velocity(self, center_pos, center_vel, target_pos, target_vel, sim_time):
        """
        Fast Response Proportional Navigation (FRPN) control
        Args:
            center_pos: 中心无人机当前位置 [x, y, z]
            center_vel: 中心无人机当前速度 [x, y, z]
            target_pos: 目标位置 [x, y, z]
            target_vel: 目标速度 [x, y, z]
            sim_time: 当前仿真时间
        Returns:
            control_velocity: FRPN控制速度 [x, y, z]
        """
        # 计算相对位置向量 Δp (目标位置 - 拦截器位置)
        delta_p = target_pos - center_pos

        # 计算相对速度向量 Δv (目标速度 - 拦截器速度)
        delta_v = target_vel - center_vel

        # 计算距离和相对速度模长
        distance = np.linalg.norm(delta_p)
        relative_speed = np.linalg.norm(delta_v)

        print(f"DEBUG FRPN: Distance to target = {distance:.2f}m")
        print(f"DEBUG FRPN: Relative speed = {relative_speed:.2f}m/s")

        # 如果距离太近，直接追踪
        if distance < 2.0:
            print("DEBUG FRPN: Too close to target, direct tracking")
            direction = delta_p / (distance + 1e-6)
            return self.fixed_speed * direction

        # 计算预估碰撞时间 tgo = ||Δp|| / ||Δv||
        if relative_speed < 0.1:
            # 相对速度太小时，使用纯追踪
            print("DEBUG FRPN: Low relative speed, using pure pursuit")
            direction = delta_p / distance
            return self.fixed_speed * direction

        tgo = distance / relative_speed

        # 防止tgo过小导致数值不稳定
        tgo = max(tgo, 0.1)

        print(f"DEBUG FRPN: Time to go = {tgo:.2f}s")

        # FRPN控制律: acmd = G × [(1 - W) × (Δp + Δv · tgo)/tgo² + W × Δp]
        # LPN项: (Δp + Δv · tgo) / tgo²
        lpn_term = (delta_p + delta_v * tgo) / (tgo * tgo)

        # Pure Pursuit项: Δp
        pp_term = delta_p

        # FRPN融合
        frpn_acceleration = self.frpn_gain * (
                (1 - self.frpn_weight) * lpn_term + self.frpn_weight * pp_term
        )

        print(f"DEBUG FRPN: LPN term magnitude = {np.linalg.norm(lpn_term):.3f}")
        print(f"DEBUG FRPN: PP term magnitude = {np.linalg.norm(pp_term):.3f}")
        print(f"DEBUG FRPN: Weight W = {self.frpn_weight:.2f}")
        print(f"DEBUG FRPN: Acceleration command = {frpn_acceleration}")

        # 将加速度指令转换为速度方向
        if np.linalg.norm(frpn_acceleration) > 1e-6:
            desired_direction = frpn_acceleration / np.linalg.norm(frpn_acceleration)
        else:
            # 备用方案：直接追踪
            desired_direction = delta_p / distance

        # 固定速度大小
        control_velocity = self.fixed_speed * desired_direction

        # 验证速度大小
        actual_speed = np.linalg.norm(control_velocity)
        print(f"DEBUG FRPN: Output speed = {actual_speed:.2f}m/s (target={self.fixed_speed})")
        print(f"DEBUG FRPN: Output direction = {desired_direction}")

        return control_velocity

    def calculate_los_vector(self, interceptor_pos, target_pos):
        """
        计算3D视线向量 (Line-of-Sight vector)
        Args:
            interceptor_pos: 拦截机位置 [x, y, z]
            target_pos: 目标位置 [x, y, z]
        Returns:
            los_unit_vector: 单位视线向量
            los_distance: 视线距离
        """
        # 计算相对位置向量（从拦截机指向目标）
        relative_pos = target_pos - interceptor_pos
        los_distance = np.linalg.norm(relative_pos)

        if los_distance < 1e-6:  # 避免除零
            return np.array([1.0, 0.0, 0.0]), los_distance

        # 单位视线向量
        los_unit_vector = relative_pos / los_distance

        return los_unit_vector, los_distance

    def calculate_pn_control_velocity(self, center_pos, center_vel, target_pos, sim_time):
        """
        使用比例导航律计算中心无人机的控制速度（3D版本，固定速度模长）
        Args:
            center_pos: 中心无人机当前位置 [x, y, z]
            center_vel: 中心无人机当前速度 [x, y, z]
            target_pos: 目标位置 [x, y, z]
            sim_time: 当前仿真时间
        Returns:
            control_velocity: PN控制速度 [x, y, z]
        """
        # 计算当前视线向量
        los_unit_vector, los_distance = self.calculate_los_vector(center_pos, target_pos)

        print(f"DEBUG PN: Distance to target = {los_distance:.2f}m")
        print(f"DEBUG PN: LOS vector = {los_unit_vector}")

        # 如果距离太近，直接朝向目标
        if los_distance < 2.0:
            print("DEBUG PN: Too close to target, direct tracking")
            return self.fixed_speed * los_unit_vector

        # 如果是第一次调用，初始化
        if self.prev_los_angle is None or self.prev_time is None:
            self.prev_los_angle = los_unit_vector.copy()  # 存储视线向量而不是角度
            self.prev_time = sim_time
            print("DEBUG PN: First call, direct tracking")
            return self.fixed_speed * los_unit_vector

        # 计算时间间隔
        dt = sim_time - self.prev_time
        if dt <= 1e-6:
            print("DEBUG PN: dt too small, keeping current direction")
            current_speed = np.linalg.norm(center_vel)
            if current_speed > 1e-6:
                return self.fixed_speed * center_vel / current_speed
            else:
                return self.fixed_speed * los_unit_vector

        # 计算视线向量变化率 (这等价于视线角变化率在3D空间)
        los_vector_rate = (los_unit_vector - self.prev_los_angle) / dt
        los_rate_magnitude = np.linalg.norm(los_vector_rate)

        print(f"DEBUG PN: LOS rate magnitude = {los_rate_magnitude:.4f}")

        # 简化的PN控制：如果视线向量变化很小，直接追踪
        if los_rate_magnitude < 1e-4:
            print("DEBUG PN: LOS stable, direct tracking")
            new_velocity = self.fixed_speed * los_unit_vector
        else:
            # 计算闭合速度（相对速度在视线方向的投影）
            closing_velocity = np.dot(center_vel, los_unit_vector)

            # PN制导律：期望方向 = 视线方向 + 导航常数 * 视线变化率 * 比例因子
            # 这里简化处理：直接在视线方向基础上添加修正
            correction_factor = self.navigation_constant * los_rate_magnitude

            # 限制修正量，避免过度修正
            correction_factor = min(correction_factor, 1.0)

            # 计算修正方向：视线变化的方向
            if los_rate_magnitude > 1e-6:
                correction_direction = los_vector_rate / los_rate_magnitude
            else:
                correction_direction = np.zeros(3)

            # 期望方向 = 基础追踪方向 + PN修正
            desired_direction = los_unit_vector + correction_factor * correction_direction

            # 归一化
            desired_magnitude = np.linalg.norm(desired_direction)
            if desired_magnitude > 1e-6:
                desired_direction = desired_direction / desired_magnitude
            else:
                desired_direction = los_unit_vector

            new_velocity = self.fixed_speed * desired_direction

            print(f"DEBUG PN: Correction factor = {correction_factor:.3f}")
            print(f"DEBUG PN: Desired direction = {desired_direction}")

        # 更新历史记录
        self.prev_los_angle = los_unit_vector.copy()
        self.prev_time = sim_time

        # 验证速度模长
        actual_speed = np.linalg.norm(new_velocity)
        print(f"DEBUG PN: Output speed = {actual_speed:.2f}m/s (target={self.fixed_speed})")

        return new_velocity

    def get_perpendicular_plane_vectors(self, velocity):
        """Get two orthogonal vectors perpendicular to velocity"""
        # Normalize velocity vector
        v_norm = velocity / (np.linalg.norm(velocity) + 1e-10)

        # Find a vector not parallel to velocity
        if abs(v_norm[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])

        # First perpendicular vector
        u1 = np.cross(v_norm, temp)
        u1 = u1 / (np.linalg.norm(u1) + 1e-10)

        # Second perpendicular vector
        u2 = np.cross(v_norm, u1)
        u2 = u2 / (np.linalg.norm(u2) + 1e-10)

        return u1, u2

    def calculate_outer_drone_positions(self, center_pos, velocity):
        """Calculate positions of three outer drones forming equilateral triangle"""
        u1, u2 = self.get_perpendicular_plane_vectors(velocity)

        # Angles for equilateral triangle vertices (120 degrees apart)
        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        positions = []

        # Distance from center to vertex of equilateral triangle
        radius = self.formation_size / np.sqrt(3)

        for angle in angles:
            # Position in the perpendicular plane
            offset = radius * (np.cos(angle) * u1 + np.sin(angle) * u2)
            drone_pos = center_pos + offset
            positions.append(drone_pos)

        return positions
    def setup_pid_controller(self):
        """设置PID控制器参数"""
        # PID参数 - 调优后的稳定参数
        self.kp = np.array([12.0, 12.0, 16.0])  # 比例增益
        self.ki = np.array([0.1, 0.1, 0.2])  # 积分增益
        self.kd = np.array([1.0, 1.0, 1.5])  # 微分增益

        # PID状态
        self.reset_pid_state()

        # 安全限制
        self.max_velocity = 500.0  # 最大速度 m/s
        #self.max_acceleration = 3.0  # 最大加速度 m/s²

    def reset_pid_state(self):
        """重置PID状态"""
        self.error_integral = np.zeros(3)
        self.error_previous = np.zeros(3)

    def compute_pid_velocity(self, target_position, current_position, dt):
        """
        使用PID控制器计算目标速度

        Args:
            target_position: 目标位置
            current_position: 当前位置
            dt: 时间步长

        Returns:
            目标速度
        """
        # 计算位置误差
        error = target_position - current_position

        # 积分项
        self.error_integral += error * dt
        # 防止积分饱和
        self.error_integral = np.clip(self.error_integral, -10.0, 10.0)

        # 微分项
        error_derivative = (error - self.error_previous) / dt if dt > 0 else np.zeros(3)
        self.error_previous = error.copy()

        # PID计算
        velocity_command = (self.kp * error +
                            self.ki * self.error_integral +
                            self.kd * error_derivative)

        # 速度限制
        velocity_magnitude = np.linalg.norm(velocity_command)
        if velocity_magnitude > self.max_velocity:
            velocity_command = velocity_command / velocity_magnitude * self.max_velocity

        return velocity_command


    def get_airsim_state(self, vehicle_name):
        """从AirSim获取单个UAV状态"""
        with self.airsim_lock:
            state = self.client.getMultirotorState(vehicle_name=vehicle_name)
            pos = [state.kinematics_estimated.position.x_val+pos_offsestx[index[vehicle_name]],
                   state.kinematics_estimated.position.y_val+pos_offsesty[index[vehicle_name]],
                   state.kinematics_estimated.position.z_val]
            vel = [state.kinematics_estimated.linear_velocity.x_val,
                   state.kinematics_estimated.linear_velocity.y_val,
                   state.kinematics_estimated.linear_velocity.z_val]
        return np.array(pos), np.array(vel)

    def update_all_states_from_airsim(self):
        """从AirSim更新所有UAV状态"""
        # 更新中心UAV
        with self.airsim_lock:
            center_pos, center_vel = self.get_airsim_state(self.uav_names['center'])
            self.center_uav.position = center_pos
            self.center_uav.velocity = center_vel

        # 更新外围UAV
            for i, uav in enumerate(self.outer_uavs):
                pos, vel = self.get_airsim_state(self.uav_names[f'outer{i + 1}'])
                uav.position = pos
                uav.velocity = vel

        # 更新敌机
            enemy_pos, enemy_vel = self.get_airsim_state(self.uav_names['enemy'])
            self.enemy_uav.position = enemy_pos
            self.enemy_uav.velocity = enemy_vel

    def send_formation_commands_to_airsim(self):
        """发送编队控制指令到AirSim"""
        uav_commands = []

        with self.lock:
            # 中心UAV
            if hasattr(self.center_uav, 'target_velocity'):
                vel = self.center_uav.target_velocity
                uav_commands.append((self.uav_names['center'], vel[0], vel[1], vel[2]))

            # 外围UAV
            for i, uav in enumerate(self.outer_uavs):
                if hasattr(uav, 'target_velocity'):
                    vel = uav.target_velocity
                    uav_commands.append((self.uav_names[f'outer{i + 1}'], vel[0], vel[1], vel[2]))

            # 敌机
            if hasattr(self.enemy_uav, 'target_velocity'):
                vel = self.enemy_uav.target_velocity
                uav_commands.append((self.uav_names['enemy'], vel[0], vel[1], vel[2]))

        if uav_commands:
            self.parallel_controller.send_all_commands_parallel(uav_commands, self.dt)

    def initialize_airsim_formation(self):
        """初始化AirSim编队 - 修复版"""
        print("Initializing AirSim formation...")

        try:
            with self.airsim_lock:
                # 1. 首先启用API控制
                print("Enabling API control for all vehicles...")
                for vehicle_name in self.uav_names.values():
                    print(f"  Enabling API control for {vehicle_name}")
                    self.client.enableApiControl(True, vehicle_name)
                    self.client.armDisarm(True, vehicle_name)
                    time.sleep(0.5)  # 给每个无人机一些时间

                print("All vehicles armed, starting takeoff sequence...")

                # 2. 起飞所有UAV (顺序起飞，避免冲突)
                for vehicle_name in self.uav_names.values():
                    print(f"  Taking off {vehicle_name}...")
                    try:
                        takeoff_future = self.client.takeoffAsync(vehicle_name=vehicle_name)
                        takeoff_future.join()  # 等待起飞完成
                        print(f"    {vehicle_name} takeoff completed")
                        time.sleep(1.0)  # 起飞间隔
                    except Exception as e:
                        print(f"    ERROR: Takeoff failed for {vehicle_name}: {e}")

                print("All takeoffs completed, moving to initial positions...")

            # 3. 移动到初始编队位置 (在锁外执行，避免长时间占用)
            self.move_to_initial_positions()

            # 4. 验证初始化
            print("Verifying formation initialization...")
            time.sleep(2)  # 等待稳定

            # 检查所有无人机的状态
            all_ready = True
            for vehicle_name in self.uav_names.values():
                try:
                    pos, vel = self.get_airsim_state(vehicle_name)
                    print(f"  {vehicle_name}: pos={pos}, z={pos[2]:.1f}")
                    if pos[2] > -2:  # 检查是否在空中 (z坐标为负值)
                        print(f"    WARNING: {vehicle_name} may not be airborne properly")
                        all_ready = False
                except Exception as e:
                    print(f"    ERROR: Cannot get state for {vehicle_name}: {e}")
                    all_ready = False

            if all_ready:
                print("✅ AirSim formation initialized successfully!")
            else:
                print("⚠️ AirSim formation initialization completed with warnings")

        except Exception as e:
            print(f"ERROR in AirSim initialization: {e}")
            import traceback
            traceback.print_exc()

    def move_to_initial_positions(self):
        """移动到算法初始位置 - 调试版本"""
        print("Moving to initial formation positions...")
        offsest_x=[-3,3,-3,30]
        offsest_y=[-3,0,3,0]
        try:
            with self.airsim_lock:
                # 使用顺序移动，避免同时发送太多命令
                move_commands = []

                # 中心UAV到(0,0,-10)
                move_commands.append(("center", 0, 0, -15))

                # 外围UAV到初始三角形位置
                formation_radius = 3
                for i in range(3):
                    angle = -np.pi/6+i * 2 * np.pi / 3
                    x = 0
                    y = formation_radius * np.cos(angle)
                    z = formation_radius * np.sin(angle)
                    move_commands.append((f"outer{i + 1}", offsest_x[i], y+offsest_y[i], -15-z))

                # 敌机到初始位置(-30,0,-5)
                move_commands.append(("enemy", 0, 0, -15))

                # 🔍 调试：打印所有目标位置
                print("🎯 Target positions:")
                for uav_type, x, y, z in move_commands:
                    vehicle_name = self.uav_names[uav_type]
                    print(f"  {vehicle_name} -> ({x+pos_offsestx[index[vehicle_name]]:.1f}, {y+pos_offsesty[index[vehicle_name]]:.1f}, {z:.1f})")

                # 🔍 调试：检查起飞后的当前位置
                print("\n📍 Current positions before move:")
                for uav_type, _, _, _ in move_commands:
                    vehicle_name = self.uav_names[uav_type]
                    try:
                        pos, vel = self.get_airsim_state(vehicle_name)
                        print(f"  {vehicle_name}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                    except Exception as e:
                        print(f"  {vehicle_name}: ERROR getting position - {e}")

                # 执行移动命令
                print("\n🚁 Starting sequential moves...")
                for uav_type, x, y, z in move_commands:
                    vehicle_name = self.uav_names[uav_type]
                    print(f"\n  Moving {vehicle_name} to ({x+pos_offsestx[index[vehicle_name]]:.1f}, {y+pos_offsesty[index[vehicle_name]]:.1f}, {z:.1f})")

                    try:
                        # 🔍 记录移动前位置
                        pos_before, _ = self.get_airsim_state(vehicle_name)
                        print(f"    Before: ({pos_before[0]:.1f}, {pos_before[1]:.1f}, {pos_before[2]:.1f})")

                        # 发送移动命令
                        future = self.client.moveToPositionAsync(
                            x, y, z,
                            velocity=2.0,
                            timeout_sec=30,  # 增加到30秒
                            vehicle_name=vehicle_name
                        )

                        print(f"    Command sent, waiting for completion...")

                        # 等待移动完成
                        future.join()

                        # 🔍 验证移动结果
                        time.sleep(2)  # 等待稳定
                        pos_after, _ = self.get_airsim_state(vehicle_name)
                        print(f"    After:  ({pos_after[0]:.1f}, {pos_after[1]:.1f}, {pos_after[2]:.1f})")

                        # 🔍 计算误差
                        target_pos = np.array([x+pos_offsestx[index[vehicle_name]], y+pos_offsesty[index[vehicle_name]], z])
                        actual_pos = np.array(pos_after)
                        error = np.linalg.norm(actual_pos - target_pos)

                        if error < 2.0:
                            print(f"    ✅ {vehicle_name} SUCCESS (error: {error:.1f}m)")
                        elif error < 5.0:
                            print(f"    ⚠️ {vehicle_name} PARTIAL (error: {error:.1f}m)")
                        else:
                            print(f"    ❌ {vehicle_name} FAILED (error: {error:.1f}m)")
                            print(f"       Expected: ({x+pos_offsestx[index[vehicle_name]]:.1f}, {y+pos_offsesty[index[vehicle_name]]:.1f}, {z:.1f})")
                            print(f"       Got:      ({pos_after[0]:.1f}, {pos_after[1]:.1f}, {pos_after[2]:.1f})")

                        # 给下一个UAV移动留时间
                        time.sleep(3)

                    except Exception as e:
                        print(f"    ❌ Move command failed for {vehicle_name}: {e}")
                        # 即使失败也继续下一个
                        time.sleep(2)

                print("\n  All moves completed sequentially")

            # 最终验证位置
            print("\n🔍 Final position verification:")
            time.sleep(3)

            expected_positions = {
                'center': (0, 0, -15),
                'outer1': (0, 2.59, -13.5),
                'outer2': (0, 0, -18),  # cos(120°), sin(120°)
                'outer3': (0, -2.59, -13.5),  # cos(240°), sin(240°)
                'enemy': (-30, 0, -15)
            }

            all_positioned_correctly = True
            for uav_type in ['center', 'outer1', 'outer2', 'outer3', 'enemy']:
                try:
                    vehicle_name = self.uav_names[uav_type]
                    pos, vel = self.get_airsim_state(vehicle_name)
                    expected = expected_positions[uav_type]
                    error = np.linalg.norm(np.array(pos) - np.array(expected))

                    status = "✅" if error < 3.0 else "❌"
                    print(f"  {status} {vehicle_name}:")
                    print(f"      Expected: ({expected[0]:.1f}, {expected[1]:.1f}, {expected[2]:.1f})")
                    print(f"      Actual:   ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                    print(f"      Error:    {error:.1f}m")

                    if error >= 3.0:
                        all_positioned_correctly = False

                except Exception as e:
                    print(f"  ❌ {vehicle_name}: ERROR getting final position - {e}")
                    all_positioned_correctly = False

            if all_positioned_correctly:
                print("\n🎉 All UAVs positioned correctly!")
            else:
                print("\n⚠️ Some UAVs not in correct positions")
                print("\n🔧 Possible issues to check:")
                print("1. AirSim coordinate system (NED vs ENU)")
                print("2. UAV physics settings in AirSim")
                print("3. Obstacle collisions")
                print("4. UAV controller responsiveness")
                print("5. Network communication delays")

        except Exception as e:
            print(f"ERROR in move_to_initial_positions: {e}")
            import traceback
            traceback.print_exc()
    def setup_csv_file(self):
        self.csv_file_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file_handle)
        header = ['timestamp',
                  'center_x', 'center_y', 'center_z', 'center_vx', 'center_vy', 'center_vz',
                  'outer1_x', 'outer1_y', 'outer1_z', 'outer1_vx', 'outer1_vy', 'outer1_vz',
                  'outer2_x', 'outer2_y', 'outer2_z', 'outer2_vx', 'outer2_vy', 'outer2_vz',
                  'outer3_x', 'outer3_y', 'outer3_z', 'outer3_vx', 'outer3_vy', 'outer3_vz',
                  'enemy_x', 'enemy_y', 'enemy_z', 'enemy_vx', 'enemy_vy', 'enemy_vz',
                  'distance_to_enemy', 'control_state', 'frames_collected',
                  'centerline_start_x', 'centerline_start_y', 'centerline_start_z',
                  'centerline_end_x', 'centerline_end_y', 'centerline_end_z',
                  'intercept_phase', 'distance_to_projection',
                  'baseline_update_count', 'centerline_deviation',
                  'motion_mode', 'motion_time', 'sigma_threshold']
        self.csv_writer.writerow(header)
        self.csv_file_handle.flush()

    def start(self):
        self.running = True
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.start()
        print(f"Four UAV Formation System started with enemy motion: {self.enemy_uav.motion_mode}")

    def stop(self):
        self.running = False
        self.mission_complete = True

        if self.simulation_thread:
            self.simulation_thread.join()

        if self.csv_file_handle:
            self.csv_file_handle.close()
        print("Simulation stopped")
        print(f"Formation trajectory data saved to {self.csv_file}")

    def simulation_loop(self):
        print("Formation simulation thread started")
        self.initialize_airsim_formation()
        # self.client.enableApiControl(True, vehicle_name="PX4")
        # self.client.armDisarm(True, vehicle_name="PX4")
        # self.client.enableApiControl(True, vehicle_name="EnemyUAV")
        # self.client.armDisarm(True, vehicle_name="EnemyUAV")
        # try:
        #     takeoff_future = self.client.takeoffAsync(vehicle_name="PX4")
        #     takeoff_future.join()  # 等待起飞完成
        #     print("PX4 takeoff completed")
        #     time.sleep(1.0)  # 起飞间隔
        #     takeoff_future = self.client.takeoffAsync(vehicle_name="EnemyUAV")
        #     takeoff_future.join()  # 等待起飞完成
        #     print("EnemyUAV takeoff completed")
        #     time.sleep(1.0)  # 起飞间隔
        # except Exception as e:
        #     print(f"    ERROR: Takeoff failed for PX4")
        #     return
        # self.client.moveToPositionAsync(0, 0, -5, 5, vehicle_name="PX4").join()
        # self.client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="EnemyUAV").join()
        start_real_time = time.time()
        simulation_time = 0.0

        while self.running and not self.mission_complete:
            try:

                loop_start_time = time.time()
                enemy_target_velocity = self.enemy_uav.step(self.dt)
                self.enemy_uav.target_velocity = enemy_target_velocity
                with self.airsim_lock:
                    self.client.moveByVelocityAsync(
                        enemy_target_velocity[0], enemy_target_velocity[1], enemy_target_velocity[2],
                        self.dt, vehicle_name=self.uav_names['enemy']
                    )
                self.update_all_states_from_airsim()
                center_pos = self.center_uav.position
                center_vel = self.center_uav.velocity
                enemy_pos = self.enemy_uav.position
                enemy_vel = self.enemy_uav.velocity
                outer_positions = []
                outer_velocities = []
                for uav in self.outer_uavs:
                    outer_positions.append(uav.position)
                    outer_velocities.append(uav.velocity)

                # Check for interception (only center UAV within 0.7m of enemy)
                center_distance_to_enemy = np.linalg.norm(center_pos - enemy_pos)
                # Check for interception (any UAV within 0.7m of enemy)
                all_our_positions = [center_pos] + outer_positions
                distances_to_enemy = [np.linalg.norm(pos - enemy_pos) for pos in all_our_positions]


                self.debug_frame_count += 1
                if self.debug_frame_count % 100 == 0:
                    print(
                        f"DEBUG Frame {self.debug_frame_count}: center={center_pos}, enemy={enemy_pos}, center_dist={center_distance_to_enemy:.2f}")

                if center_distance_to_enemy <= 0.8:
                    self.mission_complete = True
                    self.interception_time = simulation_time
                    print(f"Interception successful by Center UAV!")
                    print(f"   Interception time: {self.interception_time:.2f} seconds")
                    print(f"   Final distance: {center_distance_to_enemy:.3f} meters")
                    break

                current_time = start_real_time + simulation_time
                with self.lock:
                    self.center_positions.append(center_pos.copy())
                    for i, pos in enumerate(outer_positions):
                        self.outer_positions[i].append(pos.copy())
                    self.enemy_positions.append(enemy_pos.copy())
                    self.enemy_velocities.append(enemy_vel.copy())
                    self.timestamps.append(current_time)
                    self.trajectory_buffer.append(enemy_pos.copy())
                    self.update_training_data()

                # Control step
                self.control_step(center_pos, center_vel, outer_positions, outer_velocities,
                                  enemy_pos, enemy_vel, center_distance_to_enemy, simulation_time)
                self.send_formation_commands_to_airsim()
                # CSV logging
                centerline_data = [np.nan] * 6
                centerline_deviation = np.nan
                baseline_info = self.centerline_predictor.get_baseline_info()
                sigma_threshold = self.mdn_trainer.get_sigma_threshold()

                if self.current_centerline is not None:
                    start_point = self.current_centerline[:3]
                    direction_vector = self.current_centerline[3:]
                    end_point = start_point + direction_vector
                    centerline_data = [start_point[0], start_point[1], start_point[2],
                                       end_point[0], end_point[1], end_point[2]]

                intercept_phase_str = self.intercept_phase if self.intercept_phase else 'NONE'
                baseline_update_count = baseline_info['update_count'] if baseline_info else 0

                csv_row = [current_time,
                           center_pos[0], center_pos[1], center_pos[2],
                           center_vel[0], center_vel[1], center_vel[2]]

                for i in range(3):
                    csv_row.extend([outer_positions[i][0], outer_positions[i][1], outer_positions[i][2],
                                    outer_velocities[i][0], outer_velocities[i][1], outer_velocities[i][2]])

                csv_row.extend([enemy_pos[0], enemy_pos[1], enemy_pos[2],
                                enemy_vel[0], enemy_vel[1], enemy_vel[2],
                                center_distance_to_enemy, self.control_state, self.frames_collected])

                csv_row.extend(centerline_data)
                csv_row.extend([intercept_phase_str, self.distance_to_enemy_projection,
                                baseline_update_count, centerline_deviation,
                                self.enemy_uav.motion_mode, self.enemy_uav.current_time, sigma_threshold])

                self.csv_writer.writerow(csv_row)
                self.csv_file_handle.flush()

                if int(simulation_time * 4) % 20 == 0:
                    baseline_age = baseline_info['age_seconds'] if baseline_info else 0
                    print(
                        f"Time: {simulation_time:.1f}s, CenterDist: {center_distance_to_enemy:.2f}m, State: {self.control_state}, "
                        f"Motion: {self.enemy_uav.motion_mode}, "
                        f"Frames: {len(self.trajectory_buffer)}, Phase: {self.intercept_phase}, "
                        f"DistToProj: {self.distance_to_enemy_projection:.2f}m")

                simulation_time += self.dt

                elapsed = time.time() - loop_start_time
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"Formation simulation error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def control_step(self, center_pos, center_vel, outer_positions, outer_velocities,
                     enemy_pos, enemy_vel,center_distance_to_enemy , sim_time):

        target_position = enemy_pos.copy()
        self.frames_collected = len(self.trajectory_buffer)
        current_sigma = None

        print(
            f"DEBUG FORMATION CONTROL [{sim_time:.1f}s]: center_distance_to_enemy={center_distance_to_enemy:.2f}m, state={self.control_state}, frames={self.frames_collected}")

        try:
            # Check if should switch to MDN
            if hasattr(self,
                       'distance_to_enemy_projection') and self.distance_to_enemy_projection <= self.distance_threshold:
                if self.control_state != 'MDN_PREDICTION':
                    print(
                        f"\n>>> Distance to enemy projection {self.distance_to_enemy_projection:.2f}m <= threshold {self.distance_threshold}m")
                    print(f">>> Switching to MDN_PREDICTION")
                    self.control_state = 'MDN_PREDICTION'

            if self.control_state == 'MDN_PREDICTION':
                print(
                    f"DEBUG: Formation MDN_PREDICTION mode - distance_to_projection={self.distance_to_enemy_projection:.2f}m")

                # if len(self.training_data_X) >= 5:
                #     X_batch = list(self.training_data_X)[-5:]
                #     y_batch = list(self.training_data_y)[-5:]
                #     self.mdn_trainer.update_model(X_batch, y_batch)

                predicted_pos, sigma = self.predict_enemy_position()
                current_sigma = sigma
                if predicted_pos is not None:
                    target_position = predicted_pos
                    print(f"DEBUG: Using MDN prediction: {predicted_pos}, sigma: {sigma}")
                else:
                    target_position = enemy_pos
                    print(f"DEBUG: MDN prediction failed, using direct tracking")

            else:
                print(f"DEBUG: Formation far mode - using centerline intercept strategy")

                if self.control_state == 'COLLECTING_DATA':
                    print(f"DEBUG: COLLECTING_DATA - need 20 frames, have {self.frames_collected}")

                    # if len(self.training_data_X) >= 5:
                    #     X_batch = list(self.training_data_X)[-5:]
                    #     y_batch = list(self.training_data_y)[-5:]
                    #     self.mdn_trainer.update_model(X_batch, y_batch)
                    target_position = enemy_pos
                    if self.frames_collected >= 40:
                        print(f"\n>>> Collected {self.frames_collected} frames, switching to CENTERLINE_TRACKING")
                        self.control_state = 'CENTERLINE_TRACKING'



                elif self.control_state == 'CENTERLINE_TRACKING':
                    print(f"DEBUG: CENTERLINE_TRACKING - frames={self.frames_collected}")

                    predicted_centerline = self.centerline_predictor.predict_centerline(
                        list(self.trajectory_buffer)
                    )

                    if predicted_centerline is not None:
                        centerline_changed, deviation = self.centerline_predictor.is_centerline_changed(
                            predicted_centerline, threshold=2.0
                        )

                        if centerline_changed:
                            print(f">>> CENTERLINE CHANGED! Deviation: {deviation:.3f}")
                            self.centerline_predictor.update_baseline_centerline(predicted_centerline)
                        else:
                            print(f">>> Centerline stable, deviation: {deviation:.3f}")

                        self.current_centerline = self.centerline_predictor.get_current_centerline()

                        if self.current_centerline is not None:
                            target_position, intercept_phase, distance_to_projection = self.trajectory_planner.plan_intercept_strategy(
                                center_pos, enemy_pos, self.current_centerline
                            )
                            self.distance_to_enemy_projection = distance_to_projection
                            self.intercept_phase = intercept_phase

                            print(
                                f"DEBUG: Intercept strategy - phase={intercept_phase}, distance_to_projection={distance_to_projection:.2f}")


                        else:
                            print(f"DEBUG: No bdaseline centerline available yet")
                            target_position = enemy_pos
                            self.intercept_phase = 'NO_BASELINE'
                    else:
                        print(f"DEBUG: Centerline prediction failed")
                        target_position = enemy_pos
                        self.intercept_phase = 'PREDICTION_FAILED'

                    if len(self.training_data_X) >= 5:
                        X_batch = list(self.training_data_X)[-5:]
                        y_batch = list(self.training_data_y)[-5:]
                        self.mdn_trainer.update_model(X_batch, y_batch)

            # Target position validation
            target_distance_from_center = np.linalg.norm(target_position - center_pos)
            print(
                f"DEBUG: Final target_position={target_position}, distance_from_center={target_distance_from_center:.2f}")

            if (target_distance_from_center > 100.0 or
                    np.any(np.isnan(target_position)) or
                    np.any(np.isinf(target_position))):
                print(
                    f"DEBUG: WARNING - Invalid target (distance={target_distance_from_center:.2f}m), using enemy position")
                target_position = enemy_pos

            # Calculate control velocities for all UAVs

            with self.lock:
                self.calculate_formation_control(center_pos, center_vel, outer_positions,
                                         target_position, current_sigma)
        except Exception as e:
            print(f"Formation control step error: {e}")
            import traceback
            traceback.print_exc()



    def calculate_formation_control(self, center_pos, center_vel, outer_positions,
                                    target_position, current_sigma):
        """Calculate control velocities for all UAVs in formation"""
        kd = 0.125
        k_sigma = 1
        # Control center UAV
        center_control_velocity = self.calculate_control_velocity(center_pos, target_position)
        self.center_uav.target_velocity = center_control_velocity
        # 计算中心无人机的位移
        center_displacement = center_vel * self.dt
        # Get sigma threshold
        sigma_threshold = self.mdn_trainer.get_sigma_threshold()

        # Control outer UAVs based on current mode
        if self.control_state == 'MDN_PREDICTION':
            # Near phase: rotational control around center
            print(f"DEBUG: Formation near phase control, sigma={current_sigma}, threshold={sigma_threshold}")

            # 第一步：外围无人机跟随中心无人机运动
            followed_outer_positions = []
            for i, pos in enumerate(outer_positions):
                followed_position = pos + center_displacement
                followed_outer_positions.append(followed_position)


            # 第二步：双层控制计算目标位置
            target_positions, omega_values, intersection_point, u_vec, v_vec = self.formation_controller.calculate_target_positions(
                center_pos, target_position, followed_outer_positions, current_sigma, sigma_threshold
            )

            # # 第三步：pid
            for i, uav in enumerate(self.outer_uavs):
                    # 创建临时Drone对象用于轨迹规划

                target_velocity=self.compute_pid_velocity(target_positions[i],uav.position,self.dt)
                uav.target_velocity = target_velocity

            # 性能监控
            perp_error = self.formation_controller.check_perpendicularity_error(
                center_pos, target_position, target_positions
            )
            print(f"DEBUG: Perpendicularity error: {perp_error:.2f}°")

        else:
            # Far phase: leader following with formation maintenance
            print(f"DEBUG: Formation far phase control")

            ideal_positions = self.formation_controller.calculate_formation_positions(
                center_pos, center_vel
            )

            for i, uav in enumerate(self.outer_uavs):
                uav_control_velocity = self.formation_controller.calculate_control_velocity_far(
                    i, outer_positions[i], center_pos, center_vel, ideal_positions[i]
                )
                uav.target_velocity = uav_control_velocity

    def calculate_control_velocity(self, current_pos, target_pos):
        """Calculate control velocity for center UAV (same as original)"""
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance > 0.01:
            max_speed = self.center_uav.max_speed
            raw_velocity = (direction / distance) * max_speed
        else:
            raw_velocity = np.zeros(3)

        if len(self.velocity_buffer) == 0:
            smoothed_velocity = raw_velocity
        else:
            alpha = 0.8
            prev_velocity = np.mean(np.array(self.velocity_buffer), axis=0)
            smoothed_velocity = raw_velocity

        self.velocity_buffer.append(smoothed_velocity.tolist())
        return smoothed_velocity

    def update_training_data(self):
        """Update training data (same as original)"""
        if len(self.enemy_positions) < self.history_frames + self.prediction_frames:
            return

        enemy_positions_list = [pos.copy() for pos in list(self.enemy_positions)]
        enemy_velocities_list = [vel.copy() for vel in list(self.enemy_velocities)]

        if len(enemy_positions_list) >= self.history_frames + self.prediction_frames:
            start_idx = len(enemy_positions_list) - self.history_frames - self.prediction_frames

            input_features = []
            for i in range(start_idx, start_idx + self.history_frames):
                pos = enemy_positions_list[i]
                vel = enemy_velocities_list[i]

                if np.any(np.isnan(pos)) or np.any(np.isnan(vel)) or np.any(np.isinf(pos)) or np.any(np.isinf(vel)):
                    print("WARNING: Invalid data in training sequence, skipping")
                    return

                input_features.extend([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])

            current_pos = enemy_positions_list[start_idx + self.history_frames - 1]
            future_pos = enemy_positions_list[start_idx + self.history_frames + self.prediction_frames - 1]
            position_delta = future_pos - current_pos

            delta_magnitude = np.linalg.norm(position_delta)
            if delta_magnitude > 20.0:
                print(f"WARNING: Large position delta ({delta_magnitude:.2f}m), skipping training sample")
                return

            if np.any(np.isnan(position_delta)) or np.any(np.isinf(position_delta)):
                print("WARNING: Invalid position delta, skipping training sample")
                return

            self.training_data_X.append(input_features)
            self.training_data_y.append(position_delta.copy())

    def predict_enemy_position(self):
        """Predict enemy position using MDN"""
        with self.lock:
            if len(self.enemy_positions) < self.history_frames:
                return None, None

            enemy_positions_list = list(self.enemy_positions)
            enemy_velocities_list = list(self.enemy_velocities)

            input_features = []
            for i in range(-self.history_frames, 0):
                pos = enemy_positions_list[i]
                vel = enemy_velocities_list[i]
                input_features.extend([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])

            current_position = enemy_positions_list[-1].copy()
            predicted_pos, sigma = self.mdn_trainer.predict(input_features, current_position)
            return predicted_pos, sigma

    def get_status(self):
        """Get current system status"""
        with self.lock:
            baseline_info = self.centerline_predictor.get_baseline_info()
            motion_info = self.enemy_uav.get_motion_info()

            # Calculate formation distances
            center_pos = self.center_uav.position
            formation_distances = []
            for uav in self.outer_uavs:
                dist = np.linalg.norm(uav.position - center_pos)
                formation_distances.append(dist)

            return {
                'center_positions_count': len(self.center_positions),
                'enemy_positions_count': len(self.enemy_positions),
                'training_samples': len(self.training_data_X),
                'control_state': self.control_state,
                'frames_collected': self.frames_collected,
                'mission_complete': self.mission_complete,
                'interception_time': self.interception_time,
                'center_uav_position': self.center_uav.position.copy(),
                'outer_uav_positions': [uav.position.copy() for uav in self.outer_uavs],
                'enemy_uav_position': self.enemy_uav.position.copy(),
                'formation_distances': formation_distances,
                'baseline_updates': baseline_info['update_count'] if baseline_info else 0,
                'baseline_age': baseline_info['age_seconds'] if baseline_info else 0,
                'intercept_phase': self.intercept_phase,
                'distance_to_projection': self.distance_to_enemy_projection,
                'motion_info': motion_info,
                'sigma_threshold': self.mdn_trainer.get_sigma_threshold(),
                'center_distance_to_enemy': np.linalg.norm(center_pos - self.enemy_uav.position),
            }

    def create_3d_animation(self):
        """Create 3D animation of the formation pursuit"""
        try:
            df = pd.read_csv(self.csv_file)
            print(f"Creating 3D animation from {len(df)} data points")

            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')

            # Set up the plot
            motion_mode = df['motion_mode'].iloc[0] if 'motion_mode' in df.columns else 'unknown'
            ax.set_title(f'Four UAV Formation Pursuit - {motion_mode.upper()} Enemy', fontsize=14, fontweight='bold')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')

            # Initialize empty lines for trajectories
            center_line, = ax.plot([], [], [], 'b-', linewidth=3, label='Center UAV', alpha=0.8)
            outer1_line, = ax.plot([], [], [], 'g-', linewidth=2, label='Outer UAV 1', alpha=0.8)
            outer2_line, = ax.plot([], [], [], 'orange', linewidth=2, label='Outer UAV 2', alpha=0.8)
            outer3_line, = ax.plot([], [], [], 'purple', linewidth=2, label='Outer UAV 3', alpha=0.8)
            enemy_line, = ax.plot([], [], [], 'r-', linewidth=3, label=f'Enemy UAV ({motion_mode})', alpha=0.9)

            # Current position markers
            center_point, = ax.plot([], [], [], 'bo', markersize=5, label='Center Current')
            outer1_point, = ax.plot([], [], [], 'go', markersize=4)
            outer2_point, = ax.plot([], [], [], 'o', color='orange', markersize=4)
            outer3_point, = ax.plot([], [], [], 'o', color='purple', markersize=4)
            enemy_point, = ax.plot([], [], [], 'ro', markersize=5, label='Enemy Current')

            # Formation lines (connecting adjacent UAVs)
            formation_lines = []
            for i in range(3):
                line, = ax.plot([], [], [], 'k--', linewidth=1, alpha=0.6)
                formation_lines.append(line)
            center_to_outer_lines = []
            for i in range(3):
                line, = ax.plot([], [], [], 'b--', linewidth=1, alpha=0.4)
                center_to_outer_lines.append(line)

            ax.legend()

            # Set axis limits based on data
            all_x = df[['center_x', 'outer1_x', 'outer2_x', 'outer3_x', 'enemy_x']].values.flatten()
            all_y = df[['center_y', 'outer1_y', 'outer2_y', 'outer3_y', 'enemy_y']].values.flatten()
            all_z = df[['center_z', 'outer1_z', 'outer2_z', 'outer3_z', 'enemy_z']].values.flatten()

            margin = 5
            ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
            ax.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)
            ax.set_zlim(np.min(all_z) - margin, np.max(all_z) + margin)

            # Animation function
            def animate(frame):
                # Trail length for trajectories
                trail_length = min(100, frame + 1)
                start_idx = max(0, frame - trail_length + 1)

                # Update trajectory lines
                center_line.set_data_3d(df['center_x'].iloc[start_idx:frame + 1],
                                        df['center_y'].iloc[start_idx:frame + 1],
                                        df['center_z'].iloc[start_idx:frame + 1])
                outer1_line.set_data_3d(df['outer1_x'].iloc[start_idx:frame + 1],
                                        df['outer1_y'].iloc[start_idx:frame + 1],
                                        df['outer1_z'].iloc[start_idx:frame + 1])
                outer2_line.set_data_3d(df['outer2_x'].iloc[start_idx:frame + 1],
                                        df['outer2_y'].iloc[start_idx:frame + 1],
                                        df['outer2_z'].iloc[start_idx:frame + 1])
                outer3_line.set_data_3d(df['outer3_x'].iloc[start_idx:frame + 1],
                                        df['outer3_y'].iloc[start_idx:frame + 1],
                                        df['outer3_z'].iloc[start_idx:frame + 1])
                enemy_line.set_data_3d(df['enemy_x'].iloc[start_idx:frame + 1],
                                       df['enemy_y'].iloc[start_idx:frame + 1],
                                       df['enemy_z'].iloc[start_idx:frame + 1])

                # Update current position markers
                center_point.set_data_3d([df['center_x'].iloc[frame]],
                                         [df['center_y'].iloc[frame]],
                                         [df['center_z'].iloc[frame]])
                outer1_point.set_data_3d([df['outer1_x'].iloc[frame]],
                                         [df['outer1_y'].iloc[frame]],
                                         [df['outer1_z'].iloc[frame]])
                outer2_point.set_data_3d([df['outer2_x'].iloc[frame]],
                                         [df['outer2_y'].iloc[frame]],
                                         [df['outer2_z'].iloc[frame]])
                outer3_point.set_data_3d([df['outer3_x'].iloc[frame]],
                                         [df['outer3_y'].iloc[frame]],
                                         [df['outer3_z'].iloc[frame]])
                enemy_point.set_data_3d([df['enemy_x'].iloc[frame]],
                                        [df['enemy_y'].iloc[frame]],
                                        [df['enemy_z'].iloc[frame]])

                # Update formation lines (connect adjacent outer UAVs)
                outer_positions = [
                    [df['outer1_x'].iloc[frame], df['outer1_y'].iloc[frame], df['outer1_z'].iloc[frame]],
                    [df['outer2_x'].iloc[frame], df['outer2_y'].iloc[frame], df['outer2_z'].iloc[frame]],
                    [df['outer3_x'].iloc[frame], df['outer3_y'].iloc[frame], df['outer3_z'].iloc[frame]]
                ]
                center_pos = [df['center_x'].iloc[frame], df['center_y'].iloc[frame], df['center_z'].iloc[frame]]

                # Connect adjacent outer UAVs (triangle formation)
                for i in range(3):
                    j = (i + 1) % 3
                    formation_lines[i].set_data_3d([outer_positions[i][0], outer_positions[j][0]],
                                                   [outer_positions[i][1], outer_positions[j][1]],
                                                   [outer_positions[i][2], outer_positions[j][2]])

                # Connect center to outer UAVs
                for i in range(3):
                    center_to_outer_lines[i].set_data_3d([center_pos[0], outer_positions[i][0]],
                                                         [center_pos[1], outer_positions[i][1]],
                                                         [center_pos[2], outer_positions[i][2]])

                # Update title with current info
                current_time = df['timestamp'].iloc[frame] - df['timestamp'].iloc[0]
                control_state = df['control_state'].iloc[frame]
                distance = df['distance_to_enemy'].iloc[frame]
                ax.set_title(f'Four UAV Formation Pursuit - {motion_mode.upper()} Enemy\n'
                             f'Time: {current_time:.1f}s, State: {control_state}, Center Distance: {distance:.2f}m',
                             fontsize=12, fontweight='bold')

                return ([center_line, outer1_line, outer2_line, outer3_line, enemy_line,
                         center_point, outer1_point, outer2_point, outer3_point, enemy_point] +
                        formation_lines + center_to_outer_lines)

            # Create animation
            frames = len(df)
            interval = max(100, int(2000 * self.dt))  # Animation speed based on simulation dt

            print(f"Creating animation with {frames} frames, interval={interval}ms")
            anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval,
                                           blit=False, repeat=True)

            # Save animation
            timestamp = self.csv_file.split('_')[-1].split('.')[0]
            animation_filename = f'four_uav_formation_{motion_mode}_{timestamp}.gif'
            print(f"Saving animation as {animation_filename}...")
            anim.save(animation_filename, writer='pillow', fps=5)
            print(f"Animation saved as '{animation_filename}'")

            plt.show()

        except Exception as e:
            print(f"Error creating 3D animation: {e}")
            import traceback
            traceback.print_exc()

    def plot_trajectories_from_csv(self):
        """Plot static analysis of formation trajectories"""
        try:
            df = pd.read_csv(self.csv_file)
            print(f"Loaded {len(df)} data points from {self.csv_file}")

            time_relative = (df['timestamp'] - df['timestamp'].iloc[0])

            fig = plt.figure(figsize=(24, 20))
            motion_mode = df['motion_mode'].iloc[0] if 'motion_mode' in df.columns else 'unknown'
            fig.suptitle(f'Four UAV Formation System Analysis - Enemy Motion: {motion_mode.upper()}',
                         fontsize=16, fontweight='bold')

            # 3D trajectory plot
            ax1 = fig.add_subplot(2, 4, 1, projection='3d')
            ax1.plot(df['center_x'], df['center_y'], df['center_z'], 'b-', linewidth=3, label='Center UAV', alpha=0.8)
            ax1.plot(df['outer1_x'], df['outer1_y'], df['outer1_z'], 'g-', linewidth=2, label='Outer UAV 1', alpha=0.8)
            ax1.plot(df['outer2_x'], df['outer2_y'], df['outer2_z'], 'orange', linewidth=2, label='Outer UAV 2',
                     alpha=0.8)
            ax1.plot(df['outer3_x'], df['outer3_y'], df['outer3_z'], 'purple', linewidth=2, label='Outer UAV 3',
                     alpha=0.8)
            ax1.plot(df['enemy_x'], df['enemy_y'], df['enemy_z'], 'r-', linewidth=3,
                     label=f'Enemy UAV ({motion_mode})', alpha=0.9)

            # Starting positions
            ax1.scatter(df['center_x'].iloc[0], df['center_y'].iloc[0], df['center_z'].iloc[0],
                        c='blue', s=150, marker='o', edgecolors='black', linewidths=2, label='Formation Start')
            ax1.scatter(df['enemy_x'].iloc[0], df['enemy_y'].iloc[0], df['enemy_z'].iloc[0],
                        c='red', s=150, marker='s', edgecolors='black', linewidths=2, label='Enemy Start')

            if self.interception_time is not None:
                ax1.scatter(df['center_x'].iloc[-1], df['center_y'].iloc[-1], df['center_z'].iloc[-1],
                            c='gold', s=200, marker='*', edgecolors='black', linewidths=2, label='Interception')

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title(f'3D Formation Trajectory - {motion_mode.upper()}')
            ax1.legend()

            # XY projection
            ax2 = fig.add_subplot(2, 4, 2)
            ax2.plot(df['center_x'], df['center_y'], 'b-', linewidth=3, label='Center UAV', alpha=0.8)
            ax2.plot(df['outer1_x'], df['outer1_y'], 'g-', linewidth=2, label='Outer UAV 1', alpha=0.8)
            ax2.plot(df['outer2_x'], df['outer2_y'], 'orange', linewidth=2, label='Outer UAV 2', alpha=0.8)
            ax2.plot(df['outer3_x'], df['outer3_y'], 'purple', linewidth=2, label='Outer UAV 3', alpha=0.8)
            ax2.plot(df['enemy_x'], df['enemy_y'], 'r-', linewidth=2,
                     label=f'Enemy UAV ({motion_mode})', alpha=0.8)

            ax2.scatter(df['center_x'].iloc[0], df['center_y'].iloc[0], c='blue', s=100, marker='o', edgecolors='black')
            ax2.scatter(df['enemy_x'].iloc[0], df['enemy_y'].iloc[0], c='red', s=100, marker='s', edgecolors='black')

            if self.interception_time is not None:
                ax2.scatter(df['center_x'].iloc[-1], df['center_y'].iloc[-1], c='gold', s=150, marker='*',
                            edgecolors='black', linewidths=2, label='Interception Point')

            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title(f'XY Formation Plane - {motion_mode.upper()}')
            ax2.legend()
            ax2.grid(True)
            ax2.axis('equal')

            # Distance analysis
            ax3 = fig.add_subplot(2, 4, 3)
            ax3.plot(time_relative, df['distance_to_enemy'], 'b-', linewidth=3, label='Center UAV to Enemy')
            ax3.axhline(y=0.7, color='gold', linestyle='--', label='Interception Distance (0.7m)')
            if 'distance_to_projection' in df.columns:
                projection_mask = ~df['distance_to_projection'].isna()
                if projection_mask.any():
                    ax3.plot(time_relative[projection_mask], df.loc[projection_mask, 'distance_to_projection'],
                             'orange', linewidth=2, label='Distance to Enemy Projection')

            ax3.axhline(y=self.distance_threshold, color='r', linestyle='--',
                        label=f'Threshold ({self.distance_threshold}m)')

            if self.interception_time is not None:
                ax3.axvline(x=self.interception_time, color='gold', linestyle='-', linewidth=3,
                            label=f'Interception at {self.interception_time:.2f}s')

            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Distance (m)')
            ax3.set_title('Distance Analysis (Center UAV Only)')
            ax3.legend()
            ax3.grid(True)

            # Formation integrity
            ax4 = fig.add_subplot(2, 4, 4)

            # Calculate formation distances
            center_to_outer1 = np.sqrt((df['center_x'] - df['outer1_x']) ** 2 +
                                       (df['center_y'] - df['outer1_y']) ** 2 +
                                       (df['center_z'] - df['outer1_z']) ** 2)
            center_to_outer2 = np.sqrt((df['center_x'] - df['outer2_x']) ** 2 +
                                       (df['center_y'] - df['outer2_y']) ** 2 +
                                       (df['center_z'] - df['outer2_z']) ** 2)
            center_to_outer3 = np.sqrt((df['center_x'] - df['outer3_x']) ** 2 +
                                       (df['center_y'] - df['outer3_y']) ** 2 +
                                       (df['center_z'] - df['outer3_z']) ** 2)

            ax4.plot(time_relative, center_to_outer1, 'g-', linewidth=2, label='Center-Outer1', alpha=0.7)
            ax4.plot(time_relative, center_to_outer2, 'orange', linewidth=2, label='Center-Outer2', alpha=0.7)
            ax4.plot(time_relative, center_to_outer3, 'purple', linewidth=2, label='Center-Outer3', alpha=0.7)
            ax4.axhline(y=1.5, color='r', linestyle='--', label='Target Distance (1.5m)')

            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Distance (m)')
            ax4.set_title('Formation Integrity')
            ax4.legend()
            ax4.grid(True)

            # Speed analysis
            ax5 = fig.add_subplot(2, 4, 5)
            center_speeds = np.sqrt(df['center_vx'] ** 2 + df['center_vy'] ** 2 + df['center_vz'] ** 2)
            outer1_speeds = np.sqrt(df['outer1_vx'] ** 2 + df['outer1_vy'] ** 2 + df['outer1_vz'] ** 2)
            outer2_speeds = np.sqrt(df['outer2_vx'] ** 2 + df['outer2_vy'] ** 2 + df['outer2_vz'] ** 2)
            outer3_speeds = np.sqrt(df['outer3_vx'] ** 2 + df['outer3_vy'] ** 2 + df['outer3_vz'] ** 2)
            enemy_speeds = np.sqrt(df['enemy_vx'] ** 2 + df['enemy_vy'] ** 2 + df['enemy_vz'] ** 2)

            ax5.plot(time_relative, center_speeds, 'b-', linewidth=2, label='Center Speed', alpha=0.8)
            ax5.plot(time_relative, outer1_speeds, 'g-', linewidth=1, label='Outer1 Speed', alpha=0.7)
            ax5.plot(time_relative, outer2_speeds, 'orange', linewidth=1, label='Outer2 Speed', alpha=0.7)
            ax5.plot(time_relative, outer3_speeds, 'purple', linewidth=1, label='Outer3 Speed', alpha=0.7)
            ax5.plot(time_relative, enemy_speeds, 'r-', linewidth=2,
                     label=f'Enemy Speed ({motion_mode})', alpha=0.8)

            if self.interception_time is not None:
                ax5.axvline(x=self.interception_time, color='gold', linestyle='-', linewidth=3, alpha=0.7)

            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Speed (m/s)')
            ax5.set_title(f'Formation Speeds - {motion_mode.upper()}')
            ax5.legend()
            ax5.grid(True)

            # Control state timeline
            ax6 = fig.add_subplot(2, 4, 6)
            states = df['control_state'].unique()
            state_colors = ['blue', 'orange', 'green', 'red', 'purple']

            for i, state in enumerate(states):
                state_data = df[df['control_state'] == state]
                if len(state_data) > 0:
                    state_times = state_data['timestamp'] - df['timestamp'].iloc[0]
                    ax6.scatter(state_times, [i] * len(state_times),
                                c=state_colors[i % len(state_colors)],
                                label=state, alpha=0.7, s=20)

            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Control State')
            ax6.set_title('Control State Timeline')
            ax6.legend()
            ax6.grid(True)

            # Sigma threshold analysis
            ax7 = fig.add_subplot(2, 4, 7)
            if 'sigma_threshold' in df.columns:
                ax7.plot(time_relative, df['sigma_threshold'], 'purple', linewidth=2, label='Sigma Threshold')
                ax7.set_xlabel('Time (s)')
                ax7.set_ylabel('Sigma Threshold')
                ax7.set_title('Uncertainty Threshold Evolution')
                ax7.legend()
                ax7.grid(True)

            # Performance metrics
            ax8 = fig.add_subplot(2, 4, 8)
            ax8.axis('off')

            min_distance = df['distance_to_enemy'].min()
            avg_distance = df['distance_to_enemy'].mean()

            state_durations = {}
            for state in df['control_state'].unique():
                state_data = df[df['control_state'] == state]
                duration = len(state_data) * 0.05
                state_durations[state] = duration

            total_baseline_updates = df['baseline_update_count'].max() if 'baseline_update_count' in df.columns else 0

            # Formation quality metrics
            avg_formation_error = []
            target_radius = 1.5
            for _, row in df.iterrows():
                center_pos = np.array([row['center_x'], row['center_y'], row['center_z']])
                outer_positions = [
                    np.array([row['outer1_x'], row['outer1_y'], row['outer1_z']]),
                    np.array([row['outer2_x'], row['outer2_y'], row['outer2_z']]),
                    np.array([row['outer3_x'], row['outer3_y'], row['outer3_z']])
                ]

                formation_errors = []
                for pos in outer_positions:
                    error = abs(np.linalg.norm(pos - center_pos) - target_radius)
                    formation_errors.append(error)
                avg_formation_error.append(np.mean(formation_errors))

            avg_formation_error = np.mean(avg_formation_error)

            metrics_text = f"FORMATION PERFORMANCE - {motion_mode.upper()}\n\n"
            metrics_text += f"Mission Status: {'SUCCESS' if self.interception_time else 'INCOMPLETE'}\n"
            if self.interception_time:
                metrics_text += f"Interception Time: {self.interception_time:.2f}s\n"
            metrics_text += f"Final Distance: {df['distance_to_enemy'].iloc[-1]:.2f}m\n"
            metrics_text += f"Minimum Distance: {min_distance:.2f}m\n"
            metrics_text += f"Average Distance: {avg_distance:.2f}m\n\n"

            metrics_text += f"FORMATION QUALITY:\n"
            metrics_text += f"  Target Formation Radius: {target_radius:.1f}m\n"
            metrics_text += f"  Average Formation Error: {avg_formation_error:.3f}m\n"
            metrics_text += f"  Formation Integrity: {'GOOD' if avg_formation_error < 0.3 else 'FAIR' if avg_formation_error < 0.6 else 'POOR'}\n\n"

            metrics_text += f"MOTION TYPE: {motion_mode.upper()}\n"
            metrics_text += f"Motion Complexity: {'HIGH' if motion_mode in ['helix', 'polyline'] else 'MEDIUM'}\n\n"

            metrics_text += f"CENTERLINE TRACKING:\n"
            metrics_text += f"  Total Updates: {int(total_baseline_updates)}\n"
            if time_relative.iloc[-1] > 0:
                metrics_text += f"  Update Frequency: {total_baseline_updates / time_relative.iloc[-1]:.3f}/s\n"

            metrics_text += f"\nSTATE DURATIONS:\n"
            for state, duration in state_durations.items():
                metrics_text += f"  {state}: {duration:.1f}s\n"

            metrics_text += f"\nFORMATION CONTROL:\n"
            metrics_text += f"  UAVs in Formation: 4 (1 center + 3 outer)\n"
            metrics_text += f"  Formation Type: Equilateral Triangle\n"
            metrics_text += f"  Control Strategy: Leader-Follower + Rotational\n"

            ax8.text(0.05, 0.95, metrics_text, transform=ax8.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)

            timestamp = self.csv_file.split('_')[-1].split('.')[0]
            plot_filename = f'four_uav_formation_{motion_mode}_{timestamp}.png'
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Formation analysis plot saved as '{plot_filename}'")

            plt.show()

        except FileNotFoundError:
            print(f"Error: {self.csv_file} not found. Run the simulation first.")
        except Exception as e:
            print(f"Error plotting formation trajectories: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function for four UAV formation system"""
    print("=" * 80)
    print("🚁 FOUR UAV FORMATION PURSUIT SYSTEM")
    print("Formation: 1 Center UAV + 3 Outer UAVs in Equilateral Triangle")
    print("Control: Centerline Tracking + MDN Learning + Formation Control")
    print("=" * 80)

    # Available motion modes
    motion_modes = ['helix', 'zigzag', 'linear', 'linear_turn_linear', 'polyline']

    print("Available enemy motion modes:")
    for i, mode in enumerate(motion_modes):
        description = {
            'helix': 'Cylindrical spiral motion',
            'zigzag': 'Oscillating motion with drift',
            'linear': 'Straight line motion',
            'linear_turn_linear': 'Straight-semicircle detour-straight',
            'polyline': 'Multi-segment with sharp corners'
        }
        print(f"  {i + 1}. {mode} - {description[mode]}")

    # Let user choose or default to helix
    try:
        choice = input(f"\nSelect enemy motion mode (1-{len(motion_modes)}) or press Enter for helix: ").strip()
        if choice:
            motion_mode = motion_modes[int(choice) - 1]
        else:
            motion_mode = 'helix'
    except (ValueError, IndexError):
        print("Invalid choice, using helix")
        motion_mode = 'helix'

    print(f"\nStarting Four UAV Formation System with enemy motion: {motion_mode}")
    print("Formation Features:")
    print("  - Center UAV: Centerline tracking + MDN prediction")
    print("  - Outer UAVs: Leader-follower + rotational control")
    print("  - Formation radius: 1.5m (equilateral triangle)")
    print("  - Adaptive control based on prediction uncertainty")

    # Create formation system
    formation_system = FourUAVFormationSystem(
        centerline_model_path='unified_centerline_model.pth',
        history_frames=10,
        prediction_frames=5,
        sample_rate=20,
        enemy_motion_mode=motion_mode
    )

    # Configure UAV parameters
    formation_system.center_uav.max_speed = 6.0
    for uav in formation_system.outer_uavs:
        uav.max_speed = 8.0

    # Start simulation
    formation_system.start()

    try:
        last_status_time = time.time()

        while not formation_system.mission_complete:
            current_time = time.time()

            if current_time - last_status_time >= 5.0:
                status = formation_system.get_status()
                print(f"\n=== Formation Status Report - {motion_mode.upper()} Enemy ===")
                print(f"  Frames collected: {status['frames_collected']}")
                print(f"  Control state: {status['control_state']}")
                print(f"  Training samples: {status['training_samples']}")
                print(f"  Baseline updates: {status['baseline_updates']}")
                print(f"  Intercept phase: {status['intercept_phase']}")
                print(f"  Distance to projection: {status['distance_to_projection']:.2f}m")
                print(f"  Sigma threshold: {status['sigma_threshold']:.4f}")

                motion_info = status['motion_info']
                print(f"  Motion type: {motion_info['motion_mode']}")
                print(f"  Motion time: {motion_info['current_time']:.1f}s")

                # Formation status
                print(f"  Formation distances: {[f'{d:.2f}m' for d in status['formation_distances']]}")
                min_distance = min([np.linalg.norm(status['center_uav_position'] - status['enemy_uav_position'])] +
                                   [np.linalg.norm(pos - status['enemy_uav_position']) for pos in
                                    status['outer_uav_positions']])
                print(f"  Minimum distance to enemy: {min_distance:.2f}m")

                last_status_time = current_time

            time.sleep(1.0)

        final_status = formation_system.get_status()
        if final_status['interception_time'] is not None:
            print(f"\n🎉 {motion_mode.upper()} enemy formation pursuit completed successfully!")
            print(f"   Interception time: {final_status['interception_time']:.2f} seconds")
            print(f"   Total baseline updates: {final_status['baseline_updates']}")
            print(
                f"   Formation maintained with average error: {np.mean(final_status['formation_distances']) - 1.5:.3f}m")
            print(f"   Successfully intercepted {motion_mode} enemy with 4-UAV formation!")
        else:
            print(f"\nMission incomplete")

        print(f"\n" + "=" * 60)
        print(f"FORMATION MISSION COMPLETE - Generating {motion_mode.upper()} Analysis...")
        print("=" * 60)

        # Generate both static analysis and 3D animation
        formation_system.plot_trajectories_from_csv()
        formation_system.create_3d_animation()

    except KeyboardInterrupt:
        print("Interrupted by user")
        print(f"Generating analysis for partial {motion_mode} formation mission...")
        formation_system.plot_trajectories_from_csv()
        formation_system.create_3d_animation()

    finally:
        formation_system.stop()


if __name__ == "__main__":
    main()
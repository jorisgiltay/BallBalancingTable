import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
# Optional: VecNormalize (commented out by default)
# from stable_baselines3.common.vec_env import VecNormalize
import os
import torch
import threading
import time
import subprocess
import sys
import webbrowser
from ball_balance_env import BallBalanceEnv


class RenderToggleCallback(BaseCallback):
    """Callback that allows toggling rendering during training"""

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.render_enabled = False
        self.listener_thread = None
        self.running = True
        self.should_stop = False

    def _on_training_start(self) -> None:
        # Only start keyboard listener if running in a TTY (terminal)
        if sys.stdin.isatty():
            print("\n🎮 Rendering Controls:")
            print("  Press 'r' + Enter to toggle rendering ON/OFF")
            print("  Press 'q' + Enter to quit training")
            print("  Current rendering: OFF (for speed)")
            self.listener_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
            self.listener_thread.start()
        else:
            print("⚠️ Non-interactive environment detected: Render toggle disabled")

    def _keyboard_listener(self):
        """Listen for keyboard input in separate thread"""
        while self.running:
            try:
                key = input().strip().lower()
                if key == 'r':
                    self._toggle_rendering()
                elif key == 'q':
                    print("🛑 Stopping training...")
                    self.should_stop = True
                    break
            except (EOFError, KeyboardInterrupt):
                break

    def _toggle_rendering(self):
        """Toggle rendering mode"""
        self.render_enabled = not self.render_enabled

        # Unwrap to get the original environment
        underlying_env = self.env
        try:
            while hasattr(underlying_env, 'env'):
                underlying_env = underlying_env.env
        except Exception:
            pass

        if self.render_enabled:
            underlying_env.render_mode = "human"
            print("🎬 Rendering ENABLED - You can now see the training!")
        else:
            underlying_env.render_mode = "rgb_array"
            print("⚡ Rendering DISABLED - Training at full speed")

    def _on_training_end(self) -> None:
        self.running = False

    def _on_step(self) -> bool:
        """Called after each step. Return False to stop training."""
        if self.should_stop:
            print("🏁 Training stopped by user request")
            return False
        return True


def _apply_profile(underlying_env: BallBalanceEnv, profile: str):
    """Apply domain randomization and dynamics profile to an env."""
    profile = (profile or "full").lower()
    if profile == "none":
        # Clean environment similar to old setpoint_SAC
        underlying_env.randomize_actuation_bias = False
        underlying_env.randomize_target = False
        underlying_env.target_range = 0.0
        underlying_env.servo_speed_per_step = 0.013
        underlying_env.max_delta_angle = underlying_env.limit_angle / 2
    elif profile == "lite":
        # Medium-lite DR: modest randomization, faster actuation
        underlying_env.randomize_actuation_bias = True
        underlying_env.randomize_target = True
        underlying_env.target_range = 0.04
        underlying_env.act_offset_deg_range = 0.3
        underlying_env.act_gain_range = 0.02
        underlying_env.act_coupling_range = 0.02
        underlying_env.act_speed_jitter = 0.10
        underlying_env.servo_speed_per_step = 0.014
        underlying_env.max_delta_angle = underlying_env.limit_angle / 2
    else:
        # Full DR (leave as set by curriculum/stage)
        pass


def train_rl_agent(use_early_stopping=True, use_curriculum=False, render_training=False, control_freq=60, start_tensorboard=False, seed: int | None = 42, profile: str = "full", finetune_from: str | None = None, finetune_steps: int = 100_000):
    """Train a SAC agent for ball balancing"""

    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    tensorboard_process = None
    if start_tensorboard:
        try:
            print("🚀 Starting TensorBoard...")
            tensorboard_process = subprocess.Popen([
                "tensorboard",
                "--logdir=./tensorboard_logs/",
                "--port=6006",
                "--host=localhost"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(2)  # wait a bit for TensorBoard to start

            try:
                webbrowser.open("http://localhost:6006")
                print("📊 TensorBoard started at http://localhost:6006 (opened in browser)")
            except Exception:
                print("📊 TensorBoard started at http://localhost:6006")
        except FileNotFoundError:
            print("⚠️ TensorBoard not found. Install with: pip install tensorboard")
            print("📊 You can still monitor logs manually: tensorboard --logdir=./tensorboard_logs/")
        except Exception as e:
            print(f"⚠️ Failed to start TensorBoard automatically: {e}")
            print("📊 You can start it manually: tensorboard --logdir=./tensorboard_logs/")

    # Reproducibility
    if seed is not None:
        try:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"🔒 Seeding with seed={seed}")
        except Exception as e:
            print(f"⚠️ Failed to enforce full determinism: {e}")

    # Setup environments
    if render_training:
        env = BallBalanceEnv(render_mode="human", control_freq=control_freq, add_obs_noise=True,
                              use_pid_guidance=True)
        print("🎬 Training with visual rendering enabled")
    else:
        env = BallBalanceEnv(render_mode="rgb_array", control_freq=control_freq, add_obs_noise=True,
                              use_pid_guidance=True)
        print("⚡ Training without visual rendering for maximum speed")

    env = Monitor(env)

    eval_env = BallBalanceEnv(render_mode="rgb_array", control_freq=control_freq)
    eval_env = Monitor(eval_env)

    # Apply requested DR/dynamics profile before model creation
    try:
        underlying_env = env.env
        _apply_profile(underlying_env, profile)
    except Exception:
        pass
    try:
        eval_underlying_env = eval_env.env
        _apply_profile(eval_underlying_env, profile)
    except Exception:
        pass

    # Define or load SAC model
    if finetune_from:
        print(f"🔁 Loading base model for finetune: {finetune_from}")
        model = SAC.load(finetune_from)
        model.set_env(env)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=3,
            ent_coef="auto",
            target_update_interval=1,
            learning_starts=20_000,
            device="auto",
            policy_kwargs=dict(
                net_arch=[64, 64],
                activation_fn=torch.nn.Tanh
            ),
            tensorboard_log="./tensorboard_logs/"
        )

    # Adaptive early stopping threshold
    # With stronger damping/jerk penalties and target tracking, expect ~2.5–3.5 reward/step once stable
    episode_duration = 30  # seconds
    baseline_total = control_freq * episode_duration
    expected_per_step_reward = 3.2
    reward_threshold = 0.85 * expected_per_step_reward * baseline_total

    callbacks = []

    render_toggle_callback = RenderToggleCallback(env, verbose=1)
    callbacks.append(render_toggle_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="ball_balance_checkpoint"
    )
    callbacks.append(checkpoint_callback)

    # Disable early stopping when curriculum is enabled to guarantee reaching hard
    if use_early_stopping and not use_curriculum:
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        print(f"Early stopping threshold set at {reward_threshold:.1f} (~85% of {expected_per_step_reward:.1f} per-step target)")

        eval_callback = EvalCallback(
            eval_env,
            eval_freq=2500,
            best_model_save_path="./models/",
            verbose=1,
            deterministic=True,
            render=False,
            callback_on_new_best=callback_on_best,
            n_eval_episodes=10,
        )
        print("Early stopping enabled - training will stop when reward threshold is met")
    else:
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=2500,
            best_model_save_path="./models/",
            verbose=1,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
        )
        if use_curriculum and use_early_stopping:
            print("Early stopping disabled because curriculum is enabled - training will run for full duration")
        else:
            print("Early stopping disabled - training will run for full duration")

    callbacks.append(eval_callback)

    # Unique TB run name
    log_name = f"SAC_{time.strftime('%Y%m%d-%H%M%S')}"
    print("Starting training...")
    print(f"🧪 TensorBoard run: {log_name}")
    if not start_tensorboard:
        print("📊 Monitor training with: tensorboard --logdir=./tensorboard_logs/")
    print("💾 Checkpoints saved every 10k steps to ./checkpoints/")
    print("🏆 Best models saved to ./models/")

    try:
        # Simple curriculum: ramp difficulty at milestones
        total_steps = 1_200_000 if not finetune_from else finetune_steps
        milestones = [150_000, 450_000, 900_000]
        current_stage = 'easy' if use_curriculum else 'hard'
        if use_curriculum:
            try:
                # Set initial stage
                env.get_attr if False else None  # no-op to avoid linter
                # Access underlying env from Monitor
                underlying_env = env.env
                if hasattr(underlying_env, 'set_curriculum_stage'):
                    underlying_env.set_curriculum_stage('easy')
                    current_stage = 'easy'
                    print("🎓 Curriculum: stage -> easy")
                # Keep eval environment aligned with training stage
                try:
                    eval_underlying_env = eval_env.env
                    if hasattr(eval_underlying_env, 'set_curriculum_stage'):
                        eval_underlying_env.set_curriculum_stage('easy')
                        print("🧪 Eval curriculum aligned: stage -> easy")
                except Exception:
                    pass
            except Exception:
                pass

        # Adaptive curriculum thresholds (as fractions of early-stop threshold)
        # Require sustained hits to avoid switching on noise
        adaptive_required_hits = 2
        adaptive_hits = {"easy": 0, "medium": 0}
        easy_to_medium_threshold = 0.45 * reward_threshold
        medium_to_hard_threshold = 0.70 * reward_threshold

        steps_done = 0
        steps_in_current_stage = 0
        while steps_done < total_steps:
            remaining = total_steps - steps_done
            chunk = min(50_000, remaining)
            model.learn(total_timesteps=chunk, callback=callbacks, progress_bar=True, reset_num_timesteps=False, tb_log_name=log_name)
            steps_done += chunk
            steps_in_current_stage += chunk

            # Stop outer loop if early stopping threshold met during this chunk
            try:
                if hasattr(eval_callback, "best_mean_reward") and eval_callback.best_mean_reward is not None:
                    if eval_callback.best_mean_reward >= reward_threshold:
                        print("🏁 Early stopping threshold reached globally; ending training loop.")
                        break
            except Exception:
                pass

            # Performance-adaptive stage switching using recent eval results
            if use_curriculum and not finetune_from:
                try:
                    # Prefer last_mean_reward when available; fall back to best_mean_reward
                    last_eval_mean = None
                    if 'last_mean_reward' in dir(eval_callback) and getattr(eval_callback, 'last_mean_reward') is not None:
                        last_eval_mean = float(getattr(eval_callback, 'last_mean_reward'))
                    elif 'best_mean_reward' in dir(eval_callback) and getattr(eval_callback, 'best_mean_reward') is not None:
                        last_eval_mean = float(getattr(eval_callback, 'best_mean_reward'))

                    if last_eval_mean is not None:
                        # Track sustained threshold hits for the current stage
                        if current_stage == 'easy' and last_eval_mean >= easy_to_medium_threshold:
                            adaptive_hits['easy'] += 1
                        elif current_stage == 'medium' and last_eval_mean >= medium_to_hard_threshold:
                            adaptive_hits['medium'] += 1
                        else:
                            # Reset counters if we dip below
                            if current_stage == 'easy':
                                adaptive_hits['easy'] = 0
                            elif current_stage == 'medium':
                                adaptive_hits['medium'] = 0

                        # Execute switch when sustained
                        if current_stage == 'easy' and adaptive_hits['easy'] >= adaptive_required_hits:
                            try:
                                underlying_env = env.env
                                if hasattr(underlying_env, 'set_curriculum_stage'):
                                    underlying_env.set_curriculum_stage('medium')
                                    current_stage = 'medium'
                                    adaptive_hits['easy'] = 0
                                    steps_in_current_stage = 0
                                    print(f"🎓 Adaptive curriculum: stage -> medium at {steps_done} steps (last_eval_mean={last_eval_mean:.1f})")
                                # Sync eval stage
                                try:
                                    eval_underlying_env = eval_env.env
                                    if hasattr(eval_underlying_env, 'set_curriculum_stage'):
                                        eval_underlying_env.set_curriculum_stage('medium')
                                        print("🧪 Eval curriculum aligned: stage -> medium")
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        elif current_stage == 'medium' and adaptive_hits['medium'] >= adaptive_required_hits:
                            try:
                                underlying_env = env.env
                                if hasattr(underlying_env, 'set_curriculum_stage'):
                                    underlying_env.set_curriculum_stage('hard')
                                    current_stage = 'hard'
                                    adaptive_hits['medium'] = 0
                                    steps_in_current_stage = 0
                                    print(f"🎓 Adaptive curriculum: stage -> hard at {steps_done} steps (last_eval_mean={last_eval_mean:.1f})")
                                # Sync eval stage
                                try:
                                    eval_underlying_env = eval_env.env
                                    if hasattr(eval_underlying_env, 'set_curriculum_stage'):
                                        eval_underlying_env.set_curriculum_stage('hard')
                                        print("🧪 Eval curriculum aligned: stage -> hard")
                                except Exception:
                                    pass
                            except Exception:
                                pass
                except Exception:
                    pass

            # Fallback: time-based stage switching at milestones (kept for robustness)
            if use_curriculum and not finetune_from and steps_done in milestones and current_stage != 'hard':
                try:
                    underlying_env = env.env
                    if hasattr(underlying_env, 'set_curriculum_stage'):
                        next_stage = 'medium' if current_stage == 'easy' else 'hard'
                        underlying_env.set_curriculum_stage(next_stage)
                        current_stage = next_stage
                        steps_in_current_stage = 0
                        print(f"🎓 Milestone curriculum: stage -> {next_stage} at {steps_done} steps")
                        # Sync eval stage
                        try:
                            eval_underlying_env = eval_env.env
                            if hasattr(eval_underlying_env, 'set_curriculum_stage'):
                                eval_underlying_env.set_curriculum_stage(next_stage)
                                print(f"🧪 Eval curriculum aligned: stage -> {next_stage}")
                        except Exception:
                            pass
                except Exception:
                    pass

        # Final hard-stage finetune to consolidate robustness (skip in finetune-only runs)
        if use_curriculum and current_stage != 'hard' and not finetune_from:
            try:
                underlying_env = env.env
                if hasattr(underlying_env, 'set_curriculum_stage'):
                    underlying_env.set_curriculum_stage('hard')
                    current_stage = 'hard'
                    print("🎯 Forcing final stage: hard (pre-finetune)")
                try:
                    eval_underlying_env = eval_env.env
                    if hasattr(eval_underlying_env, 'set_curriculum_stage'):
                        eval_underlying_env.set_curriculum_stage('hard')
                        print("🧪 Eval curriculum aligned: stage -> hard (pre-finetune)")
                except Exception:
                    pass
            except Exception:
                pass

        if use_curriculum and not finetune_from:
            final_hard_steps = 200_000
            print(f"🔧 Final hard-stage finetune for {final_hard_steps} steps...")
            model.learn(total_timesteps=final_hard_steps, callback=callbacks, progress_bar=True, reset_num_timesteps=False, tb_log_name=log_name)
    finally:
        if tensorboard_process:
            try:
                tensorboard_process.terminate()
                print("🛑 TensorBoard stopped")
            except Exception:
                pass

    # Save with unique name when finetuning to avoid overwriting existing models
    if 'finetune_from' in locals() and finetune_from:
        out_base = f"./models/ball_balance_sac_finetuned_{profile}_{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        out_base = "./models/ball_balance_sac_final"

    model.save(out_base)
    print(f"Training completed! Model saved to {out_base}.zip")

    env.close()
    eval_env.close()

    return model


def test_trained_agent(model_path="./models/ball_balance_sac_final.zip", control_freq=60):
    """Test a trained SAC agent"""

    model = SAC.load(model_path)

    env = BallBalanceEnv(render_mode="human", control_freq=control_freq)
    print(f"Testing with control frequency: {control_freq} Hz")

    print("Testing trained agent. Press Ctrl+C to stop.")

    try:
        for episode in range(10):
            obs, info = env.reset()
        
            episode_reward = 0
            step_count = 0

            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                print("TRAINING/TEST OBS:", obs)
                episode_reward += reward
                step_count += 1

                if terminated or truncated:
                    print(f"Episode {episode + 1}: Steps: {step_count}, Reward: {episode_reward:.2f}, Distance: {info.get('distance_from_center', 0):.3f}")
                    break

    except KeyboardInterrupt:
        print("Testing stopped by user")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test SAC agent for ball balancing")
    parser.add_argument("--mode", choices=["train", "test", "recover"], default="train", help="Train, test, or recover from checkpoint")
    parser.add_argument("--model", default="./models/ball_balance_sac_final.zip", help="Path to model for testing")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping during training")
    parser.add_argument("--resume-from", type=str, help="Resume/finetune from specific model checkpoint (.zip)")
    parser.add_argument("--profile", choices=["none", "lite", "full"], default="full", help="Domain randomization/dynamics profile: none (clean), lite (moderate DR), full (default)")
    parser.add_argument("--finetune-steps", type=int, default=100000, help="Number of steps when finetuning from --resume-from")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering during training (slower)")
    parser.add_argument("--no-render", action="store_true", help="Disable visual rendering during training (faster)")
    parser.add_argument("--freq", type=int, default=60, help="Control frequency in Hz (default: 60)")
    parser.add_argument("--tensorboard", action="store_true", help="Automatically start TensorBoard during training")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (easy→medium→hard)")

    args = parser.parse_args()

    if args.render and args.no_render:
        print("Error: Cannot use both --render and --no-render")
        exit(1)
    elif args.render:
        render_training = True
    elif args.no_render:
        render_training = False
    else:
        render_training = False  # default

    if args.mode == "train":
        # Use internal training/finetuning pipeline with profile support
        train_rl_agent(use_early_stopping=not args.no_early_stop,
                       use_curriculum=args.curriculum,
                       render_training=render_training,
                       control_freq=args.freq,
                       start_tensorboard=args.tensorboard,
                       profile=args.profile,
                       finetune_from=args.resume_from,
                       finetune_steps=args.finetune_steps)
    elif args.mode == "recover":
        from recovery_tool import rollback_to_checkpoint
        rollback_to_checkpoint()
    else:
        test_trained_agent(args.model, control_freq=args.freq)

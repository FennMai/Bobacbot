# Author: ZMAI (based on policies.py by Jimmy Wu)
#
# Keyboard-controlled policy for inference without phone

import threading
import time
from pynput import keyboard
from policies import RemotePolicy


class KeyboardRemotePolicy(RemotePolicy):
    """
    Remote policy controlled by keyboard instead of phone.

    Press SPACE to:
    - Start episode (first press)
    - End episode (second press)
    - Reset environment (third press, then cycle repeats)
    """

    def __init__(self):
        # Initialize keyboard control state
        self.keyboard_state = 'waiting_start'  # waiting_start -> running -> ended -> waiting_reset
        self.space_press_count = 0
        self.space_key_lock = threading.Lock()

        # Call parent __init__ (this starts WebServer and listener threads)
        super().__init__()

        # Start keyboard listener thread
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()

        print("\n" + "="*60)
        print("  Keyboard Control Mode Activated")
        print("="*60)
        print("  Press SPACE to start episode")
        print("="*60 + "\n")

    def _on_key_press(self, key):
        """Callback for keyboard events"""
        try:
            # Check if space key is pressed
            if key == keyboard.Key.space:
                with self.space_key_lock:
                    self._handle_space_press()
        except AttributeError:
            pass  # Special keys handling

    def _handle_space_press(self):
        """Handle space key press based on current state"""
        if self.keyboard_state == 'waiting_start':
            # First press: Start episode
            print("\n[Keyboard] SPACE pressed - Starting episode...")
            self.teleop_state = 'episode_started'
            self.enabled = True  # Enable policy execution
            self.keyboard_state = 'running'
            print("[Keyboard] Episode started! Policy is now running.")
            print("[Keyboard] Press SPACE again to end episode.\n")

        elif self.keyboard_state == 'running':
            # Second press: End episode
            print("\n[Keyboard] SPACE pressed - Ending episode...")
            self.teleop_state = 'episode_ended'
            self.enabled = False  # Disable policy execution
            self.keyboard_state = 'ended'
            print("[Keyboard] Episode ended!")
            print("[Keyboard] Press SPACE to reset environment.\n")

        elif self.keyboard_state == 'ended':
            # Third press: Reset environment
            print("\n[Keyboard] SPACE pressed - Resetting environment...")
            self.teleop_state = 'reset_env'
            self.keyboard_state = 'waiting_reset'
            print("[Keyboard] Environment will reset...")
            print("[Keyboard] Press SPACE to start next episode.\n")

        elif self.keyboard_state == 'waiting_reset':
            # After reset: Ready to start new episode
            print("\n[Keyboard] SPACE pressed - Starting new episode...")
            self.teleop_state = 'episode_started'
            self.enabled = True
            self.keyboard_state = 'running'
            print("[Keyboard] New episode started! Policy is now running.")
            print("[Keyboard] Press SPACE again to end episode.\n")

    def reset(self):
        """
        Override reset to use keyboard instead of phone.
        This is called by run_episode() in main.py
        """
        # Initialize teleop_controller if this is the first reset
        if self.teleop_controller is None:
            from policies import TeleopController
            self.teleop_controller = TeleopController()

        # Reset internal state
        self.teleop_controller.targets_initialized = False
        self.episode_ended = False

        # Reset keyboard state for new episode
        with self.space_key_lock:
            if self.keyboard_state in ['ended', 'waiting_reset']:
                # After first episode, wait for user to start next one
                self.keyboard_state = 'waiting_reset'
                self.teleop_state = None
                print("\n" + "="*60)
                print("  Press SPACE to start next episode")
                print("="*60 + "\n")
            else:
                # First episode
                self.keyboard_state = 'waiting_start'
                self.teleop_state = None

        # Wait for user to press SPACE to start episode
        while self.teleop_state != 'episode_started':
            time.sleep(0.01)

        print("[Keyboard] Connecting to policy server and resetting policy...")

        # Check connection to policy server and reset policy
        # (copied from RemotePolicy.reset(), but without calling super().reset())
        import zmq
        default_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # Temporarily set 1000 ms timeout
        self.socket.send_pyobj({'reset': True})
        try:
            self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
            print("[Keyboard] Policy server reset successful!")
        except zmq.error.Again as e:
            raise Exception('Could not communicate with policy server') from e
        self.socket.setsockopt(zmq.RCVTIMEO, default_timeout)  # Put default timeout back

        # Enable policy execution (already set in _handle_space_press, but ensure it's set)
        self.enabled = True

        print("[Keyboard] Policy is ready! Episode running...\n")

    def close(self):
        """Clean up keyboard listener"""
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()


if __name__ == '__main__':
    # Test the keyboard policy
    import numpy as np
    from constants import POLICY_CONTROL_PERIOD

    print("Testing KeyboardRemotePolicy...")
    print("Make sure policy_server.py is running!")

    obs = {
        'base_pose': np.zeros(3),
        'arm_pos': np.zeros(3),
        'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
        'gripper_pos': np.zeros(1),
        'base_image': np.zeros((640, 360, 3), dtype=np.uint8),
        'wrist_image': np.zeros((640, 480, 3), dtype=np.uint8),
    }

    try:
        policy = KeyboardRemotePolicy()

        # Simulate episode loop
        for episode_num in range(3):
            print(f"\n=== Episode {episode_num + 1} ===")
            policy.reset()

            # Run for a few steps
            for step in range(20):
                action = policy.step(obs)
                if action == 'end_episode':
                    print(f"Episode {episode_num + 1} ended at step {step}")
                    break
                elif action == 'reset_env':
                    print(f"Reset signal received at step {step}")
                    break
                elif action is not None and isinstance(action, dict):
                    print(f"Step {step}: Got action")
                time.sleep(POLICY_CONTROL_PERIOD)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if 'policy' in locals():
            policy.close()
        print("Test finished")

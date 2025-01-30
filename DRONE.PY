import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

class Drone3DEnv(gym.Env):
    def __init__(self):
        super(Drone3DEnv, self).__init__()

        # Initialize PyBullet Simulation
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load the ground plane and the drone
        self.planeId = p.loadURDF("plane.urdf")
        self.droneId = p.loadURDF("samurai.urdf", [0, 0, 1])  # Load a dummy object as a drone

        # Define action space (thrust in X, Y, Z directions)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Define observation space (drone position)
        self.observation_space = spaces.Box(low=np.array([-5, -5, 0]), high=np.array([5, 5, 5]), dtype=np.float32)

        # Set initial position
        self.target_pos = np.array([2, 2, 2])  # Target point in space

    def reset(self, seed=None, options=None):
        # Reset the drone to a random position
        start_pos = np.random.uniform(low=-4, high=4, size=(3,))
        start_pos[2] = 1  # Start slightly above ground
        p.resetBasePositionAndOrientation(self.droneId, start_pos, [0, 0, 0, 1])
        return start_pos, {}

    def step(self, action):
        # Apply action (move the drone)
        current_pos, _ = p.getBasePositionAndOrientation(self.droneId)
        new_pos = np.array(current_pos) + action * 0.1  # Small step movement
        p.resetBasePositionAndOrientation(self.droneId, new_pos, [0, 0, 0, 1])

        # Calculate distance to target
        distance = np.linalg.norm(new_pos - self.target_pos)
        reward = -distance  # Negative reward for being far from target

        # Check if drone reached the target
        done = distance < 0.5

        return new_pos, reward, done, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()


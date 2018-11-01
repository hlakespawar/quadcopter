import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 350
        self.action_high = 750
        self.action_range = self.action_high - self.action_low
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, speeds):
        """Uses current pose of sim to return reward."""
        
        dist = np.linalg.norm(self.sim.pose[:3] - self.target_pos, 2)
        
        penalty = 0
        # penalize distances further than 1 from target; range is [1, inf) before log
        if dist > 1:
            penalty = np.log(dist)
            
        # penalize tilting the quadcopter; range is [-pi/2, pi/2] before tanh
        maxAngle = np.tanh(max(self.sim.pose[3:5] - np.pi)/2)
        
        # constrain the rotor speeds by penalizing large differences; range is [0, 1] before and after tanh
        variance = np.tanh(np.linalg.norm(speeds - np.average(speeds), 1) / (2 * self.action_range))
        
        # penalize high velocities; range is [0, 2] before tanh
        #vel_pen = np.tanh(np.linalg.norm(self.sim.v, 2))
        
        # penalize crashes; range is [-4, 4]
        crash_risk = 2 * np.tanh(self.sim.v[2] / (self.sim.pose[2] + 1))
        
        # give a reward based on distance from the target position; range is [0, 5]
        reward = 5 / (1 + dist)
        
        reward = reward - penalty - maxAngle - variance - crash_risk
        
#         print("dist", dist)
#         print("pen", penalty)
#         print("angle", maxAngle)
#         print("variance", variance)
#         print('crash', crash_risk)
#         print(reward)
#         print('---------')

#         reward = 3 + 0.1 * self.sim.v[2] \
#                  - 0.005 * np.tanh(np.square((self.sim.pose[3:5] - np.pi)/2).sum()) \
#                  - 0.001 * np.tanh(np.square(self.sim.angular_v).sum()) 

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
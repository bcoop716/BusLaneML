import gym
from gym import spaces
import numpy as np
import traci
import sumolib
import random

class SumoBusLaneEnv(gym.Env):
    def __init__(self):
        super(SumoBusLaneEnv, self).__init__()

        # Define action space: 0 = keep bus-only, 1 = allow passenger cars
        self.action_space = spaces.Discrete(2)

        # Define observation space (example: 10 values like vehicle counts/wait times)
        self.observation_space = spaces.Box(low=0, high=30000, shape=(10,), dtype=np.float32)

        # Track total metrics
        self.total_distance = 0
        self.total_wait_time = 0
        self.total_passengers = 0
        
        self.step_length = 0.1
        self.sumo_cmd =["sumo-gui", "-c", "SUMO_bus_lanes.sumocfg", "--delay", "200", "--start", "--step-length", str(self.step_length)]

        self.step_count = 0

    def reset(self):
        try:
            traci.start(self.sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            return None

        self.step_count = 0

        # Reset accumulated metrics
        self.total_distance = 0
        self.total_wait_time = 0
        self.total_passengers = 0

        # Initialize vehicle tracking dictionary
        self.vehicle_data = {}

        return self._get_obs()
    
    def all_vehicles_arrived(self):
# Get all vehicles in the simulation
        all_vehicles = traci.vehicle.getIDList()

        # Get vehicles that have already arrived
        arrived_vehicles = traci.simulation.getArrivedIDList()

        # Check if the number of arrived vehicles matches the number of all vehicles
        if len(all_vehicles) == len(arrived_vehicles):
            return True  # All vehicles have arrived
        else:
            return False  # Some vehicles haven't arrived yet
        
    def step(self, action):
        try:
            self._apply_action(action)
            for vehicle_id in traci.vehicle.getIDList():
                if vehicle_id not in self.vehicle_data:
                    self.vehicle_data[vehicle_id] = {
                        "route_length": 0,
                        "waiting_time": 0,
                        "passenger_count": 0
                    }

                self.vehicle_data[vehicle_id]["route_length"] = traci.vehicle.getDistance(vehicle_id)
                self.vehicle_data[vehicle_id]["waiting_time"] = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                self.vehicle_data[vehicle_id]["passenger_count"] = traci.vehicle.getPersonNumber(vehicle_id)
            traci.simulationStep()  # Step through the simulation
            print(self.step_count)
            self.step_count += 1

            # Collect and accumulate metrics for vehicles that finished their journey
            self._collect_metrics()

            obs = self._get_obs()
            reward = self._get_reward()
            if(self.step_count >= 30000):
                # print("max steps hit")
                done = True
            elif(self.all_vehicles_arrived()):
                # print("all vehicles done")
                done = True
            else:
                done = False
            

            return obs, reward, done, {}
        except Exception as e:
            print(f"Error at step {self.step_count}: {e}")
            return None, 0, True, {}

    def _apply_action(self, action):
        # Get bus lanes and apply the action (0 = bus-only, 1 = allow passenger cars)
        bus_lanes = self.get_bus_lanes()
        print(action)
        allowed = ["bus", "passenger"] if action == 1 else ["bus"]
        for lane in bus_lanes:
            traci.lane.setAllowed(lane, allowed)
        # allowed = ["bus"]
        # for lane in bus_lanes:
        #     traci.lane.setAllowed(lane, allowed)

    def _get_obs(self):
        # Return occupancy or waiting times from specific lanes
        lane_ids = self.get_bus_lanes()
        obs = []
        for lane_id in lane_ids[:10]:  # Pick up to 10 lanes for fixed shape
            obs.append(traci.lane.getLastStepVehicleNumber(lane_id))
        obs += [0] * (10 - len(obs))  # Pad if fewer than 10
        return np.array(obs, dtype=np.float32)

    def _get_reward(self):
        # Example reward: negative total waiting time (you want to minimize it)
        total_wait = sum(
            traci.vehicle.getAccumulatedWaitingTime(v) for v in traci.vehicle.getIDList()
        )
        return -total_wait

    def _collect_metrics(self):
        arrived_vehicles = traci.simulation.getArrivedIDList()

        if not arrived_vehicles:
            print(f"Step {self.step_count}: No vehicles arrived yet.")
        
        for vehicle_id in arrived_vehicles:
            if vehicle_id in self.vehicle_data:
                vehicle_info = self.vehicle_data[vehicle_id]

                route_length = vehicle_info.get("route_length", 0)
                waiting_time = vehicle_info.get("waiting_time", 0)
                passenger_count = vehicle_info.get("passenger_count", 0)

                self.total_distance += route_length
                self.total_wait_time += waiting_time
                self.total_passengers += passenger_count
            else:
                print(f"Vehicle {vehicle_id} arrived but had no pre-tracked data.")

    def close(self):
        # Close the SUMO simulation when done
        traci.close()

    def get_bus_lanes(self):
        # Get bus lanes (can be extracted as shown in the earlier part)
        bus_lanes = []
        network = sumolib.net.readNet("SUMO_bus_lanes.net.xml")
        for edge in network.getEdges():
            for lane in edge.getLanes():
                if "bus" in lane._allowed and "passenger" not in lane._allowed:
                    bus_lanes.append(lane.getID())
        return bus_lanes
   

# Example of initializing and running the environment
if __name__ == "__main__":
    env = SumoBusLaneEnv()
    done = False
    state = env.reset()

    while not done:
        action = env.action_space.sample()  # Random action, replace with your agent's decision
        next_state, reward, done, _ = env.step(action)

    # At the end, print out the accumulated metrics
    print(f"Total distance traveled: {env.total_distance}")
    print(f"Total waiting time: {env.total_wait_time}")
    print(f"Total passengers: {env.total_passengers}")
    print(f"Closing")
    env.close()
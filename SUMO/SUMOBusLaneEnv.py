import gym
from gym import spaces
import numpy as np
import traci
import sumolib
import os

class SumoBusLaneEnv(gym.Env):
    def __init__(self):
        super(SumoBusLaneEnv, self).__init__()

        self.action_space = spaces.Discrete(2) 
        self.observation_space = spaces.Box(low=0, high=30000, shape=(10,), dtype=np.float32)

        self.step_length = 0.1
        self.sumo_cmd = ["sumo", "-c", "low_demand.sumocfg", "--start", "--step-length", str(self.step_length)]
        self.step_count = 0
        self.dispatch_interval = 30  
        self.count = 0
        self.buscount = 0
        self.carcount = 0
        self.total_distance = 0
        self.total_wait_time = 0
        self.total_passengers = 0
        self.vehicle_data = {}

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
        self.buscount = 0
        self.carcount = 0
        self.count = 0
        self.step_count = 0
        self.total_distance = 0
        self.total_wait_time = 0
        self.total_passengers = 0
        self.vehicle_data = {}
        
        return self._get_obs()

    def step(self, action):
        #action = 1
        self._apply_action(action)
        
        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id not in self.vehicle_data:
                self.vehicle_data[vehicle_id] = {
                    "route_length": 0,
                    "waiting_time": 0,
                    "passenger_count": 0,
                    "typeofv": "",
                }

            self.vehicle_data[vehicle_id]["route_length"] = traci.vehicle.getDistance(vehicle_id)
            self.vehicle_data[vehicle_id]["waiting_time"] = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            self.vehicle_data[vehicle_id]["passenger_count"] = traci.vehicle.getPersonNumber(vehicle_id)
            self.vehicle_data[vehicle_id]["typeofv"] = traci.vehicle.getTypeID(vehicle_id)

        traci.simulationStep()
        self.step_count += 1

        self._collect_metrics()

        obs = self._get_obs()
        reward = self._get_reward()

        done = self.step_count >= 30000 or self.all_vehicles_arrived()

        return obs, reward, done, {}

    def close(self):
        if traci.isLoaded():
            traci.close()

    def _apply_action(self, action):
        if self.step_count % 300 == 0: 
            bus_lanes = self.get_bus_lanes()
            allowed = ["bus", "passenger"] if action == 1 else ["bus"]
            for lane in bus_lanes:
                traci.lane.setAllowed(lane, allowed)

    def _get_obs(self):
        lane_ids = self.get_bus_lanes()
        obs = []
        for lane_id in lane_ids[:10]:
            obs.append(traci.lane.getLastStepVehicleNumber(lane_id))
        obs += [0] * (10 - len(obs))
        return np.array(obs, dtype=np.float32)

    def _get_reward(self):
        total_wait = sum(
            traci.vehicle.getAccumulatedWaitingTime(v) for v in traci.vehicle.getIDList()
        )
        avg_speed = np.mean([traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]) if traci.vehicle.getIDList() else 0
        return -total_wait + 10 * avg_speed

    def _collect_metrics(self):
        arrived_vehicles = traci.simulation.getArrivedIDList()
        for vehicle_id in arrived_vehicles:
            self.count = self.count + 1 
             #   self.buscount = self.buscount + 1
            #if traci.vehicle.getVehicleClass(vehicle_id) == CAR_CLASS_ID:
             #       self.carcount = self.carcount + 1
            #vehicle_class = traci.vehicle.getVehicleClass(vehicle_id)
            #print(f"Vehicle ID: {vehicle_id}, Class ID: {vehicle_class}")
            if vehicle_id in self.vehicle_data:                
                vehicle_info = self.vehicle_data[vehicle_id]
                self.total_distance += vehicle_info.get("route_length", 0)
                self.total_wait_time += vehicle_info.get("waiting_time", 0)
                self.total_passengers += vehicle_info.get("passenger_count", 0)
            #if traci.vehicle.getIDList() and vehicle_id in traci.vehicle.getIDList():
                #vehicle_class = traci.vehicle.getTypeID(vehicle_id)
                vehicle_class = vehicle_info.get("typeofv")
                #self.count = self.count + 1
                #print(f"Vehicle ID: {vehicle_id}, Class ID: {vehicle_class}")
                if vehicle_class == "bus":
                    self.buscount += 1
                elif vehicle_class == "car":
                    self.carcount += 1
    def get_bus_lanes(self):
        bus_lanes = []
        network = sumolib.net.readNet("SUMO_bus_lanes.net.xml")
        for edge in network.getEdges():
            for lane in edge.getLanes():
                if "bus" in lane._allowed and "passenger" not in lane._allowed:
                    bus_lanes.append(lane.getID())
        return bus_lanes

    def all_vehicles_arrived(self):
        return len(traci.vehicle.getIDList()) == 0

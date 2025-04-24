import traci
import sumolib

# returns list of all lane IDs that allow buses
def get_bus_lanes():
    bus_lanes = []
    network = sumolib.net.readNet("SUMO_bus_lanes.net.xml")

    for edge in network.getEdges():
        for lane in edge.getLanes():
            if "bus" in lane._allowed and "passenger" not in lane._allowed:
                bus_lanes.append(lane.getID())
    return bus_lanes

# allowed_vehicles is a list of str: ["bus", "passenger"]
def switch_lane_permission(lane_id, allowed_vehicles):
    traci.lane.setAllowed(lane_id, allowed_vehicles)

def reroute_restricted_vehicles():
    print(bus_lanes)
    for vehicle_id in traci.vehicle.getIDList():
        vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
        vehicle_type = traci.vehicle.getVehicleClass(vehicle_id)

        if vehicle_lane in bus_lanes and vehicle_type != "bus":
            try:
                traci.vehicle.rerouteTraveltime(vehicle_id)
            except:
                print("Could not reroute ", vehicle_id)

def main():
    sumoBinary = "sumo-gui"
    step_length = 0.1
    sumoCmd = [sumoBinary, "-c", "SUMO_bus_lanes.sumocfg", "--delay", "200", "--start", "--step-length", str(step_length)]
    traci.start(sumoCmd)
    bus_lanes = get_bus_lanes()

    step = 0
    while step < 500:
        if step >= 100.0 and step <= 101.0:
            print("switching lane permissions")
            for lane in bus_lanes:
                switch_lane_permission(lane, ["bus", "passenger"])
        elif step >= 200.0 and step <= 201.0:
            print("bus lanes are back")
            for lane in bus_lanes:
                switch_lane_permission(lane, ["bus"])
        traci.simulationStep()
        step += step_length
    traci.close()

bus_lanes = []
main()
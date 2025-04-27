import traci
import os
import time

# Define the SUMO configuration file (update this to your actual configuration file path)
sumo_cfg = "SUMO_bus_lanes.sumocfg"  # Replace with your SUMO configuration file path
sumo_port = 5000  # Use a custom port to avoid conflicts (can change if needed)

def check_sumo_connection():
    try:
        # Start SUMO using the specific configuration and port
        sumo_cmd = [
            "sumo", "-c", sumo_cfg, "--start", "--remote-port", str(sumo_port)
        ]

        # Start the SUMO simulation (this should start the simulation in the background)
        traci.start(sumo_cmd)
        print("Successfully connected to SUMO simulation!")

        # Run a few simulation steps to ensure the connection works
        for step in range(10):
            traci.simulationStep()  # Step through the simulation
            print(f"Step {step + 1} completed")

        # Close the simulation connection
        traci.close()
        print("Connection closed successfully")

    except Exception as e:
        print(f"Error connecting to SUMO: {e}")

if __name__ == "__main__":
    # Check the SUMO connection
    check_sumo_connection()
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Load Saved Data
# ===========================
rewards = np.load('rewards.npy')
distances = np.load('distances.npy')
waiting_times = np.load('waiting_times.npy')
passengers = np.load('passengers.npy')
bus_counts = np.load('bus_counts.npy')
car_counts = np.load('car_counts.npy')
vehicle_counts = np.load('vehicle_counts.npy')

# ===========================
# Set Baseline Values (From your run.py results)
# ===========================
# Update these manually if needed!
baseline_distance = 850000  # Example baseline
baseline_waiting_time = 680
baseline_passengers = 100
baseline_total_vehicles = 300

# ===========================
# Helper Function: Moving Average
# ===========================
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ===========================
# Plot 1: Total Reward vs Episode
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(rewards, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward vs Episode')
plt.legend()
plt.grid()
plt.savefig('reward_vs_episode.png')
plt.show()

# ===========================
# Plot 2: Moving Average Reward
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(moving_average(rewards), label='Moving Average Reward (Window=10)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Moving Average of Reward')
plt.legend()
plt.grid()
plt.savefig('moving_avg_reward_vs_episode.png')
plt.show()

# ===========================
# Plot 3: Total Distance vs Episode
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(distances, label='Total Distance per Episode')
plt.xlabel('Episode')
plt.ylabel('Distance Traveled')
plt.title('Distance Traveled vs Episode')
plt.legend()
plt.grid()
plt.savefig('distance_vs_episode.png')
plt.show()

# ===========================
# Plot 4: Total Waiting Time vs Episode
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(waiting_times, label='Total Waiting Time per Episode', color='orange')
plt.xlabel('Episode')
plt.ylabel('Waiting Time')
plt.title('Waiting Time vs Episode')
plt.legend()
plt.grid()
plt.savefig('waiting_time_vs_episode.png')
plt.show()

# ===========================
# Plot 5: Total Passengers Transported vs Episode
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(passengers, label='Passengers Transported per Episode', color='green')
plt.xlabel('Episode')
plt.ylabel('Passengers')
plt.title('Passengers Transported vs Episode')
plt.legend()
plt.grid()
plt.savefig('passengers_vs_episode.png')
plt.show()

# ===========================
# Plot 6: Bus, Car, Vehicle Counts vs Episode
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(bus_counts, label='Buses', color='purple')
plt.plot(car_counts, label='Cars', color='red')
plt.plot(vehicle_counts, label='Total Vehicles', color='blue')
plt.xlabel('Episode')
plt.ylabel('Count')
plt.title('Vehicle Counts vs Episode')
plt.legend()
plt.grid()
plt.savefig('vehicle_counts_vs_episode.png')
plt.show()

# ===========================
# Plot 7: Baseline vs DQL Comparison (Bar Chart)
# ===========================
final_distance = distances[-1]
final_waiting_time = waiting_times[-1]
final_passengers = passengers[-1]
final_total_vehicles = vehicle_counts[-1]

categories = ['Distance', 'Waiting Time', 'Passengers', 'Total Vehicles']
baseline_values = [baseline_distance, baseline_waiting_time, baseline_passengers, baseline_total_vehicles]
dql_values = [final_distance, final_waiting_time, final_passengers, final_total_vehicles]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, baseline_values, width, label='Baseline')
plt.bar(x + width/2, dql_values, width, label='DQL Agent')
plt.ylabel('Value')
plt.title('Baseline vs DQL Performance Comparison')
plt.xticks(x, categories)
plt.legend()
plt.grid(axis='y')
plt.savefig('baseline_vs_dql_comparison.png')
plt.show()

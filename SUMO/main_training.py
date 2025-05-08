import sys
import os
print(os.getcwd())
print(sys.executable)

from dqn_agent import DQNAgent
from SUMOBusLaneEnv import SumoBusLaneEnv
import torch
import matplotlib.pyplot as plt
import numpy as np

def train():
    num_episodes = 300  # More episodes for better training
    max_steps = 6001
    state_size = 10
    action_size = 2

    env = SumoBusLaneEnv()
    agent = DQNAgent(state_size, action_size)

    all_rewards = []
    all_distances = []
    all_waiting_times = []
    all_passengers = []
    bus_counts = []
    car_counts = []
    total_vehicle_counts = []


    baseline = 0
    number = 0
    baseline = int(input("Enter 0 for DQL or 1 for baseline options: "))
    if(baseline != 0):
        number = int(input("Enter an 0 for Buses only. Enter 1 for Mixed Traffic: "))
        if(number == 0):
            b0 = 0
        else:
            b0 = 1
    else:
        b0 = 3

    for episode in range(num_episodes):
        state = env.reset()

        
        episode_buscount = 0
        episode_carcount = 0
        episode_vehiclecount = 0
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(state)
            if(b0 == 0):
                next_state, reward, done, _ = env.step(0)
            # next_state, reward, done, _ = env.step(action)
            if(b0 == 1):
                next_state, reward, done, _ = env.step(1)
                
            if(b0 == 3):
                next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward
            steps += 1

        # After episode finished
        episode_buscount = env.buscount
        episode_carcount = env.carcount
        episode_vehiclecount = env.count

        print()
        print(f"Episode {episode+1}/{num_episodes}")
        print(f"Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.4f}")
        print(f"Total Distance Traveled: {env.total_distance:.2f}")
        print(f"Total Waiting Time: {env.total_wait_time:.2f}")
        print(f"Total Buses: {episode_buscount}")
        print(f"Total Cars: {episode_carcount}")
        print(f"Total Vehicles: {episode_vehiclecount}")
        print()

        all_rewards.append(total_reward)
        all_distances.append(env.total_distance)
        all_waiting_times.append(env.total_wait_time)
        all_passengers.append(env.total_passengers)
        bus_counts.append(episode_buscount)
        car_counts.append(episode_carcount)
        total_vehicle_counts.append(episode_vehiclecount)

        #  Decay epsilon after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Save model every 10 episodes
        if (episode + 1) % 10 == 0:
            torch.save(agent.policy_net.state_dict(), f"dqn_model_episode_{episode+1}.pth")
            print(f"Model saved at episode {episode+1}")

    env.close()

    # Save final model
    torch.save(agent.policy_net.state_dict(), "dqn_model_final.pth")
    print("Final model saved!")

    #  Save episode data for plotting separately later
    np.save('rewards.npy', np.array(all_rewards))
    np.save('distances.npy', np.array(all_distances))
    np.save('waiting_times.npy', np.array(all_waiting_times))
    np.save('passengers.npy', np.array(all_passengers))
    np.save('bus_counts.npy', np.array(bus_counts))
    np.save('car_counts.npy', np.array(car_counts))
    np.save('vehicle_counts.npy', np.array(total_vehicle_counts))

    print("Training statistics saved!")

    #  Plot Total Reward Progress
    plt.figure(figsize=(10, 6))
    plt.plot(all_rewards, label="Total Reward per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Deep Q-Learning - Reward Progress')
    plt.legend()
    plt.grid()
    plt.savefig('reward_vs_episode.png')
    plt.show()

if __name__ == "__main__":
    
    train()

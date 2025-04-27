from dqn_agent import DQNAgent
from SumoBusLaneEnv import SumoBusLaneEnv  
import torch

def train():
    num_episodes = 100  # Can increase to 500-1000 for better training
    max_steps = 3000     # Maximum steps per episode
    state_size = 10      # 10 lanes
    action_size = 2      # 0 = keep bus-only, 1 = allow passengers

    env = SumoBusLaneEnv()
    agent = DQNAgent(state_size, action_size)

    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward
            steps += 1

        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")
        all_rewards.append(total_reward)

    env.close()

    # Optionally, save trained model
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("Training finished and model saved!")

if __name__ == "__main__":
    train()

# Intro to Machine Learning – Group 15

## Group Members
- Brandon Cooper  
- Gredrick Dillistone  
- Raghav Doshi  
- Caden Thompson  
- Danny Wang  

---

## 📦 Project Overview

This repository contains our final project for the **Intro to Machine Learning** course. The project showcases practical implementations of core ML concepts including:
- Data preprocessing
- Model training and evaluation
- Visualization of results
- Custom model tuning and interpretation

## 🗂️ Project Structure
├── SUMOBusLaneEnv.py<br>            # Custom SUMO Gym-like environment
├── dqn_agent.py<br>                  # DQN agent implementation
├── main_training.py              # Main training loop for RL
├── analyze_results.py            # Evaluation and visualization of agent performance
├── run.py                        # Testing/inference script to run trained models
│
├── high_demand.*                 # SUMO configs for high traffic demand
├── low_demand.*                  # SUMO configs for low traffic demand
├── high_flow.xml                 # Vehicle flow settings for high demand
├── low_flow.xml                  # Vehicle flow settings for low demand
│
├── SUMO_bus_lanes.*              # Base network and simulation configs (net, rou, sumocfg)
├── *.rou.alt.xml                 # Alternate route configuration files
├── *.IDZone.Identifier           # Optional identifier files for routing/zoning
│
└── requirements.txt              # Python dependencies

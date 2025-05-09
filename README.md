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
├── SUMOBusLaneEnv.py            # Custom SUMO Gym-like environment<br>
├── dqn_agent.py                  # DQN agent implementation<br>
├── main_training.py              # Main training loop for RL<br>
├── analyze_results.py            # Evaluation and visualization of agent performance<br>
├── run.py                        # Testing/inference script to run trained models<br>
│<br><br>
├── high_demand.*                 # SUMO configs for high traffic demand<br>
├── low_demand.*                  # SUMO configs for low traffic demand<br>
├── high_flow.xml                 # Vehicle flow settings for high demand<br>
├── low_flow.xml                  # Vehicle flow settings for low demand<br>
│<br><br>
├── SUMO_bus_lanes.*              # Base network and simulation configs (net, rou, sumocfg)<br>
├── *.rou.alt.xml                 # Alternate route configuration files<br>
├── *.IDZone.Identifier           # Optional identifier files for routing/zoning<br>
│<br><br>
└── requirements.txt              # Python dependencies<br>

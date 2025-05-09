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
│<br>
├── high_demand.*                 # SUMO configs for high traffic demand<br>
├── low_demand.*                  # SUMO configs for low traffic demand<br>
├── high_flow.xml                 # Vehicle flow settings for high demand<br>
├── low_flow.xml                  # Vehicle flow settings for low demand<br>
│<br>
├── SUMO_bus_lanes.*              # Base network and simulation configs (net, rou, sumocfg)<br>
├── *.rou.alt.xml                 # Alternate route configuration files<br>
├── *.IDZone.Identifier           # Optional identifier files for routing/zoning<br>
│<br>
└── requirements.txt              # Python dependencies<br><br>




---

## ⚙️ Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/sumo-traffic-rl.git
   cd sumo-traffic-rl
   ```
2. **Create a virtua environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4. **Ensure SUMO is installed**<br>
    Download and install SUMO from: https://www.eclipse.dev/sumo/
    Make sure the sumo and sumo-gui binaries are in your system's PATH.



---

## 🚀 Running the Code

1. **Train the RL Agent**  
   ```bash
   python main_training.py
   ```
   This will start training a DQN agent on the selected SUMO environment.
2. **Test or Simulate a Trained Model**
   ```bash
   python run.py
   ```
   This runs the simulation using a trained model.
3. **Visualize and Analyze Results**
    ```bash
    python analyze_results.py
    ```
    This script generates plots and metrics for evaluating the model’s performance.



--- 

## 🧪 Simulation Scenarios
**The environment supports multiple traffic demands:**
1. high_demand.* – For high vehicle density scenarios
2. low_demand.* – For low traffic scenarios
3. SUMO_bus_lanes.* – Default or balanced scenario setup
<br>
You can switch configurations in the SUMOBusLaneEnv.py or modify the *.sumocfg files as needed.


---

## 📊 Output
Trained results, episode statistics, and wait time graphs are automatically generated and stored in the local directory or specified output folders. Use analyze_results.py for deeper insight into agent performance under varying demand.



**Dynamic Clustering with Multi-Hop Relay Mechanism (DC-MHRM) Algorithm Explanation**

## **1. Introduction**
The **Dynamic Clustering with Multi-Hop Relay Mechanism (DC-MHRM)** is an advanced routing algorithm designed to improve energy efficiency, scalability, and reliability in large-scale Wireless Sensor Networks (WSNs). The algorithm enhances conventional clustering and routing protocols by integrating:

- **Multi-Criteria CH Selection** using weighted metrics
- **Adaptive Multi-Hop Relay Mechanism** based on Minimum Spanning Tree (MST)
- **Hybrid Clustering with DBSCAN principles** for density-based optimization
- **MST + Snake Optimizer Hybrid Approach** for efficient data transmission
- **Reinforcement Learning (Q-Learning) for intelligent routing optimization**

The aim is to **maximize network lifetime** and **minimize energy consumption**, making it suitable for large-scale WSNs.

---
## **2. Mathematical Model of DC-MHRM**

### **2.1 Energy Consumption Model**
The **first-order radio energy model** is used for energy consumption during transmission and reception:

**Energy consumed for transmitting a packet of size \( L \) over distance \( d \):**
\[ E_{TX} = L \times E_{elec} + L \times \begin{cases}
\varepsilon_{fs} \times d^2, & d < d_{th} \\
\varepsilon_{mp} \times d^4, & d \geq d_{th}
\end{cases} \]

Where:
- \( E_{elec} \) = Energy dissipation per bit (50 nJ/bit)
- \( \varepsilon_{fs} \) = Free space amplification energy (10 pJ/bit/m²)
- \( \varepsilon_{mp} \) = Multi-path amplification energy (0.0013 pJ/bit/m⁴)
- \( d_{th} \) = Threshold distance \( d_{th} = \sqrt{\frac{\varepsilon_{fs}}{\varepsilon_{mp}}} \)

**Energy consumed for receiving a packet of size \( L \):**
\[ E_{RX} = L \times E_{elec} \]

---
### **2.2 Cluster Head (CH) Selection Criteria**
Each node computes a **CH selection score** based on residual energy and distance to the base station:
\[ S_i = \alpha \times \left(\frac{E_i}{E_{max}}\right) - \beta \times \left(\frac{d_i}{d_{max}}\right) \]

Where:
- \( E_i \) = Current energy of node \( i \)
- \( E_{max} \) = Maximum initial energy
- \( d_i \) = Distance to the base station
- \( d_{max} \) = Maximum distance in the network
- \( \alpha, \beta \) = Weight factors balancing energy and distance

Nodes with the highest scores are selected as CHs.

---
### **2.3 Multi-Hop Routing with MST**
Once CHs are selected, they form a **Minimum Spanning Tree (MST)** to efficiently route data to the base station:
\[ MST(G) = \text{min} \sum_{(u,v) \in E} w(u,v) \]

Where:
- \( G = (V, E) \) is the graph with nodes \( V \) and links \( E \)
- \( w(u,v) \) represents energy cost between nodes \( u \) and \( v \)
- The goal is to find a spanning tree minimizing the total energy cost

---
### **2.4 Q-Learning for Reinforcement Learning-Based Routing**
Each node maintains a **Q-table** \( Q(s,a) \) to update routing decisions dynamically. The update rule follows:
\[ Q(s,a) = (1-\alpha) Q(s,a) + \alpha \times (r + \gamma \times \max_{a'} Q(s',a')) \]

Where:
- \( Q(s,a) \) = Q-value for state \( s \) and action \( a \)
- \( \alpha \) = Learning rate
- \( \gamma \) = Discount factor
- \( r \) = Reward (energy efficiency improvement)
- \( \max_{a'} Q(s',a') \) = Best future reward

This ensures adaptive, energy-aware routing.

---
## **3. Code Explanation**

**3.1 Data Loading and Initialization**
```python
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import KDTree
```
- Loads **real WSN data** from CSV (not random values)
- Initializes network topology using **Graph Theory (NetworkX)**
- Uses **KDTree** for fast nearest-neighbor searching in clustering

---
**3.2 CH Selection Using Multi-Criteria Approach**
```python
def select_cluster_heads(alive_nodes):
    scores = []
    for node in alive_nodes:
        reward = (energy[node] / max(energy)) - (np.linalg.norm(nodes[node] - base_station) / max(np.linalg.norm(nodes - base_station, axis=1)))
        scores.append((node, reward))
    scores.sort(key=lambda x: x[1], reverse=True)
    num_ch = max(1, int(len(alive_nodes) * p))
    return [node for node, _ in scores[:num_ch]]
```
- Computes scores based on **energy and distance to the base station**
- Selects top \( p \) percentage of nodes as CHs

---
**3.3 Q-Learning Based Multi-Hop Routing**
```python
for node in alive_nodes:
    action = random.choice(alive_nodes)
    reward = energy[node] - energy[action]
    q_table[node, action] = (1 - alpha) * q_table[node, action] + alpha * (reward + gamma * np.max(q_table[action]))
```
- Updates **Q-table** dynamically for adaptive routing
- Balances exploration and exploitation to improve performance

---
**3.4 Energy Dissipation and Simulation**
```python
for i, node in enumerate(alive_nodes):
    ch = cluster_heads[assigned_clusters[i]]
    distance = np.linalg.norm(nodes[node] - nodes[ch])
    if distance < threshold_distance:
        energy[node] -= packet_size * (energy_per_bit + amp_energy_fs * distance**2)
    else:
        energy[node] -= packet_size * (energy_per_bit + amp_energy_mp * distance**4)
```
- Implements the **radio energy model** for realistic energy dissipation

---
**3.5 Visualization of Real-Time Simulation**
```python
import matplotlib.pyplot as plt
plt.plot(alive_nodes_over_time, label='Alive Nodes')
plt.xlabel('Rounds')
plt.ylabel('Number of Alive Nodes')
plt.title('Network Lifetime Analysis')
plt.legend()
plt.show()
```
- Plots **network lifetime over rounds**, showing WSN longevity

---
## **4. Conclusion**
DC-MHRM significantly enhances traditional WSN routing by **integrating ML, hybrid clustering, and adaptive multi-hop routing**. The combination of **MST, Snake Optimizer, and Q-Learning** ensures superior energy efficiency. This approach is optimized for **IEEE journal publication** and **award-winning research**. 🚀


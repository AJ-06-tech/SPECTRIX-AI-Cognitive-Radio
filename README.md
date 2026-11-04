\# 📡 SPECTRIX

\### AI-Based Cognitive Radio Spectrum Management for Dynamic Wireless Networks



<p align="center">

&#x20; <strong>An intelligent spectrum management framework combining Cooperative Spectrum Sensing, Hidden Markov Models, and Fuzzy Logic for adaptive channel allocation.</strong>

</p>



<p align="center">

&#x20; <img src="https://img.shields.io/badge/Python-3.13.1-blue?logo=python\&logoColor=white" alt="Python">

&#x20; <img src="https://img.shields.io/badge/NumPy-2.5.4-013243?logo=numpy" alt="NumPy">

&#x20; <img src="https://img.shields.io/badge/Pandas-3.1.0-150458?logo=pandas" alt="Pandas">

&#x20; <img src="https://img.shields.io/badge/Matplotlib-3.11.2-11557c" alt="Matplotlib">

&#x20; <img src="https://img.shields.io/badge/Scikit--Fuzzy-0.5.0-orange" alt="Scikit-fuzzy">

&#x20; <img src="https://img.shields.io/badge/Project-Final%20Year%20Project-success" alt="Project">

</p>



\---



\## 📑 Table of Contents



\- \[Overview](#-overview)

\- \[Problem Statement](#-problem-statement)

\- \[Objectives](#-project-objectives)

\- \[System Architecture](#-system-architecture)

\- \[How SPECTRIX Works](#-how-spectrix-works)

\- \[Proposed ASSM Algorithm](#-proposed-assm-algorithm)

\- \[Technology Stack](#-technology-stack)

\- \[Simulation Configuration](#-simulation-configuration)

\- \[Experimental Methodology](#-experimental-methodology)

\- \[Performance Results](#-performance-results)

\- \[Visual Results](#-visual-results)

\- \[Project Structure](#-project-structure)

\- \[Installation and Setup](#-installation-and-setup)

\- \[Running the Project](#-running-the-project)

\- \[Limitations](#-limitations)

\- \[Future Scope](#-future-scope)

\- \[Conclusion](#-conclusion)



\---



\## 🔍 Overview



\*\*SPECTRIX\*\* is a Python-based cognitive radio spectrum management simulation framework developed to improve channel selection and reduce transmission collisions in dynamic wireless networks.



The system integrates multiple intelligent techniques:



\- \*\*Cooperative Spectrum Sensing:\*\* Combines sensing reports from multiple secondary users to determine channel availability.

\- \*\*Hidden Markov Model (HMM):\*\* Predicts future channel occupancy using historical channel-state information.

\- \*\*Fuzzy Logic:\*\* Evaluates channel suitability using signal-to-noise ratio, occupancy, and interference.

\- \*\*Adaptive Spectrum Selection and Management (ASSM):\*\* Selects suitable channels using the combined sensing, prediction, and channel-quality information.



The proposed ASSM approach is evaluated against a Standard First-Available Channel Allocation baseline using common simulation scenarios.



> \[!NOTE]

> SPECTRIX uses synthetic simulation data. The reported results represent simulated network behavior and are not measurements from a physical wireless deployment.



\## 🎯 Problem Statement



Wireless spectrum is a limited resource, and channel occupancy can change dynamically over time.



Traditional channel-selection approaches may select the first available channel without considering future occupancy, interference, or overall channel suitability.



This can result in inefficient spectrum utilization, transmission collisions, and increased packet-delivery delays.



SPECTRIX addresses this problem by exploring an intelligent channel-selection approach that combines cooperative sensing, occupancy prediction, and fuzzy-based channel evaluation.



\## 🎯 Project Objectives



1\. Generate a seven-day synthetic wireless channel occupancy dataset.

2\. Simulate primary-user and secondary-user activities.

3\. Implement individual and cooperative spectrum sensing.

4\. Predict channel occupancy using an HMM.

5\. Evaluate channel suitability through fuzzy logic.

6\. Develop the proposed ASSM channel allocator.

7\. Compare ASSM with a Standard First-Available baseline.

8\. Evaluate system performance under different user counts and traffic loads.



\---



\## 🏗️ System Architecture



The following diagram represents the major processing stages implemented in SPECTRIX.



```mermaid

flowchart TD

&#x20;   A\["Synthetic Channel Occupancy Dataset"] --> B\["Channel Simulator"]

&#x20;   B --> C\["Primary User \& Secondary User Simulation"]

&#x20;   C --> D\["Spectrum Sensing Engine"]

&#x20;   D --> E\["Cooperative Spectrum Sensing"]

&#x20;   E --> F\["HMM Occupancy Predictor"]

&#x20;   E --> G\["Fuzzy Logic Engine"]

&#x20;   F --> H\["ASSM Channel Allocator"]

&#x20;   G --> H

&#x20;   H --> I\["Transmission Simulator"]

&#x20;   I --> J\["Performance Evaluation"]

&#x20;   J --> K\["Graphs, Reports \& Metrics"]



&#x20;   style A fill:#dbeafe,stroke:#2563eb,color:#111827

&#x20;   style H fill:#dcfce7,stroke:#16a34a,color:#111827

&#x20;   style K fill:#fef3c7,stroke:#d97706,color:#111827

```



\### Main System Components



| Component | Responsibility |

|---|---|

| Database Generator | Generates synthetic occupancy records |

| Primary User Simulator | Models licensed-channel activity |

| Secondary User Simulator | Generates transmission requests |

| Channel Simulator | Provides channel occupancy information |

| Spectrum Sensing Engine | Performs individual sensing |

| Cooperative Sensing | Combines sensing reports using majority voting |

| HMM Predictor | Predicts channel occupancy |

| Fuzzy Logic Engine | Calculates channel suitability |

| ASSM Allocator | Selects the most suitable eligible channel |

| Transmission Simulator | Simulates packet transmission and delivery |

| Performance Analysis | Generates comparison metrics and graphs |



\## ⚙️ How SPECTRIX Works



```mermaid

flowchart TD

&#x20;   A\["Start Simulation"] --> B\["Load Channel Occupancy Data"]

&#x20;   B --> C\["Generate Secondary User Requests"]

&#x20;   C --> D\["Perform Spectrum Sensing"]

&#x20;   D --> E\["Combine Sensing Reports"]

&#x20;   E --> F\["Predict Future Occupancy Using HMM"]

&#x20;   F --> G\["Calculate Fuzzy Suitability Score"]

&#x20;   G --> H{"Channel Eligible?"}

&#x20;   H -->|No| I\["Evaluate Next Channel"]

&#x20;   I --> H

&#x20;   H -->|Yes| J\["Select Highest-Scoring Channel"]

&#x20;   J --> K\["Simulate Packet Transmission"]

&#x20;   K --> L\["Calculate Performance Metrics"]

&#x20;   L --> M\["Generate Reports and Graphs"]

```



\## 🧠 Proposed ASSM Algorithm



The proposed Adaptive Spectrum Selection and Management algorithm combines three major decision inputs.



\### 1. Cooperative Spectrum Sensing



Multiple secondary users participate in spectrum sensing.



The project uses majority voting to combine the individual sensing decisions.



The cooperative sensing configuration is:



| Parameter | Value |

|---|---:|

| Individual detection probability | 0.90 |

| Individual false alarm probability | 0.05 |

| Cooperative detection probability | 0.99144 |

| Cooperative false alarm probability | 0.00116 |

| Number of cooperative sensors | 5 |



\### 2. HMM Occupancy Prediction



The Hidden Markov Model estimates future channel occupancy based on observed channel-state transitions.



The model is trained using the first five days of simulation data and evaluated on the final two days.



\### 3. Fuzzy Logic Evaluation



The fuzzy logic engine evaluates channel suitability using:



\- Signal-to-Noise Ratio (SNR)

\- Channel occupancy

\- Interference



It uses nine heuristic fuzzy rules and centroid defuzzification to calculate a suitability score.



\### 4. Channel Allocation Conditions



ASSM selects a channel only when:



| Condition | Requirement |

|---|---|

| Cooperative sensing | Channel must be IDLE |

| HMM prediction | Predicted busy probability ≤ 0.20 |

| Fuzzy suitability | Score ≥ 40 |

| Allocation | Highest-scoring eligible channel |

| Duplicate allocation | Not allowed within the same simulation slot |



The selected channel is then passed to the transmission simulator.



\---



\## 🛠️ Technology Stack



| Technology | Application |

|---|---|

| Python 3.13.1 | Core implementation |

| NumPy | Numerical computation and randomization |

| Pandas | Dataset generation and processing |

| Matplotlib | Graphs and visual analysis |

| SciPy | Scientific computing |

| Scikit-fuzzy | Fuzzy inference system |

| VS Code | Development environment |



\## 📊 Simulation Configuration



| Parameter | Value |

|---|---:|

| Simulation duration | 7 days |

| Sampling interval | 10 seconds |

| Total simulation slots | 60,480 |

| Total channels | 8 |

| Licensed channels | 6 |

| Unlicensed channels | 2 |

| Primary users | 6 |

| Initial secondary users | 5 |

| Packet size | 1,500 bytes |

| Maximum transmission attempts | 3 |

| Random seed | 42 |

| Total occupancy records | 483,840 |



\### Channel Configuration



| Channel | Type | Classification |

|---|---|---|

| CH01 | Broadcast | Licensed |

| CH02 | Broadcast | Licensed |

| CH03 | Cellular | Licensed |

| CH04 | Cellular | Licensed |

| CH05 | Business | Licensed |

| CH06 | Business | Licensed |

| CH07 | ISM | Unlicensed |

| CH08 | ISM | Unlicensed |



\---



\## 🧪 Experimental Methodology



\### Training and Evaluation



The seven-day dataset is divided into two periods.



| Period | Dates | Purpose |

|---|---|---|

| Training / Warm-up | Nov 2–6, 2026 | HMM training and simulation warm-up |

| Held-out Testing | Nov 7–8, 2026 | Performance evaluation |



The HMM is trained using the first five days. The final two days are reserved for held-out evaluation.



\### Baseline vs ASSM Comparison



Two channel-allocation approaches are compared using the same saved scenario inputs.



\*\*Standard First-Available Baseline\*\*



\- Uses cooperative spectrum sensing.

\- Selects the first available channel in a fixed channel order.

\- Avoids duplicate allocations within the same simulation slot.



\*\*Proposed ASSM\*\*



\- Uses cooperative spectrum sensing.

\- Incorporates HMM-based occupancy prediction.

\- Evaluates channel suitability through fuzzy logic.

\- Selects the highest-scoring eligible channel.



Both algorithms use common simulation scenarios to support a controlled comparison.



\### Additional Experiments



| Experiment | Configurations |

|---|---|

| Main comparison | Standard Baseline vs ASSM |

| Scalability | 3, 5, 8 and 10 secondary users |

| Traffic-load testing | Low, Normal and High |

| HMM evaluation | Prediction metrics and calibration |

| Part A analysis | Occupancy and transmission performance |



\---



\## 📈 Performance Results



\### 1. Main Baseline vs ASSM Comparison



Held-out evaluation results for the two-day testing period:



| Performance Metric | Standard Baseline | Proposed ASSM |

|---|---:|---:|

| Packet Delivery Ratio | 99.6755% | \*\*99.8491%\*\* |

| Collisions | 246 | \*\*1\*\* |

| Collision Rate | 1.6069% | \*\*0.0068%\*\* |

| Throughput | 917.2917 bps | \*\*918.8889 bps\*\* |

| Average Latency | 12.2326 s | \*\*11.7964 s\*\* |

| Allocation Utilization | 11.0742% | 10.6641% |

| Jain's Fairness Index | 0.99955 | 0.99957 |



\*\*Key observations:\*\*



\- Approximately 99.59% reduction in collisions.

\- Improved packet delivery ratio.

\- Slightly higher throughput.

\- Lower average latency.

\- Nearly equal fairness between the two approaches.



\### 2. Scalability Analysis



The proposed algorithm was evaluated with different numbers of secondary users.



| Secondary Users | Baseline Collisions | ASSM Collisions | Collision Reduction |

|---:|---:|---:|---:|

| 3 | 563 | 0 | 100.00% |

| 5 | 244 | 0 | 100.00% |

| 8 | 172 | 1 | 99.42% |

| 10 | 61 | 0 | 100.00% |



\*\*Overall observations:\*\*



\- Higher PDR with ASSM in all four experiments.

\- Higher throughput with ASSM in all four experiments.

\- Lower latency with ASSM in three out of four experiments.

\- Aggregate collision reduction of approximately 99.90%.



\### 3. Traffic-Load Analysis



| Traffic Condition | Baseline Collisions | ASSM Collisions |

|---|---:|---:|

| Low Load | 147 | 0 |

| Normal Load | 246 | 1 |

| High Load | 309 | 1 |



ASSM substantially reduced collisions across all three tested traffic conditions.



\### 4. HMM Evaluation



| Evaluation Metric | Result |

|---|---:|

| Accuracy | 97.258% |

| Precision | 96.923% |

| Recall | 96.907% |

| F1-score | 96.915% |

| False Positive Rate | 2.461% |

| Miss Rate | 3.093% |



The HMM evaluation also includes probability calibration and comparison against a persistence baseline.



\---



\## 📊 Visual Results



The following are selected graphs generated by the project.



\### Baseline vs ASSM Performance



!\[Overall Performance Comparison](outputs/graphs/baseline\_comparison/overall\_performance\_comparison.png)



!\[Collision Rate Comparison](outputs/graphs/baseline\_comparison/collision\_rate\_comparison.png)



!\[Channel Allocation Comparison](outputs/graphs/baseline\_comparison/channel\_allocation\_comparison.png)



\### Scalability Analysis



!\[Scalability Collision Comparison](outputs/graphs/scalability\_analysis/collision\_comparison.png)



!\[Scalability Throughput Comparison](outputs/graphs/scalability\_analysis/throughput\_comparison.png)



!\[Scalability Latency Comparison](outputs/graphs/scalability\_analysis/latency\_comparison.png)



\### Channel Occupancy and HMM Analysis



!\[Channel Occupancy Heatmap](outputs/graphs/channel\_heatmap.png)



!\[HMM Probability Calibration](outputs/graphs/hmm\_probability\_calibration.png)



!\[Weekly Occupancy](outputs/graphs/weekly\_occupancy.png)



\*Additional graphs and detailed experiment reports are available in the `outputs/` directory.\*



\---



\## 📁 Project Structure



```text

SPECTRIX/

│

├── README.md

├── requirements.txt

├── .gitignore

│

├── data/

│   ├── raw/

│   │   └── occupancy\_7days.csv

│   │

│   └── processed/

│       ├── baseline\_comparison/

│       ├── robustness\_testing/

│       └── traffic\_load\_testing/

│

├── src/

│   ├── config.py

│   ├── database\_generator.py

│   ├── primary\_user.py

│   ├── secondary\_user.py

│   ├── channel\_simulator.py

│   ├── spectrum\_sensing.py

│   ├── hmm\_predictor.py

│   ├── fuzzy\_logic.py

│   ├── assm\_allocator.py

│   ├── baseline\_allocator.py

│   ├── transmission\_simulator.py

│   ├── scenario\_generator.py

│   ├── baseline\_comparison\_simulation.py

│   └── ... (additional analysis and experiment scripts)

│

└── outputs/

&#x20;   ├── graphs/

&#x20;   │   ├── baseline\_comparison/

&#x20;   │   └── scalability\_analysis/

&#x20;   │

&#x20;   └── reports/

&#x20;       ├── baseline\_comparison/

&#x20;       ├── robustness\_testing/

&#x20;       └── traffic\_load\_testing/

```



> \[!IMPORTANT]

> Large generated scenario CSV files are excluded from normal Git tracking to keep the repository manageable. Their generation scripts and compact metadata are retained. The raw occupancy dataset is included as the primary simulation input.



\---



\## 💻 Installation and Setup



\### Prerequisites



\- Python 3.13.1

\- Git

\- Visual Studio Code (recommended)



\### 1. Clone the Repository



```bash

git clone https://github.com/YOUR-USERNAME/SPECTRIX-AI-Cognitive-Radio.git

```



```bash

cd SPECTRIX-AI-Cognitive-Radio

```



\### 2. Create a Virtual Environment



\*\*Windows PowerShell:\*\*



```powershell

python -m venv .venv

```



Activate it:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 3. Install Dependencies



```bash

python -m pip install -r requirements.txt

```



\---



\## ▶️ Running the Project



Run all commands from the repository's root directory.



\### Step 1: Validate the Channel Simulator



```powershell

python src/channel\_simulator.py

```



\### Step 2: Generate the Raw Occupancy Dataset (if needed)



```powershell

python src/database\_generator.py

```



The generated dataset is saved to:



```text

data/raw/occupancy\_7days.csv

```



\### Step 3: Generate Common Simulation Scenarios



```powershell

python src/scenario\_generator.py

```



This generates:



```text

data/processed/baseline\_comparison/common\_scenarios\_7days.csv

```



\### Step 4: Run Baseline vs ASSM Comparison



```powershell

python src/baseline\_comparison\_simulation.py

```



The comparison generates performance reports for both algorithms.



> \[!WARNING]

> Generation and simulation scripts write files into the project directories and may overwrite existing generated results. Keep backups before rerunning experiments.



Additional scripts support HMM evaluation, robustness testing, traffic-load analysis, scalability experiments, and visualization.



\---



\## ⚠️ Limitations



\- The project uses synthetic channel-occupancy data.

\- Channel-quality parameters are simulated assumptions rather than physical RF measurements.

\- Results depend on simulation configuration, randomization, traffic patterns and allocation rules.

\- Latency improvements are not consistent across all tested scalability scenarios.

\- The system has not been validated using a real-world wireless deployment.



\## 🚀 Future Scope



\- Integration with Software Defined Radio (SDR) hardware.

\- Validation using real-world spectrum measurements.

\- Adaptive HMM training and prediction.

\- Evaluation of additional intelligent channel-allocation algorithms.

\- Testing under more complex and dynamic wireless environments.

\- Extension to larger cognitive radio network simulations.



\## 🏁 Conclusion



SPECTRIX demonstrates a simulation-based approach to intelligent spectrum management using cooperative spectrum sensing, HMM-based occupancy prediction, and fuzzy logic-based channel evaluation.



The experimental results demonstrate substantial collision reduction and improvements in packet delivery ratio and throughput across the tested scenarios.



The project also highlights the importance of evaluating multiple performance metrics, as latency improvement is not consistent under every operating condition.



SPECTRIX provides a foundation for further exploration of intelligent spectrum allocation in dynamic wireless networks.



\---



<p align="center">

&#x20; <strong>SPECTRIX — Smarter Spectrum. Intelligent Allocation.</strong>

</p>



<p align="center">

&#x20; Final-Year Academic Project | Python | Cognitive Radio | AI-Based Spectrum Management

</p>




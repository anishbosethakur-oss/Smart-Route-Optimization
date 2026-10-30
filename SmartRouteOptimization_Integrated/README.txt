SMART ROUTE OPTIMIZATION — SIX-ROUTER DQN + BASELINE COMPARISON
=================================================================

PROJECT
Python network simulation integrated with a Deep Q-Network (DQN) routing agent.

UPDATED TOPOLOGY
- Routers: R1, R2, R3, R4, R5, R6
- Existing links retained; new links: R2-R6 (12 ms, 70 Mbps), R6-R5 (8 ms, 80 Mbps).
- State size is now 28 features: 12 router indicators + 16 link delay/utilization features.
- The old five-router model has incompatible input/output dimensions and must be retrained.

SETUP (Windows PowerShell; run from this project root)
    python -m pip install -r requirements.txt

1. Verify integration:
    python integration_test.py

2. Retrain the DQN for the six-router state/action dimensions:
    python member2_ai/train.py

3. Test the trained agent:
    python member2_ai/test_agent.py

4. Compare DQN against adaptive Dijkstra (100 matched traffic seeds per method):
    python member2_ai/compare_routing.py
   Results: results/dqn_vs_dijkstra.csv

5. Show an actual one-step reinforcement calculation:
    python member2_ai/reinforcement_calculation.py

6. Show the topology and Dijkstra route:
    python code/network.py

WHAT THE REINFORCEMENT CALCULATION SHOWS
- Q(s,a): DQN's predicted value for the selected action.
- Reward r: immediate environment feedback.
- Target y = r + gamma * max_a' Q_target(s',a') for a non-terminal step; y = r for terminal.
- TD error = y - Q(s,a).
- Training uses Smooth L1 (Huber) loss between current Q prediction and target.

FAIR-COMPARISON NOTES
- The comparison uses the same topology and paired traffic seeds for both methods.
- Adaptive Dijkstra recomputes the shortest path from current effective link delays at each hop.
- The script prints measured success rate, mean delay for successful routes, and mean reward.
- Do not claim DQN is better unless your measured results support it. If DQN performs worse, use that as evidence to improve training/reward/state and rerun.
- Results are simulator results, not measurements from real internet routers.

DEPENDENCIES
torch
numpy
matplotlib
networkx


LATEST LOCAL VALIDATION (after six-router retraining)
- Integration test: passed; state size 28, action size 6.
- DQN test: 30/30 successful routes in the run performed during package preparation.
- Paired comparison (100 traffic seeds): DQN mean successful delay 37.54 ms;
  Static Dijkstra 37.06 ms; Adaptive Dijkstra 37.06 ms. All reported 100% success.
- In this experiment DQN did NOT outperform Dijkstra on mean delay. Do not claim it is better yet.
- The comparison CSV and reinforcement calculation output can be regenerated with the commands above.

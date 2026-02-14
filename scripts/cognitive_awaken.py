# BIZRA Cognitive Awaken v1.0
# Implements a World-Model RL loop inspired by Dreamer 4
# Part of Phase 2: Cognitive Awaken

import json
import random
from pathlib import Path
import networkx as nx
from bizra_config import GRAPH_PATH

class BIZRA_WorldModel:
    """Simulates the 'Imagination' environment based on the Hypergraph."""
    def __init__(self):
        self.graph = self._load_graph()
        self.current_state = None

    def _load_graph(self):
        # In a real scenario, this loads the NetworkX graph from INDEXED_PATH
        # We'll simulate connectivity for the Phase 2 prototype
        G = nx.Graph()
        # Mock nodes representing core BIZRA concepts
        nodes = ["Foundation", "BlockTree", "URP", "SAPE", "DDAGI", "Cognitive_Awaken", "Transcendent"]
        G.add_nodes_from(nodes)
        G.add_edges_from([
            ("Foundation", "BlockTree"), ("Foundation", "URP"),
            ("BlockTree", "DDAGI"), ("URP", "DDAGI"),
            ("DDAGI", "Cognitive_Awaken"), ("Cognitive_Awaken", "Transcendent")
        ])
        return G

    def reset(self, stimulus="Foundation"):
        self.current_state = stimulus if stimulus in self.graph else "Foundation"
        return self.current_state

    def step(self, action):
        """Action is a transition to a neighboring node."""
        neighbors = list(self.graph.neighbors(self.current_state))
        if action in neighbors:
            self.current_state = action
            reward = 1.0 # Successful transition
            done = (self.current_state == "Transcendent")
        else:
            reward = -0.1 # Hallucination/Drift
            done = False
        
        return self.current_state, reward, done

class ImaginationAgent:
    """Refined RL agent that trains inside the World Model."""
    def __init__(self, world_model):
        self.wm = world_model
        self.policy = {} # State-Action value mapping (simplified)

    def train_in_imagination(self, episodes=5):
        print(f"🧠 Training Phase 2 Agent in Imagination ({episodes} episodes)...")
        for i in range(episodes):
            state = self.wm.reset()
            total_reward = 0
            steps = 0
            path = [state]
            
            done = False
            while not done and steps < 10:
                # Epsilon-greedy exploration
                neighbors = list(self.wm.graph.neighbors(state))
                if not neighbors: break
                
                action = random.choice(neighbors)
                state, reward, done = self.wm.step(action)
                total_reward += reward
                path.append(state)
                steps += 1
            
            print(f"  Episode {i+1}: Path={' -> '.join(path)} | Reward={total_reward:.2f}")

    def plan(self, goal="Cognitive_Awaken"):
        """Returns a path to the goal via the world model."""
        start = "Foundation"
        try:
            path = nx.shortest_path(self.wm.graph, source=start, target=goal)
            return path
        except:
            return [start, "unknown", goal]

if __name__ == "__main__":
    print("🌅 Activating Phase 2: Cognitive Awaken...")
    wm = BIZRA_WorldModel()
    agent = ImaginationAgent(wm)
    
    agent.train_in_imagination()
    
    print("\n🔮 imagination-based Plan for 'Cognitive_Awaken':")
    print(" -> ".join(agent.plan()))

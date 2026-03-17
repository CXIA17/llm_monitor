#!/usr/bin/env python3
"""
Interaction Graph
=================

Defines how agents communicate with each other using a directed graph.
Supports various topologies: linear, round-robin, hub-spoke, custom.

Usage:
    graph = InteractionGraph()
    graph.add_edge("proposer", "critic", context_window=2)
    graph.add_edge("critic", "judge")
    
    # Or use presets
    graph = InteractionGraph.create_debate_topology(["p", "c", "j"])
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from enum import Enum
import json


class EdgeType(Enum):
    """Type of interaction between agents."""
    SEQUENTIAL = "sequential"      # A speaks, then B speaks
    REACTIVE = "reactive"          # B responds to A's message
    BROADCAST = "broadcast"        # A sends to all connected
    CONDITIONAL = "conditional"    # Edge activates based on condition
    BIDIRECTIONAL = "bidirectional"  # Both directions


@dataclass
class InteractionEdge:
    """A directed edge in the interaction graph."""
    source: str
    target: str
    edge_type: EdgeType = EdgeType.SEQUENTIAL
    
    # How many previous messages from source does target see?
    context_window: int = 1
    
    # Optional: only activate edge under certain conditions
    condition: Optional[Callable[[Dict], bool]] = None
    condition_description: str = ""
    
    # Edge weight (for prioritization)
    weight: float = 1.0
    
    # Custom prompt injection for this edge
    edge_prompt: Optional[str] = None  # e.g., "Respond to the previous argument:"
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "context_window": self.context_window,
            "condition_description": self.condition_description,
            "weight": self.weight,
            "edge_prompt": self.edge_prompt,
            "metadata": self.metadata
        }


@dataclass 
class ConversationState:
    """Tracks the current state of a multi-agent conversation."""
    round_number: int = 0
    current_speaker: Optional[str] = None
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    agent_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Termination tracking
    terminated: bool = False
    termination_reason: str = ""
    
    def add_message(self, agent_id: str, content: str, metadata: Optional[Dict] = None):
        self.message_history.append({
            "round": self.round_number,
            "agent": agent_id,
            "content": content,
            "metadata": metadata or {}
        })
    
    def get_agent_messages(self, agent_id: str, last_n: Optional[int] = None) -> List[Dict]:
        msgs = [m for m in self.message_history if m["agent"] == agent_id]
        if last_n:
            return msgs[-last_n:]
        return msgs
    
    def get_context_for_agent(self, agent_id: str, context_window: int) -> str:
        """Get formatted context string for an agent."""
        relevant = self.message_history[-context_window:] if context_window > 0 else []
        lines = []
        for msg in relevant:
            lines.append(f"[{msg['agent']}]: {msg['content']}")
        return "\n\n".join(lines)


class InteractionGraph:
    """
    Directed graph defining agent interactions.
    
    Nodes = Agents
    Edges = Communication channels with properties
    """
    
    def __init__(self):
        self._nodes: Set[str] = set()
        self._edges: Dict[Tuple[str, str], InteractionEdge] = {}
        self._adjacency: Dict[str, List[str]] = {}  # outgoing edges
        self._reverse_adjacency: Dict[str, List[str]] = {}  # incoming edges
        
        # Execution order (computed from topology)
        self._execution_order: List[str] = []
        
    # === Graph Construction ===
    
    def add_node(self, agent_id: str):
        """Add an agent node to the graph."""
        self._nodes.add(agent_id)
        if agent_id not in self._adjacency:
            self._adjacency[agent_id] = []
        if agent_id not in self._reverse_adjacency:
            self._reverse_adjacency[agent_id] = []
    
    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType = EdgeType.SEQUENTIAL,
        context_window: int = 1,
        weight: float = 1.0,
        edge_prompt: Optional[str] = None,
        condition: Optional[Callable] = None,
        condition_description: str = "",
        **metadata
    ) -> InteractionEdge:
        """
        Add a directed edge from source to target.
        
        Args:
            source: Source agent ID
            target: Target agent ID  
            edge_type: Type of interaction
            context_window: How many messages target sees from conversation
            weight: Edge priority weight
            edge_prompt: Custom prompt for this transition
            condition: Optional function(state) -> bool for conditional edges
        """
        # Ensure nodes exist
        self.add_node(source)
        self.add_node(target)
        
        edge = InteractionEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            context_window=context_window,
            weight=weight,
            edge_prompt=edge_prompt,
            condition=condition,
            condition_description=condition_description,
            metadata=metadata
        )
        
        self._edges[(source, target)] = edge
        
        if target not in self._adjacency[source]:
            self._adjacency[source].append(target)
        if source not in self._reverse_adjacency[target]:
            self._reverse_adjacency[target].append(source)
            
        return edge
    
    def add_bidirectional_edge(self, agent_a: str, agent_b: str, **kwargs) -> Tuple[InteractionEdge, InteractionEdge]:
        """Add edges in both directions."""
        e1 = self.add_edge(agent_a, agent_b, **kwargs)
        e2 = self.add_edge(agent_b, agent_a, **kwargs)
        return e1, e2
    
    def remove_edge(self, source: str, target: str):
        """Remove an edge."""
        key = (source, target)
        if key in self._edges:
            del self._edges[key]
            self._adjacency[source].remove(target)
            self._reverse_adjacency[target].remove(source)
    
    def get_edge(self, source: str, target: str) -> Optional[InteractionEdge]:
        """Get edge between two agents."""
        return self._edges.get((source, target))
    
    def get_outgoing(self, agent_id: str) -> List[str]:
        """Get agents that this agent sends to."""
        return self._adjacency.get(agent_id, [])
    
    def get_incoming(self, agent_id: str) -> List[str]:
        """Get agents that send to this agent."""
        return self._reverse_adjacency.get(agent_id, [])
    
    # === Topology Analysis ===
    
    def get_roots(self) -> List[str]:
        """Get agents with no incoming edges (conversation starters)."""
        return [n for n in self._nodes if len(self._reverse_adjacency.get(n, [])) == 0]
    
    def get_leaves(self) -> List[str]:
        """Get agents with no outgoing edges (conversation enders)."""
        return [n for n in self._nodes if len(self._adjacency.get(n, [])) == 0]
    
    def compute_execution_order(self, start_node: Optional[str] = None) -> List[str]:
        """
        Compute topological execution order.
        For cyclic graphs, returns a round-robin order.
        """
        if not self._nodes:
            return []
            
        # Try topological sort first
        visited = set()
        order = []
        temp_visited = set()
        
        def dfs(node):
            if node in temp_visited:
                return False  # Cycle detected
            if node in visited:
                return True
            temp_visited.add(node)
            for neighbor in self._adjacency.get(node, []):
                if not dfs(neighbor):
                    return False
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
            return True
        
        # Start from specified node or roots
        start_nodes = [start_node] if start_node else (self.get_roots() or list(self._nodes))
        
        is_dag = True
        for node in start_nodes:
            if not dfs(node):
                is_dag = False
                break
        
        if is_dag:
            # Reverse for correct order
            self._execution_order = order[::-1]
        else:
            # Cyclic graph: use BFS-based order
            self._execution_order = self._compute_cyclic_order(start_nodes[0])
            
        return self._execution_order
    
    def _compute_cyclic_order(self, start: str) -> List[str]:
        """For cyclic graphs, compute a reasonable execution order."""
        order = []
        visited = set()
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for neighbor in self._adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        # Add any disconnected nodes
        for node in self._nodes:
            if node not in order:
                order.append(node)
                
        return order
    
    def is_cyclic(self) -> bool:
        """Check if graph contains cycles."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self._adjacency.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False
        
        for node in self._nodes:
            if node not in visited:
                if has_cycle(node):
                    return True
        return False
    
    # === Execution ===
    
    def get_next_speakers(
        self, 
        current_speaker: str, 
        state: ConversationState
    ) -> List[Tuple[str, InteractionEdge]]:
        """
        Get the next agents to speak after current_speaker.
        Returns list of (agent_id, edge) tuples.
        Filters by edge conditions if present.
        """
        results = []
        for target in self._adjacency.get(current_speaker, []):
            edge = self._edges[(current_speaker, target)]
            
            # Check condition if present
            if edge.condition:
                state_dict = {
                    "round": state.round_number,
                    "history": state.message_history,
                    "agent_states": state.agent_states
                }
                if not edge.condition(state_dict):
                    continue
                    
            results.append((target, edge))
        
        # Sort by weight (higher first)
        results.sort(key=lambda x: x[1].weight, reverse=True)
        return results
    
    # === Preset Topologies ===
    
    @classmethod
    def create_linear(cls, agent_ids: List[str], context_window: int = 2) -> "InteractionGraph":
        """
        Linear chain: A → B → C → D
        Each agent responds to the previous.
        """
        graph = cls()
        for i in range(len(agent_ids) - 1):
            graph.add_edge(
                agent_ids[i], 
                agent_ids[i + 1],
                context_window=context_window,
                edge_prompt=f"Respond to {agent_ids[i]}:"
            )
        return graph
    
    @classmethod
    def create_round_robin(cls, agent_ids: List[str], num_rounds: int = 3, context_window: int = 3) -> "InteractionGraph":
        """
        Round-robin: A → B → C → A → B → C → ...
        Cyclic with rounds.
        """
        graph = cls()
        for i in range(len(agent_ids)):
            next_idx = (i + 1) % len(agent_ids)
            graph.add_edge(
                agent_ids[i],
                agent_ids[next_idx],
                context_window=context_window
            )
        # Store round limit in metadata
        graph._metadata = {"max_rounds": num_rounds}
        return graph
    
    @classmethod
    def create_hub_spoke(cls, hub_agent: str, spoke_agents: List[str], context_window: int = 2) -> "InteractionGraph":
        """
        Hub-and-spoke: All spokes connect to central hub.
        
             B
             ↓↑
        A ↔ HUB ↔ C
             ↓↑
             D
        """
        graph = cls()
        for spoke in spoke_agents:
            graph.add_bidirectional_edge(hub_agent, spoke, context_window=context_window)
        return graph
    
    @classmethod
    def create_debate_topology(cls, agent_ids: List[str], context_window: int = 3) -> "InteractionGraph":
        """
        Standard debate: Proposer ↔ Critic, both → Judge
        Expects 3 agents: [proposer, critic, judge]
        """
        if len(agent_ids) < 3:
            raise ValueError("Debate topology requires at least 3 agents")
        
        proposer, critic, judge = agent_ids[0], agent_ids[1], agent_ids[2]
        
        graph = cls()
        
        # Proposer and Critic go back and forth
        graph.add_bidirectional_edge(
            proposer, critic,
            context_window=context_window,
            edge_prompt="Respond to the argument:"
        )
        
        # Both submit to Judge
        graph.add_edge(proposer, judge, context_window=context_window * 2)
        graph.add_edge(critic, judge, context_window=context_window * 2)
        
        return graph
    
    @classmethod
    def create_panel_discussion(cls, moderator: str, panelists: List[str], context_window: int = 4) -> "InteractionGraph":
        """
        Panel discussion: Moderator orchestrates, panelists respond.
        
        Moderator → All Panelists
        Panelists → Each other (for rebuttals)
        All → Moderator (for summary)
        """
        graph = cls()
        
        # Moderator to all panelists
        for p in panelists:
            graph.add_edge(moderator, p, context_window=context_window, weight=2.0)
            graph.add_edge(p, moderator, context_window=context_window)
        
        # Panelists can respond to each other
        for i, p1 in enumerate(panelists):
            for p2 in panelists[i+1:]:
                graph.add_bidirectional_edge(p1, p2, context_window=context_window, weight=0.5)
        
        return graph
    
    @classmethod
    def create_adversarial(cls, defender: str, attackers: List[str], judge: str, context_window: int = 3) -> "InteractionGraph":
        """
        Adversarial setup: Multiple attackers challenge a defender, judge evaluates.
        """
        graph = cls()
        
        for attacker in attackers:
            # Defender presents, attackers challenge
            graph.add_edge(defender, attacker, context_window=context_window)
            graph.add_edge(attacker, defender, context_window=context_window)
            # Attacker submits to judge
            graph.add_edge(attacker, judge, context_window=context_window * 2)
        
        # Defender submits to judge
        graph.add_edge(defender, judge, context_window=context_window * 2)
        
        return graph
    
    @classmethod
    def create_court_topology(
        cls,
        plaintiff: str = "plaintiff_attorney",
        defense: str = "defense_attorney",
        judge: str = "court_judge",
        context_window: int = 50,
    ) -> "InteractionGraph":
        """
        Court trial topology:
          Plaintiff → Defense (reactive rebuttal)
          Defense → Plaintiff (reactive rebuttal)
          Both → Judge (full transcript)

        The large context_window ensures the Judge (and each attorney)
        can see the entire transcript rather than a sliding window.
        """
        graph = cls()

        # Attorneys go back and forth
        graph.add_edge(
            plaintiff, defense,
            edge_type=EdgeType.REACTIVE,
            context_window=context_window,
            edge_prompt="The Plaintiff has presented their argument. Respond with your rebuttal.",
        )
        graph.add_edge(
            defense, plaintiff,
            edge_type=EdgeType.REACTIVE,
            context_window=context_window,
            edge_prompt="The Defense has responded. Present your next argument.",
        )

        # Both submit to Judge (full transcript)
        graph.add_edge(
            plaintiff, judge,
            edge_type=EdgeType.SEQUENTIAL,
            context_window=context_window,
            weight=2.0,
        )
        graph.add_edge(
            defense, judge,
            edge_type=EdgeType.SEQUENTIAL,
            context_window=context_window,
            weight=2.0,
        )

        return graph

    # === Serialization ===
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self._nodes),
            "edges": [e.to_dict() for e in self._edges.values()],
            "execution_order": self._execution_order,
            "metadata": getattr(self, "_metadata", {})
        }
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "InteractionGraph":
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        graph = cls()
        for node in data["nodes"]:
            graph.add_node(node)
        
        for edge_data in data["edges"]:
            graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                edge_type=EdgeType(edge_data["edge_type"]),
                context_window=edge_data["context_window"],
                weight=edge_data["weight"],
                edge_prompt=edge_data.get("edge_prompt")
            )
        
        graph._metadata = data.get("metadata", {})
        return graph
    
    # === Visualization ===
    
    def to_mermaid(self) -> str:
        """Generate Mermaid diagram syntax."""
        lines = ["graph TD"]
        for (src, tgt), edge in self._edges.items():
            arrow = "-->" if edge.edge_type != EdgeType.BIDIRECTIONAL else "<-->"
            label = f"|{edge.edge_prompt[:20]}|" if edge.edge_prompt else ""
            lines.append(f"    {src}{arrow}{label}{tgt}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"InteractionGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
    
    def __len__(self) -> int:
        return len(self._nodes)
"""
Base Agent Class with Memory and Learning Capabilities
Provides common functionality for all specialized agents
"""

import json
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


class BaseAgent:
    """
    Base agent class with decision memory and learning capabilities
    """
    
    def __init__(self, agent_name: str, memory_dir: str = "data/memories"):
        self.agent_name = agent_name
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.decision_history = []
        self.successful_actions = []
        self.failed_actions = []
        self.last_reasoning = ""
        
        self._load_memory()
    
    def _load_memory(self):
        """Load decision history from disk"""
        success_file = self.memory_dir / f"{self.agent_name}_successful.json"
        failed_file = self.memory_dir / f"{self.agent_name}_failed.json"
        
        try:
            if success_file.exists():
                with open(success_file, 'r') as f:
                    self.successful_actions = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load successful actions: {e}")
        
        try:
            if failed_file.exists():
                with open(failed_file, 'r') as f:
                    self.failed_actions = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load failed actions: {e}")
    
    def _save_memory(self):
        """Save decision history to disk"""
        try:
            success_file = self.memory_dir / f"{self.agent_name}_successful.json"
            with open(success_file, 'w') as f:
                json.dump(self.successful_actions, f, indent=2)
            
            failed_file = self.memory_dir / f"{self.agent_name}_failed.json"
            with open(failed_file, 'w') as f:
                json.dump(self.failed_actions, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory: {e}")
    
    def record_successful_action(self, context: Dict, action: Dict, outcome: Dict):
        """Record a successful action for future learning"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "action": action,
            "outcome": outcome,
            "agent": self.agent_name
        }
        self.successful_actions.append(record)
        
        # Keep only last 1000 records
        if len(self.successful_actions) > 1000:
            self.successful_actions = self.successful_actions[-1000:]
        
        self._save_memory()
    
    def record_failed_action(self, context: Dict, action: Dict, error: str):
        """Record a failed action for learning what NOT to do"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "action": action,
            "error": error,
            "agent": self.agent_name
        }
        self.failed_actions.append(record)
        
        # Keep only last 500 records
        if len(self.failed_actions) > 500:
            self.failed_actions = self.failed_actions[-500:]
        
        self._save_memory()
    
    def get_similar_past_actions(self, context: Dict, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Find similar past actions that succeeded
        Used to improve decision making by learning from experience
        """
        similar = []
        
        for record in self.successful_actions:
            # Simple similarity based on violation types
            if self._is_similar_context(context, record["context"], similarity_threshold):
                similar.append(record)
        
        return sorted(similar, key=lambda x: x["timestamp"], reverse=True)[:5]
    
    def _is_similar_context(self, context1: Dict, context2: Dict, threshold: float) -> bool:
        """
        Check if two contexts are similar
        Simple implementation comparing violation patterns
        """
        try:
            v1_types = set(context1.get("violation_types", []))
            v2_types = set(context2.get("violation_types", []))
            
            if not v1_types or not v2_types:
                return False
            
            intersection = len(v1_types & v2_types)
            union = len(v1_types | v2_types)
            
            similarity = intersection / union if union > 0 else 0
            return similarity >= threshold
        except:
            return False
    
    def get_success_rate(self) -> float:
        """Calculate success rate of actions"""
        total = len(self.successful_actions) + len(self.failed_actions)
        if total == 0:
            return 0.0
        return len(self.successful_actions) / total
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary statistics of agent memory"""
        return {
            "agent_name": self.agent_name,
            "total_successful_actions": len(self.successful_actions),
            "total_failed_actions": len(self.failed_actions),
            "success_rate": self.get_success_rate(),
            "last_successful": self.successful_actions[-1]["timestamp"] if self.successful_actions else None,
            "last_failed": self.failed_actions[-1]["timestamp"] if self.failed_actions else None
        }
    
    def clear_memory(self):
        """Clear all memory (use with caution)"""
        self.successful_actions = []
        self.failed_actions = []
        self._save_memory()
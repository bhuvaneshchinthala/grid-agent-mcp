"""
Enhanced Base Agent Class with Advanced AI Capabilities
Provides chain-of-thought reasoning, confidence scoring, and agent collaboration
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback for environments where dotenv package is missing
    print("WARNING: 'python-dotenv' not found. Attempting manual .env load.")
    def load_dotenv(*args, **kwargs):
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    load_dotenv()

# Optional ollama import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

# Optional groq import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class BaseAgent:
    """
    Enhanced base agent class with:
    - Chain-of-thought reasoning
    - Confidence scoring
    - Agent collaboration protocol
    - Decision explainability
    - Contextual memory with similarity matching
    - Support for both local (Ollama) and cloud (Groq) LLMs
    """
    
    def __init__(self, agent_name: str, model: str = "mistral", memory_dir: str = "data/memories"):
        self.agent_name = agent_name
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Groq client if API key is present
        self.groq_client = None
        self.use_groq = False
        self.model = model
        
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key and GROQ_AVAILABLE:
            try:
                self.groq_client = Groq(api_key=api_key)
                self.use_groq = True
                # Map generic model names to Groq equivalents if needed
                if model in ["mistral", "llama2", "tinyllama"]:
                    self.model = "llama-3.3-70b-versatile"  # Updated high-performance model
                print(f"[{agent_name}] Using Groq API with model {self.model}")
            except Exception as e:
                print(f"[{agent_name}] Failed to initialize Groq: {e}. Falling back to Ollama.")
        else:
             print(f"[{agent_name}] Using local Ollama with model {self.model}")

        
        self.decision_history = []
        self.successful_actions = []
        self.failed_actions = []
        self.last_reasoning = ""
        self.last_confidence = 0.0
        self.collaboration_requests = []
        
        # System prompts for different reasoning modes
        self.system_prompts = {
            "analytical": self._get_analytical_prompt(),
            "emergency": self._get_emergency_prompt(),
            "planning": self._get_planning_prompt()
        }
        
        self._load_memory()
    
    def _call_llm(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 1024) -> str:
        """Abstracts the LLM call to support both Groq and Ollama"""
        if self.use_groq and self.groq_client:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"} if "json" in messages[-1]["content"].lower() else None
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                print(f"Groq Error: {e}")
                return "{}"
        elif OLLAMA_AVAILABLE:
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": temperature, "num_predict": max_tokens}
                )
                return response["message"]["content"]
            except Exception as e:
                print(f"Ollama Error: {e}")
                return "{}"
        else:
             return '{"error": "No LLM provider available"}'

    def _get_analytical_prompt(self) -> str:
        return f"""You are {self.agent_name}, an expert AI agent for power grid control.
You approach problems methodically with careful analysis.

CRITICAL RULES:
1. Think step-by-step before proposing actions
2. Consider safety implications of every action
3. Validate feasibility against network constraints
4. Provide confidence scores (0-100%) for recommendations
5. Always explain your reasoning

OUTPUT FORMAT: Always respond with valid JSON only, no markdown or explanations outside JSON."""

    def _get_emergency_prompt(self) -> str:
        return f"""You are {self.agent_name}, an expert AI agent for EMERGENCY power grid control.
CRITICAL SITUATION - Fast response required.

EMERGENCY RULES:
1. Prioritize grid stability above all
2. Take immediate protective actions first
3. Minimize damage and cascading failures
4. Communicate clearly what actions are needed
5. Escalate to operators for major decisions

OUTPUT FORMAT: Always respond with valid JSON only."""

    def _get_planning_prompt(self) -> str:
        return f"""You are {self.agent_name}, an expert AI agent for power grid planning.
You create comprehensive action plans for complex grid scenarios.

PLANNING RULES:
1. Break complex problems into sub-tasks
2. Identify dependencies between actions
3. Estimate time and resource requirements
4. Provide alternative strategies
5. Consider long-term implications

OUTPUT FORMAT: Always respond with valid JSON only."""
    
    # =========================================================================
    # CHAIN-OF-THOUGHT REASONING
    # =========================================================================
    
    def think_step_by_step(self, problem: str, context: Dict) -> Dict:
        """
        Apply chain-of-thought reasoning to analyze a problem
        Returns structured reasoning with intermediate steps
        """
        prompt = f"""{self.system_prompts['analytical']}

PROBLEM TO ANALYZE:
{problem}

CONTEXT:
{json.dumps(context, indent=2)}

Think through this step-by-step and return JSON with this structure:
{{
    "understanding": "What is the core problem?",
    "analysis_steps": [
        {{"step": 1, "thought": "First observation...", "conclusion": "..."}},
        {{"step": 2, "thought": "Based on step 1...", "conclusion": "..."}}
    ],
    "key_factors": ["factor1", "factor2"],
    "constraints": ["constraint1", "constraint2"],
    "recommended_approach": "Overall approach to solve this",
    "confidence": 85
}}"""

        try:
            content = self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            self.last_reasoning = content
            
            # Parse JSON
            result = self._extract_json(content)
            if result:
                self.last_confidence = result.get("confidence", 50) / 100.0
                return result
            
            return {"error": "Failed to parse reasoning", "raw": content}
            
        except Exception as e:
            return {"error": str(e), "confidence": 0}
    
    def reason_with_examples(self, problem: str, examples: List[Dict]) -> Dict:
        """
        Few-shot reasoning using past successful examples
        """
        examples_text = "\n".join([
            f"Example {i+1}:\n  Situation: {ex.get('context', {})}\n  Action: {ex.get('action', {})}\n  Outcome: {ex.get('outcome', 'success')}"
            for i, ex in enumerate(examples[:3])
        ])
        
        prompt = f"""{self.system_prompts['analytical']}

PAST SUCCESSFUL EXAMPLES:
{examples_text}

CURRENT PROBLEM:
{problem}

Based on the examples and current problem, recommend actions in JSON format:
{{
    "similar_to_example": 1,
    "adaptations_needed": ["adaptation1"],
    "recommended_actions": [
        {{"action_type": "...", "target": ..., "value": ...}}
    ],
    "confidence": 85,
    "reasoning": "Brief explanation"
}}"""

        try:
            content = self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            return self._extract_json(content) or {"error": "Parse failed", "raw": content}
            
        except Exception as e:
            return {"error": str(e)}
    
    # =========================================================================
    # CONFIDENCE SCORING
    # =========================================================================
    
    def assess_confidence(self, action: Dict, context: Dict) -> Tuple[float, List[str]]:
        """
        Assess confidence in a proposed action
        Returns (confidence_score, list_of_concerns)
        """
        concerns = []
        confidence = 1.0
        
        # Check for similar past actions
        similar = self.get_similar_past_actions(context)
        if len(similar) >= 3:
            confidence *= 1.1  # Boost for experience
        elif len(similar) == 0:
            confidence *= 0.7
            concerns.append("No similar past experience")
        
        # Check for past failures with similar actions
        for failed in self.failed_actions[-20:]:
            if failed.get("action", {}).get("action_type") == action.get("action_type"):
                confidence *= 0.8
                concerns.append(f"Similar action failed before: {failed.get('error', 'unknown')}")
                break
        
        # Validate action parameters
        if action.get("value", 0) > 50:  # Large magnitude actions are risky
            confidence *= 0.9
            concerns.append("Large magnitude action - higher risk")
        
        # Cap confidence
        confidence = min(1.0, max(0.1, confidence))
        self.last_confidence = confidence
        
        return confidence, concerns
    
    def get_confidence_explanation(self) -> str:
        """Get human-readable explanation of last confidence score"""
        if self.last_confidence >= 0.9:
            return "Very High - Strong historical evidence and low risk"
        elif self.last_confidence >= 0.7:
            return "High - Good evidence, acceptable risk"
        elif self.last_confidence >= 0.5:
            return "Medium - Limited evidence or moderate risk"
        elif self.last_confidence >= 0.3:
            return "Low - Uncertain, recommend verification"
        else:
            return "Very Low - Insufficient data, manual review required"
    
    # =========================================================================
    # AGENT COLLABORATION
    # =========================================================================
    
    def request_collaboration(self, target_agent: str, problem: str, context: Dict) -> Dict:
        """
        Request another agent's analysis on a problem
        """
        request = {
            "from_agent": self.agent_name,
            "to_agent": target_agent,
            "problem": problem,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        self.collaboration_requests.append(request)
        
        return {
            "request_id": len(self.collaboration_requests),
            "status": "pending",
            "message": f"Collaboration request sent to {target_agent}"
        }
    
    def respond_to_collaboration(self, request: Dict) -> Dict:
        """
        Respond to a collaboration request from another agent
        """
        problem = request.get("problem", "")
        context = request.get("context", {})
        
        # Use chain-of-thought to analyze
        analysis = self.think_step_by_step(problem, context)
        
        return {
            "from_agent": self.agent_name,
            "to_agent": request.get("from_agent"),
            "analysis": analysis,
            "recommendations": analysis.get("recommended_approach", ""),
            "confidence": self.last_confidence
        }
    
    # =========================================================================
    # DECISION EXPLAINABILITY
    # =========================================================================
    
    def explain_decision(self, action: Dict, context: Dict) -> str:
        """
        Generate human-readable explanation for a decision
        """
        prompt = f"""Explain this power grid control action in simple terms for an operator:

ACTION: {json.dumps(action, indent=2)}
CONTEXT: {json.dumps(context, indent=2)}

Provide a clear, concise explanation (2-3 sentences) covering:
1. What the action does
2. Why it's needed
3. Expected outcome

Return JSON: {{"explanation": "Your explanation here"}}"""

        try:
            content = self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = self._extract_json(content)
            return result.get("explanation", "Unable to generate explanation")
            
        except Exception as e:
            return f"Explanation unavailable: {str(e)}"
    
    # =========================================================================
    # MEMORY AND LEARNING
    # =========================================================================
    
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
                json.dump(self.successful_actions[-1000:], f, indent=2)
            
            failed_file = self.memory_dir / f"{self.agent_name}_failed.json"
            with open(failed_file, 'w') as f:
                json.dump(self.failed_actions[-500:], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory: {e}")
    
    def record_successful_action(self, context: Dict, action: Dict, outcome: Dict):
        """Record a successful action for future learning"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "action": action,
            "outcome": outcome,
            "agent": self.agent_name,
            "confidence": self.last_confidence
        }
        self.successful_actions.append(record)
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
        self._save_memory()
    
    def get_similar_past_actions(self, context: Dict, similarity_threshold: float = 0.6) -> List[Dict]:
        """Find similar past actions that succeeded"""
        similar = []
        
        for record in self.successful_actions[-100:]:  # Check recent history
            if self._is_similar_context(context, record.get("context", {}), similarity_threshold):
                similar.append(record)
        
        return sorted(similar, key=lambda x: x["timestamp"], reverse=True)[:5]
    
    def _is_similar_context(self, context1: Dict, context2: Dict, threshold: float) -> bool:
        """Check if two contexts are similar"""
        try:
            # Compare violation types
            v1_types = set(context1.get("violation_types", []))
            v2_types = set(context2.get("violation_types", []))
            
            if not v1_types and not v2_types:
                return True
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
            return 0.5  # Default to 50% with no data
        return len(self.successful_actions) / total
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary statistics of agent memory"""
        return {
            "agent_name": self.agent_name,
            "total_successful_actions": len(self.successful_actions),
            "total_failed_actions": len(self.failed_actions),
            "success_rate": self.get_success_rate(),
            "last_confidence": self.last_confidence,
            "collaboration_requests": len(self.collaboration_requests),
            "last_successful": self.successful_actions[-1]["timestamp"] if self.successful_actions else None,
            "last_failed": self.failed_actions[-1]["timestamp"] if self.failed_actions else None
        }
    
    def clear_memory(self):
        """Clear all memory (use with caution)"""
        self.successful_actions = []
        self.failed_actions = []
        self._save_memory()
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response text"""
        try:
            # Try direct parse first
            return json.loads(text)
        except:
            pass
        
        # Try to find JSON in text
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except:
            pass
        
        # Try array
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return {"actions": json.loads(text[start:end])}
        except:
            pass
        
        return None
    
    def _validate_action(self, action: Dict) -> Tuple[bool, str]:
        """Base validation for any action"""
        if not isinstance(action, dict):
            return False, "Action must be a dictionary"
        
        if "action_type" not in action:
            return False, "Missing action_type"
        
        return True, ""
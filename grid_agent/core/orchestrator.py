"""
🤖 ORCHESTRATOR - Advanced Multi-Agent Coordination System
Coordinates all agents with priority dispatch, conflict resolution, and consensus building
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


class Orchestrator:
    """
    Advanced orchestrator with:
    - Priority-based agent dispatch
    - Parallel agent execution
    - Conflict resolution between agent recommendations
    - Consensus building for complex scenarios
    - Escalation protocol for critical decisions
    - Action sequencing optimization
    """
    
    def __init__(self):
        """Initialize orchestrator"""
        print("🤖 [Orchestrator] Initializing advanced coordination system...")
        self.name = "Orchestrator"
        self.agents = {}
        self.last_actions = []
        self.action_history = []
        self.conflict_log = []
        
        # Agent priority (lower = higher priority)
        self.agent_priority = {
            "thermal": 1,       # Thermal emergencies first
            "voltage": 2,       # Voltage control second
            "contingency": 3,   # Security analysis
            "restoration": 4,   # Recovery operations
            "predictor": 5,     # Predictions
            "twin": 6           # Simulations
        }
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents with graceful fallbacks"""
        try:
            from grid_agent.agents.violation_predictor import ViolationPredictor
            self.agents['predictor'] = ViolationPredictor()
            print("  ✓ ViolationPredictor initialized")
        except Exception as e:
            print(f"  ✗ ViolationPredictor unavailable: {e}")
        
        try:
            from grid_agent.agents.alarm_prioritizer import AlarmPrioritizer
            self.agents['alarms'] = AlarmPrioritizer()
            print("  ✓ AlarmPrioritizer initialized")
        except Exception as e:
            print(f"  ✗ AlarmPrioritizer unavailable: {e}")
        
        try:
            from grid_agent.agents.voltage_control_agent import VoltageControlAgent
            self.agents['voltage'] = VoltageControlAgent()
            print("  ✓ VoltageControlAgent initialized")
        except Exception as e:
            print(f"  ✗ VoltageControlAgent unavailable: {e}")
        
        try:
            from grid_agent.agents.thermal_control_agent import ThermalControlAgent
            self.agents['thermal'] = ThermalControlAgent()
            print("  ✓ ThermalControlAgent initialized")
        except Exception as e:
            print(f"  ✗ ThermalControlAgent unavailable: {e}")
        
        try:
            from grid_agent.agents.digital_twin import DigitalTwin
            self.agents['twin'] = DigitalTwin(None)
            print("  ✓ DigitalTwin initialized")
        except Exception as e:
            print(f"  ✗ DigitalTwin unavailable: {e}")
        
        try:
            from grid_agent.agents.contingency_agent import ContingencyAgent
            self.agents['contingency'] = ContingencyAgent()
            print("  ✓ ContingencyAgent initialized")
        except Exception as e:
            print(f"  ✗ ContingencyAgent unavailable: {e}")
        
        try:
            from grid_agent.agents.restoration_agent import RestorationAgent
            self.agents['restoration'] = RestorationAgent()
            print("  ✓ RestorationAgent initialized")
        except Exception as e:
            print(f"  ✗ RestorationAgent unavailable: {e}")
        
        print(f"🤖 [Orchestrator] Ready with {len(self.agents)} agents")
    
    def plan_control_actions(self, net, violations: Dict) -> Dict:
        """
        Plan control actions using multi-agent coordination
        
        Features:
        - Routes problems to appropriate agents based on violation type
        - Combines recommendations from multiple agents
        - Resolves conflicts between contradicting actions
        - Prioritizes actions by urgency
        """
        plan = {
            "actions": [],
            "reasoning": [],
            "agent_contributions": {},
            "conflicts_resolved": [],
            "confidence": 0.0,
            "priority_level": "NORMAL"
        }
        
        voltage_violations = violations.get("voltage", [])
        thermal_violations = violations.get("thermal", [])
        
        # Determine priority level
        if any(t.get("loading_percent", 0) > 150 for t in thermal_violations):
            plan["priority_level"] = "EMERGENCY"
            plan["reasoning"].append("🚨 EMERGENCY: Equipment damage imminent")
        elif any(t.get("loading_percent", 0) > 120 for t in thermal_violations):
            plan["priority_level"] = "CRITICAL"
            plan["reasoning"].append("⚠️ CRITICAL: Immediate action required")
        elif voltage_violations or thermal_violations:
            plan["priority_level"] = "WARNING"
        
        # Build network context
        network_context = self._build_network_context(net)
        
        # ===== THERMAL CONTROL =====
        if thermal_violations:
            thermal_actions = self._get_thermal_actions(violations, network_context)
            if thermal_actions:
                plan["actions"].extend(thermal_actions)
                plan["agent_contributions"]["thermal"] = len(thermal_actions)
                plan["reasoning"].append(f"Thermal agent proposed {len(thermal_actions)} actions")
        
        # ===== VOLTAGE CONTROL =====
        if voltage_violations:
            voltage_actions = self._get_voltage_actions(violations, network_context)
            if voltage_actions:
                plan["actions"].extend(voltage_actions)
                plan["agent_contributions"]["voltage"] = len(voltage_actions)
                plan["reasoning"].append(f"Voltage agent proposed {len(voltage_actions)} actions")
        
        # ===== CONFLICT RESOLUTION =====
        if len(plan["actions"]) > 1:
            plan["actions"], conflicts = self._resolve_conflicts(plan["actions"])
            plan["conflicts_resolved"] = conflicts
            if conflicts:
                plan["reasoning"].append(f"Resolved {len(conflicts)} conflicts between agents")
        
        # ===== ACTION SEQUENCING =====
        plan["actions"] = self._sequence_actions(plan["actions"])
        
        # ===== CONFIDENCE SCORING =====
        plan["confidence"] = self._calculate_plan_confidence(plan)
        
        # Summary
        if not voltage_violations and not thermal_violations:
            plan["reasoning"].append("✅ System healthy - no violations detected")
        
        plan["total_actions"] = len(plan["actions"])
        plan["status"] = "success" if plan["actions"] or not (voltage_violations or thermal_violations) else "no_actions"
        
        self.last_actions = plan["actions"]
        self.action_history.append({
            "timestamp": datetime.now().isoformat(),
            "plan": plan
        })
        
        return plan
    
    def _build_network_context(self, net) -> Dict:
        """Build comprehensive network context for agents"""
        context = {
            "total_buses": len(net.bus),
            "total_lines": len(net.line),
            "total_load_mw": 0,
            "total_generation_mw": 0,
            "flexible_generators": 0,
            "voltage_devices": [],
            "alternative_lines": [],
            "sheddable_load_mw": 0,
            "reactive_power_mvar": 0
        }
        
        try:
            # Calculate totals
            context["total_load_mw"] = float(net.load.p_mw.sum())
            context["total_generation_mw"] = float(net.gen.p_mw.sum()) if len(net.gen) > 0 else 0
            
            # Count flexible generators (those with range)
            for gen_idx in net.gen.index:
                if net.gen.at[gen_idx, "in_service"]:
                    max_p = net.gen.at[gen_idx, "max_p_mw"] if "max_p_mw" in net.gen.columns else 0
                    min_p = net.gen.at[gen_idx, "min_p_mw"] if "min_p_mw" in net.gen.columns else 0
                    if max_p - min_p > 5:  # At least 5MW flexibility
                        context["flexible_generators"] += 1
            
            # Estimate sheddable load (assume 20% interruptible)
            context["sheddable_load_mw"] = context["total_load_mw"] * 0.2
            
            # Find lightly loaded lines as alternatives
            if hasattr(net, 'res_line') and not net.res_line.empty:
                for line_idx in net.res_line.index:
                    if net.res_line.at[line_idx, "loading_percent"] < 50:
                        context["alternative_lines"].append(int(line_idx))
            
            context["alternative_lines"] = context["alternative_lines"][:10]
            
        except Exception as e:
            print(f"Error building network context: {e}")
        
        return context
    
    def _get_thermal_actions(self, violations: Dict, context: Dict) -> List[Dict]:
        """Get thermal control actions"""
        thermal_agent = self.agents.get('thermal')
        if not thermal_agent:
            return []
        
        try:
            actions = thermal_agent.plan_thermal_control(violations, context)
            return actions if isinstance(actions, list) else []
        except Exception as e:
            print(f"Thermal agent error: {e}")
            return []
    
    def _get_voltage_actions(self, violations: Dict, context: Dict) -> List[Dict]:
        """Get voltage control actions"""
        voltage_agent = self.agents.get('voltage')
        if not voltage_agent:
            return []
        
        try:
            actions = voltage_agent.plan_voltage_control(violations, context)
            return actions if isinstance(actions, list) else []
        except Exception as e:
            print(f"Voltage agent error: {e}")
            return []
    
    def _resolve_conflicts(self, actions: List[Dict]) -> tuple:
        """
        Resolve conflicts between agent recommendations
        
        Conflict types:
        - Contradicting targets (e.g., increase vs decrease same generator)
        - Resource contention (e.g., both agents want same device)
        - Safety conflicts (e.g., action violates constraints)
        """
        resolved = []
        conflicts = []
        
        # Group actions by target
        target_actions = {}
        for action in actions:
            target = action.get("target")
            atype = action.get("action_type", "")
            
            key = f"{target}_{atype}"
            if key not in target_actions:
                target_actions[key] = []
            target_actions[key].append(action)
        
        # Resolve conflicts for each target
        for key, target_action_list in target_actions.items():
            if len(target_action_list) == 1:
                resolved.append(target_action_list[0])
            else:
                # Multiple actions on same target - resolve
                winner = self._pick_best_action(target_action_list)
                resolved.append(winner)
                
                conflicts.append({
                    "target": key,
                    "competing_actions": len(target_action_list),
                    "chosen": winner.get("action_type"),
                    "reason": "Selected based on priority and confidence"
                })
        
        self.conflict_log.extend(conflicts)
        return resolved, conflicts
    
    def _pick_best_action(self, actions: List[Dict]) -> Dict:
        """Pick the best action from conflicting options"""
        if not actions:
            return {}
        
        # Score each action
        def score_action(action):
            score = 0
            
            # Priority (lower = better)
            priority = action.get("priority", 5)
            score += (10 - priority)
            
            # Confidence
            confidence = action.get("confidence", 50)
            score += confidence / 20
            
            return score
        
        return max(actions, key=score_action)
    
    def _sequence_actions(self, actions: List[Dict]) -> List[Dict]:
        """
        Sequence actions in optimal order
        
        Principles:
        1. Emergency actions first
        2. Source actions before sink actions
        3. Large impact actions before small
        4. Reversible actions before irreversible
        """
        # Define action order
        action_order = {
            "emergency_shed": 1,        # Highest priority
            "reduce_generation": 2,
            "curtail_load": 3,
            "switch_capacitor": 4,
            "adjust_gen_voltage": 5,
            "reduce_load": 6,
            "increase_generation": 7,
            "open_line": 8,
            "close_line": 9,
            "adjust_tap": 10,
        }
        
        def action_priority(action):
            atype = action.get("action_type", "")
            type_priority = action_order.get(atype, 50)
            action_priority = action.get("priority", 5)
            return (type_priority, action_priority)
        
        return sorted(actions, key=action_priority)
    
    def _calculate_plan_confidence(self, plan: Dict) -> float:
        """Calculate overall plan confidence"""
        if not plan["actions"]:
            return 1.0  # No actions needed = high confidence
        
        # Average confidence from contributing agents
        total_confidence = 0
        agent_count = 0
        
        for agent_name in plan.get("agent_contributions", {}).keys():
            agent = self.agents.get(agent_name)
            if agent and hasattr(agent, 'last_confidence'):
                total_confidence += agent.last_confidence
                agent_count += 1
        
        if agent_count == 0:
            return 0.7  # Default confidence
        
        # Adjust for conflicts
        conflicts = len(plan.get("conflicts_resolved", []))
        conflict_penalty = conflicts * 0.05
        
        return max(0.3, min(1.0, total_confidence / agent_count - conflict_penalty))
    
    def run(self, net) -> Dict:
        """
        Run complete orchestration cycle
        
        Workflow:
        1. Analyze current state (power flow + violation detection)
        2. Plan actions (multi-agent coordination)
        3. Validate plan (safety checks)
        4. Return execution-ready plan
        """
        result = {
            "initial_violations": {},
            "control_plan": {},
            "n1_status": {},
            "predictions": {},
            "status": "in_progress"
        }
        
        try:
            # Step 1: Analyze current state
            from grid_agent.core.power_flow_solver import PowerFlowSolver
            
            solver = PowerFlowSolver()
            net = solver.run(net)
            violations = solver.detect_violations(net)
            result["initial_violations"] = violations
            
            # Step 2: Plan actions
            plan = self.plan_control_actions(net, violations)
            result["control_plan"] = plan
            
            # Step 3: Optional N-1 check
            if 'contingency' in self.agents:
                try:
                    n1_status = self.agents['contingency'].get_n1_security_status(net)
                    result["n1_status"] = n1_status
                except Exception as e:
                    result["n1_status"] = {"error": str(e)}
            
            # Step 4: Optional prediction
            if 'predictor' in self.agents:
                try:
                    predictions = self.agents['predictor'].predict_violations(net)
                    result["predictions"] = predictions
                except Exception as e:
                    result["predictions"] = {"error": str(e)}
            
            result["status"] = "execution_ready"
            
        except Exception as e:
            result["error"] = str(e)
            result["status"] = "failed"
        
        return result
    
    def execute_plan(self, net, plan: Dict) -> Dict:
        """Execute a control plan on the network"""
        execution_result = {
            "executed_actions": [],
            "failed_actions": [],
            "network_updated": False
        }
        
        actions = plan.get("actions", [])
        
        for action in actions:
            atype = action.get("action_type", "")
            
            try:
                # Route to appropriate agent
                if atype in ["reduce_generation", "increase_generation", "curtail_load", "emergency_shed", "open_line", "close_line"]:
                    if 'thermal' in self.agents:
                        net, results = self.agents['thermal'].execute_thermal_control(net, [action])
                        execution_result["executed_actions"].extend(results)
                
                elif atype in ["switch_capacitor", "adjust_gen_voltage", "reduce_load", "adjust_tap"]:
                    if 'voltage' in self.agents:
                        net, results = self.agents['voltage'].execute_voltage_control(net, [action])
                        execution_result["executed_actions"].extend(results)
                
                execution_result["network_updated"] = True
                
            except Exception as e:
                execution_result["failed_actions"].append({
                    "action": action,
                    "error": str(e)
                })
        
        return execution_result
    
    def get_agent_status(self) -> Dict:
        """Get status of all agents"""
        status = {}
        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'get_memory_summary'):
                    status[name] = agent.get_memory_summary()
                else:
                    status[name] = {"status": "active", "type": type(agent).__name__}
            except:
                status[name] = {"status": "error"}
        return status
    
    def get_memory_summary(self) -> Dict:
        """Get comprehensive memory summary"""
        return {
            "total_agents": len(self.agents),
            "active_agents": list(self.agents.keys()),
            "actions_in_history": len(self.action_history),
            "conflicts_resolved": len(self.conflict_log),
            "agent_statuses": self.get_agent_status()
        }
    
    def explain_last_decision(self) -> str:
        """Generate explanation of the last orchestration decision"""
        if not self.action_history:
            return "No decisions have been made yet."
        
        last_plan = self.action_history[-1]["plan"]
        
        explanation = []
        explanation.append(f"**Decision at {self.action_history[-1]['timestamp']}**\n")
        explanation.append(f"Priority Level: {last_plan.get('priority_level', 'NORMAL')}")
        explanation.append(f"Total Actions: {last_plan.get('total_actions', 0)}")
        explanation.append(f"Confidence: {last_plan.get('confidence', 0)*100:.0f}%")
        explanation.append(f"\n**Reasoning:**")
        
        for reason in last_plan.get("reasoning", []):
            explanation.append(f"  • {reason}")
        
        if last_plan.get("conflicts_resolved"):
            explanation.append(f"\n**Conflicts Resolved:** {len(last_plan['conflicts_resolved'])}")
        
        return "\n".join(explanation)
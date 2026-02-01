"""
Enhanced Voltage Control Agent
Advanced voltage stability and regulation with sophisticated LLM reasoning
"""

import json
from grid_agent.agents.base_agent import BaseAgent
from typing import Dict, List, Any, Tuple

# Optional ollama import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None


class VoltageControlAgent(BaseAgent):
    """
    Advanced voltage control agent with:
    - Multi-step voltage regulation planning
    - Reactive power optimization
    - Voltage stability assessment
    - Coordinated control with other agents
    """
    
    def __init__(self, model: str = "mistral"):
        super().__init__("voltage_control", model)
        
        # Voltage-specific thresholds
        self.v_min = 0.95  # Minimum acceptable voltage (p.u.)
        self.v_max = 1.05  # Maximum acceptable voltage (p.u.)
        self.v_critical_low = 0.90  # Critical low voltage
        self.v_critical_high = 1.10  # Critical high voltage
        
        # Control device priorities
        self.device_priority = [
            "tap_changer",      # First: transformer taps
            "capacitor_bank",   # Second: capacitor banks
            "generator_avr",    # Third: generator AVR
            "load_reduction"    # Last resort: load shedding
        ]
    
    def plan_voltage_control(self, violations: Dict, network_context: Dict) -> List[Dict]:
        """
        Plan comprehensive voltage control actions using chain-of-thought reasoning
        """
        voltage_violations = violations.get("voltage", [])
        
        if not voltage_violations:
            self.last_reasoning = "No voltage violations detected - system healthy"
            return []
        
        # Get similar past successful actions
        context = {
            "violation_types": ["voltage"],
            "severity": violations.get("summary", {}).get("severity_score", 0),
            "num_violations": len(voltage_violations)
        }
        past_actions = self.get_similar_past_actions(context)
        
        # Classify violations by severity
        critical = [v for v in voltage_violations if v.get("voltage_pu", 1.0) < self.v_critical_low or v.get("voltage_pu", 1.0) > self.v_critical_high]
        moderate = [v for v in voltage_violations if v not in critical]
        
        # Build comprehensive prompt with few-shot examples
        past_examples = ""
        if past_actions:
            past_examples = "PAST SUCCESSFUL ACTIONS (apply similar strategies):\n"
            for i, ex in enumerate(past_actions[:2], 1):
                past_examples += f"  Example {i}: {json.dumps(ex.get('action', {}))}\n"
        
        prompt = f"""You are an expert VOLTAGE CONTROL SPECIALIST for power grid operations.

===== CURRENT VOLTAGE EMERGENCY =====
CRITICAL violations (V < {self.v_critical_low} or V > {self.v_critical_high}): {len(critical)} buses
MODERATE violations ({self.v_min} < V < {self.v_max} outside): {len(moderate)} buses

DETAILED VIOLATIONS:
{json.dumps(voltage_violations[:10], indent=2)}

===== NETWORK CAPABILITIES =====
• Total buses: {network_context.get("total_buses", 0)}
• Available voltage control devices: {network_context.get("voltage_devices", [])}
• Generators with AVR: {network_context.get("flexible_generators", 0)}
• Critical buses (priority protection): {network_context.get("critical_buses", [])}
• Available reactive power: {network_context.get("reactive_power_mvar", 0)} MVAR

{past_examples}

===== VOLTAGE CONTROL HIERARCHY =====
Apply controls in this order (least disruptive first):
1. TAP CHANGERS - Adjust transformer taps (±2% per step)
2. CAPACITOR BANKS - Switch capacitor banks (0.5-2 MVAR steps)
3. GENERATOR AVR - Adjust generator voltage setpoints
4. LOAD REDUCTION - Reduce load (last resort, 5-20% reduction)

===== ACTION SPECIFICATIONS =====
Available actions and their JSON format:

1. adjust_tap: {{"action_type": "adjust_tap", "target": transformer_id, "value": tap_position}}
2. switch_capacitor: {{"action_type": "switch_capacitor", "target": bus_id, "value": mvar_amount}}
3. adjust_gen_voltage: {{"action_type": "adjust_gen_voltage", "target": gen_id, "value": new_voltage_pu}}
4. reduce_load: {{"action_type": "reduce_load", "target": bus_id, "value": reduction_factor}}

===== SAFETY CONSTRAINTS =====
• Never reduce voltage setpoint below 0.95 pu
• Never exceed generator Q limits
• Limit load reduction to 30% maximum
• Prioritize critical buses

RESPOND WITH ONLY VALID JSON - an array of actions with confidence:
{{
    "analysis": "Brief analysis of voltage situation",
    "actions": [
        {{"action_type": "...", "target": ..., "value": ..., "priority": 1, "reason": "why this action"}}
    ],
    "confidence": 85,
    "expected_improvement": "Expected voltage improvement description"
}}"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompts["analytical"]},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0, "num_predict": 1024}
            )
            
            content = response["message"]["content"]
            self.last_reasoning = content
            
            # Parse response
            result = self._extract_json(content)
            
            if result:
                self.last_confidence = result.get("confidence", 50) / 100.0
                actions = result.get("actions", [])
                
                # Validate and sort by priority
                valid_actions = []
                for action in actions:
                    is_valid, error = self.validate_voltage_action(action, network_context)
                    if is_valid:
                        valid_actions.append(action)
                    else:
                        print(f"Invalid voltage action: {error}")
                
                return sorted(valid_actions, key=lambda x: x.get("priority", 99))
            
            return []
            
        except Exception as e:
            self.last_reasoning = f"Error in voltage planning: {str(e)}"
            return self._get_fallback_actions(voltage_violations, network_context)
    
    def _get_fallback_actions(self, violations: List[Dict], context: Dict) -> List[Dict]:
        """Generate simple fallback actions if LLM fails"""
        actions = []
        
        for v in violations[:3]:  # Handle up to 3 violations
            bus_id = v.get("bus")
            voltage = v.get("voltage_pu", 1.0)
            
            if voltage < self.v_min:
                # Low voltage - add reactive power
                actions.append({
                    "action_type": "switch_capacitor",
                    "target": bus_id,
                    "value": 0.5,  # Add 0.5 MVAR
                    "priority": 1,
                    "reason": f"Boost voltage at bus {bus_id} from {voltage:.3f} pu"
                })
            elif voltage > self.v_max:
                # High voltage - reduce reactive power
                actions.append({
                    "action_type": "switch_capacitor",
                    "target": bus_id,
                    "value": -0.5,  # Remove 0.5 MVAR
                    "priority": 1,
                    "reason": f"Reduce voltage at bus {bus_id} from {voltage:.3f} pu"
                })
        
        return actions
    
    def validate_voltage_action(self, action: Dict, context: Dict = None) -> Tuple[bool, str]:
        """Validate voltage control action is feasible and safe"""
        atype = action.get("action_type")
        target = action.get("target")
        value = action.get("value", 0)
        
        # Basic validation
        if atype is None:
            return False, "Missing action_type"
        if target is None:
            return False, "Missing target"
        
        if atype == "switch_capacitor":
            if abs(value) > 10:  # Max 10 MVAR
                return False, f"Capacitor value must be within ±10 MVAR, got {value}"
            return True, ""
        
        elif atype == "adjust_tap":
            if abs(value) > 20:  # Max tap position
                return False, f"Tap position must be within ±20, got {value}"
            return True, ""
        
        elif atype == "adjust_gen_voltage":
            if value < 0.9 or value > 1.1:
                return False, f"Voltage setpoint must be 0.9-1.1 pu, got {value}"
            return True, ""
        
        elif atype == "reduce_load":
            if value < 0 or value > 0.3:
                return False, f"Load reduction must be 0-30%, got {value*100}%"
            return True, ""
        
        return False, f"Unknown action type: {atype}"
    
    def execute_voltage_control(self, net, actions: List[Dict]) -> Tuple[Any, List[str]]:
        """Execute voltage control actions on network"""
        executed = []
        errors = []
        
        for action in actions:
            atype = action.get("action_type")
            target = action.get("target")
            value = action.get("value", 0)
            
            try:
                if atype == "switch_capacitor":
                    # Add shunt capacitor
                    if hasattr(net, 'shunt') and target in net.bus.index:
                        import pandapower as pp
                        pp.create_shunt(net, target, q_mvar=-value, p_mw=0)
                        executed.append(f"Added {value} MVAR capacitor at bus {target}")
                
                elif atype == "adjust_gen_voltage":
                    if target in net.gen.index:
                        old_value = net.gen.at[target, "vm_pu"]
                        net.gen.at[target, "vm_pu"] = value
                        executed.append(f"Changed gen {target} voltage: {old_value:.3f} → {value:.3f} pu")
                
                elif atype == "reduce_load":
                    bus_loads = net.load[net.load.bus == target].index
                    for load in bus_loads:
                        old_load = net.load.at[load, "p_mw"]
                        net.load.at[load, "p_mw"] *= (1 - value)
                        executed.append(f"Reduced load {load}: {old_load:.1f} → {old_load*(1-value):.1f} MW")
                
                elif atype == "adjust_tap":
                    # Find transformer connected to target bus
                    if hasattr(net, 'trafo'):
                        for trafo_idx in net.trafo.index:
                            if net.trafo.at[trafo_idx, "hv_bus"] == target or net.trafo.at[trafo_idx, "lv_bus"] == target:
                                net.trafo.at[trafo_idx, "tap_pos"] = value
                                executed.append(f"Adjusted transformer {trafo_idx} tap to {value}")
                                break
                
            except Exception as e:
                errors.append(f"Failed to execute {atype} on {target}: {str(e)}")
        
        return net, executed if executed else errors
    
    def assess_voltage_stability(self, net) -> Dict:
        """Assess overall voltage stability of the network"""
        try:
            voltages = net.res_bus["vm_pu"].values
            
            return {
                "mean_voltage": float(voltages.mean()),
                "voltage_std": float(voltages.std()),
                "min_voltage": float(voltages.min()),
                "max_voltage": float(voltages.max()),
                "buses_below_0.95": int(sum(voltages < 0.95)),
                "buses_above_1.05": int(sum(voltages > 1.05)),
                "stability_index": float(1.0 - voltages.std() / 0.1),  # Lower std = more stable
                "overall_status": "STABLE" if voltages.std() < 0.02 else "MARGINAL" if voltages.std() < 0.05 else "UNSTABLE"
            }
        except:
            return {"error": "Could not assess voltage stability", "overall_status": "UNKNOWN"}
    
    def get_voltage_improvement_forecast(self, actions: List[Dict]) -> str:
        """Forecast expected improvement from proposed actions"""
        if not actions:
            return "No actions proposed"
        
        expected_improvement = 0
        for action in actions:
            atype = action.get("action_type")
            value = abs(action.get("value", 0))
            
            if atype == "switch_capacitor":
                expected_improvement += value * 0.5  # 0.5% per MVAR
            elif atype == "adjust_gen_voltage":
                expected_improvement += value * 2  # 2% per 0.01 pu change
            elif atype == "reduce_load":
                expected_improvement += value * 5  # 5% per 10% load reduction
        
        return f"Expected voltage improvement: ~{expected_improvement:.1f}% towards target range"
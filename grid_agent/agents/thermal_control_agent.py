"""
Enhanced Thermal Control Agent
Advanced thermal management with sophisticated LLM reasoning and emergency protocols
"""

import json
from grid_agent.agents.base_agent import BaseAgent
from typing import Dict, List, Tuple, Any

# Optional ollama import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None


class ThermalControlAgent(BaseAgent):
    """
    Advanced thermal control agent with:
    - Emergency overload response
    - Power flow optimization
    - Line criticality analysis
    - Coordinated generation redispatch
    """
    
    def __init__(self, model: str = "mistral"):
        super().__init__("thermal_control", model)
        
        # Thermal thresholds
        self.loading_normal = 80.0      # Normal operation limit
        self.loading_warning = 100.0    # Warning threshold
        self.loading_critical = 120.0   # Emergency action required
        self.loading_damage = 150.0     # Equipment damage risk
        
        # Action priorities (lower = higher priority)
        self.action_priority = {
            "generation_redispatch": 1,  # Preferred: shift power flow
            "load_curtailment": 2,       # Reduce demand
            "topology_switching": 3,      # Reconfigure network
            "emergency_load_shed": 4      # Last resort
        }
    
    def plan_thermal_control(self, violations: Dict, network_context: Dict) -> List[Dict]:
        """
        Plan comprehensive thermal control actions using chain-of-thought reasoning
        """
        thermal_violations = violations.get("thermal", [])
        
        if not thermal_violations:
            self.last_reasoning = "No thermal violations detected - all lines within limits"
            return []
        
        # Classify by severity
        emergency = [t for t in thermal_violations if t.get("loading_percent", 0) >= self.loading_damage]
        critical = [t for t in thermal_violations if self.loading_critical <= t.get("loading_percent", 0) < self.loading_damage]
        warning = [t for t in thermal_violations if self.loading_warning <= t.get("loading_percent", 0) < self.loading_critical]
        
        # Get past successful actions
        context = {
            "violation_types": ["thermal"],
            "severity": violations.get("summary", {}).get("severity_score", 0),
            "num_violations": len(thermal_violations)
        }
        past_actions = self.get_similar_past_actions(context)
        
        # Build examples
        past_examples = ""
        if past_actions:
            past_examples = "PAST SUCCESSFUL THERMAL ACTIONS:\n"
            for i, ex in enumerate(past_actions[:2], 1):
                past_examples += f"  {i}. {json.dumps(ex.get('action', {}))}\n"
        
        # Determine mode based on severity
        if emergency:
            system_prompt = self.system_prompts["emergency"]
            urgency = "🚨 EMERGENCY - EQUIPMENT DAMAGE IMMINENT"
        elif critical:
            system_prompt = self.system_prompts["emergency"]
            urgency = "⚠️ CRITICAL - Immediate action required"
        else:
            system_prompt = self.system_prompts["analytical"]
            urgency = "⚡ Warning - Preventive action recommended"
        
        prompt = f"""You are an expert THERMAL MANAGEMENT SPECIALIST for power grid operations.

===== {urgency} =====

THERMAL SITUATION:
• EMERGENCY (>150% loading, damage risk): {len(emergency)} lines
• CRITICAL (120-150% loading): {len(critical)} lines  
• WARNING (100-120% loading): {len(warning)} lines

OVERLOADED LINES (sorted by severity):
{json.dumps(sorted(thermal_violations, key=lambda x: -x.get('loading_percent', 0))[:10], indent=2)}

===== NETWORK CAPABILITIES =====
• Flexible generators for redispatch: {network_context.get("flexible_generators", 0)}
• Sheddable load available: {network_context.get("sheddable_load_mw", 0)} MW
• Alternative transmission paths: {network_context.get("alternative_lines", [])}
• Total network load: {network_context.get("total_load_mw", 0)} MW

{past_examples}

===== THERMAL CONTROL HIERARCHY =====
Apply in order (least disruptive first):
1. GENERATION REDISPATCH - Shift power from congested to uncongested paths
2. LOAD CURTAILMENT - Reduce interruptible loads (5-15% typically)
3. TOPOLOGY SWITCHING - Reconfigure to distribute loading
4. EMERGENCY LOAD SHED - Last resort for imminent damage

===== ACTION SPECIFICATIONS =====

1. reduce_generation: {{"action_type": "reduce_generation", "target": gen_id, "value": MW_reduction}}
2. increase_generation: {{"action_type": "increase_generation", "target": gen_id, "value": MW_increase}}  
3. curtail_load: {{"action_type": "curtail_load", "target": load_id, "value": reduction_factor_0_to_1}}
4. open_line: {{"action_type": "open_line", "target": line_id, "value": null}}
5. close_line: {{"action_type": "close_line", "target": line_id, "value": null}}
6. emergency_shed: {{"action_type": "emergency_shed", "target": bus_id, "value": MW_to_shed}}

===== SAFETY CONSTRAINTS =====
• Never exceed generator maximum output
• Maintain N-1 security where possible
• Don't shed more than 30% of any single bus load
• Keep network connected (no islanding)
• Prioritize critical loads

===== PHYSICS REMINDER =====
• Power flows from high to low voltage
• Reducing generation near congested line source helps
• Increasing generation near congested line sink helps
• P ≈ V²/X for approximate power flow

RESPOND WITH ONLY VALID JSON:
{{
    "emergency_assessment": "Brief assessment of thermal emergency",
    "root_cause": "Why are lines overloaded?",
    "actions": [
        {{"action_type": "...", "target": ..., "value": ..., "priority": 1, "expected_relief_mw": 10, "reason": "..."}}
    ],
    "confidence": 85,
    "risk_if_no_action": "What happens if we don't act?"
}}"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0, "num_predict": 1024}
            )
            
            content = response["message"]["content"]
            self.last_reasoning = content
            
            result = self._extract_json(content)
            
            if result:
                self.last_confidence = result.get("confidence", 50) / 100.0
                actions = result.get("actions", [])
                
                # Validate actions
                valid_actions = []
                for action in actions:
                    is_valid, error = self.validate_thermal_action(action, network_context)
                    if is_valid:
                        valid_actions.append(action)
                    else:
                        print(f"Invalid thermal action skipped: {error}")
                
                return sorted(valid_actions, key=lambda x: x.get("priority", 99))
            
            return []
            
        except Exception as e:
            self.last_reasoning = f"Error in thermal planning: {str(e)}"
            return self._get_fallback_actions(thermal_violations, network_context)
    
    def _get_fallback_actions(self, violations: List[Dict], context: Dict) -> List[Dict]:
        """Generate fallback actions if LLM fails"""
        actions = []
        
        # Sort by loading (worst first)
        sorted_violations = sorted(violations, key=lambda x: -x.get("loading_percent", 0))
        
        for v in sorted_violations[:3]:
            loading = v.get("loading_percent", 0)
            line_id = v.get("line")
            from_bus = v.get("from_bus")
            
            overload_pct = loading - 100
            
            if loading >= self.loading_damage:
                # Emergency: shed load
                actions.append({
                    "action_type": "emergency_shed",
                    "target": from_bus,
                    "value": overload_pct * 0.5,  # Shed proportional to overload
                    "priority": 1,
                    "reason": f"EMERGENCY: Line {line_id} at {loading:.0f}% - damage imminent"
                })
            elif loading >= self.loading_critical:
                # Critical: curtail load
                actions.append({
                    "action_type": "curtail_load",
                    "target": from_bus,
                    "value": min(0.2, overload_pct / 100),  # Up to 20% reduction
                    "priority": 2,
                    "reason": f"CRITICAL: Line {line_id} at {loading:.0f}%"
                })
            else:
                # Warning: reduce generation if possible
                actions.append({
                    "action_type": "reduce_generation",
                    "target": 0,  # Default to first generator
                    "value": overload_pct * 0.3,
                    "priority": 3,
                    "reason": f"WARNING: Line {line_id} at {loading:.0f}%"
                })
        
        return actions
    
    def validate_thermal_action(self, action: Dict, context: Dict = None) -> Tuple[bool, str]:
        """Validate thermal control action"""
        atype = action.get("action_type")
        target = action.get("target")
        value = action.get("value")
        
        if atype is None:
            return False, "Missing action_type"
        if target is None:
            return False, "Missing target"
        
        if atype == "reduce_generation":
            if value is None or value < 0:
                return False, "Invalid reduction value"
            if value > 100:  # Sanity check
                return False, f"Reduction too large: {value} MW"
            return True, ""
        
        elif atype == "increase_generation":
            if value is None or value < 0:
                return False, "Invalid increase value"
            if value > 100:
                return False, f"Increase too large: {value} MW"
            return True, ""
        
        elif atype == "curtail_load":
            if value is None or value < 0 or value > 1.0:
                return False, f"Curtailment factor must be 0-1, got {value}"
            return True, ""
        
        elif atype == "emergency_shed":
            if value is None or value < 0:
                return False, "Invalid shed amount"
            return True, ""
        
        elif atype in ["open_line", "close_line"]:
            return True, ""
        
        return False, f"Unknown thermal action: {atype}"
    
    def execute_thermal_control(self, net, actions: List[Dict]) -> Tuple[Any, List[str]]:
        """Execute thermal control actions on network"""
        executed = []
        errors = []
        
        for action in actions:
            atype = action.get("action_type")
            target = action.get("target")
            value = action.get("value", 0)
            
            try:
                if atype == "reduce_generation":
                    if target in net.gen.index:
                        old_val = net.gen.at[target, "p_mw"]
                        new_val = max(0, old_val - value)
                        net.gen.at[target, "p_mw"] = new_val
                        executed.append(f"Reduced gen {target}: {old_val:.1f} → {new_val:.1f} MW")
                
                elif atype == "increase_generation":
                    if target in net.gen.index:
                        old_val = net.gen.at[target, "p_mw"]
                        max_p = net.gen.at[target, "max_p_mw"] if "max_p_mw" in net.gen.columns else old_val + 50
                        new_val = min(max_p, old_val + value)
                        net.gen.at[target, "p_mw"] = new_val
                        executed.append(f"Increased gen {target}: {old_val:.1f} → {new_val:.1f} MW")
                
                elif atype == "curtail_load":
                    if target in net.load.index:
                        old_load = net.load.at[target, "p_mw"]
                        net.load.at[target, "p_mw"] *= (1 - value)
                        executed.append(f"Curtailed load {target}: {old_load:.1f} → {old_load*(1-value):.1f} MW ({value*100:.0f}%)")
                
                elif atype == "emergency_shed":
                    # Find loads at bus and shed them
                    bus_loads = net.load[net.load.bus == target].index
                    total_shed = 0
                    for load_idx in bus_loads:
                        old_load = net.load.at[load_idx, "p_mw"]
                        shed_amount = min(old_load, value / len(bus_loads))
                        net.load.at[load_idx, "p_mw"] -= shed_amount
                        total_shed += shed_amount
                    executed.append(f"Emergency shed at bus {target}: {total_shed:.1f} MW")
                
                elif atype == "open_line":
                    if target in net.line.index:
                        net.line.at[target, "in_service"] = False
                        executed.append(f"Opened line {target}")
                
                elif atype == "close_line":
                    if target in net.line.index:
                        net.line.at[target, "in_service"] = True
                        executed.append(f"Closed line {target}")
                
            except Exception as e:
                errors.append(f"Failed {atype} on {target}: {str(e)}")
        
        return net, executed if executed else errors
    
    def calculate_line_criticality(self, net) -> Dict[int, Dict]:
        """
        Calculate criticality metrics for each line
        Higher criticality = more important for grid stability
        """
        criticality = {}
        
        try:
            for line_idx in net.line.index:
                if not net.line.at[line_idx, "in_service"]:
                    continue
                
                loading = net.res_line.at[line_idx, "loading_percent"]
                p_flow = abs(net.res_line.at[line_idx, "p_from_mw"])
                
                # Calculate criticality score
                score = 0
                
                # Loading contribution (0-5)
                if loading > 100:
                    score += 5
                elif loading > 80:
                    score += 3
                elif loading > 60:
                    score += 1
                
                # Power flow contribution (0-3)
                if p_flow > 50:
                    score += 3
                elif p_flow > 20:
                    score += 2
                elif p_flow > 5:
                    score += 1
                
                # At risk of cascading (0-2)
                if loading > 90:
                    score += 2
                
                criticality[int(line_idx)] = {
                    "score": min(10, score),
                    "loading_percent": float(loading),
                    "power_flow_mw": float(p_flow),
                    "status": "CRITICAL" if score >= 7 else "HIGH" if score >= 5 else "MEDIUM" if score >= 3 else "LOW"
                }
        
        except Exception as e:
            pass
        
        return criticality
    
    def get_thermal_status_summary(self, net) -> Dict[str, Any]:
        """Get comprehensive thermal status summary"""
        try:
            loadings = net.res_line.loading_percent
            
            normal = sum(1 for l in loadings if l <= self.loading_normal)
            elevated = sum(1 for l in loadings if self.loading_normal < l <= self.loading_warning)
            warning = sum(1 for l in loadings if self.loading_warning < l <= self.loading_critical)
            critical = sum(1 for l in loadings if self.loading_critical < l <= self.loading_damage)
            emergency = sum(1 for l in loadings if l > self.loading_damage)
            
            return {
                "normal_lines": int(normal),
                "elevated_lines": int(elevated),
                "warning_lines": int(warning),
                "critical_lines": int(critical),
                "emergency_lines": int(emergency),
                "max_loading": float(loadings.max()),
                "avg_loading": float(loadings.mean()),
                "total_lines": len(loadings),
                "overloaded_count": int(warning + critical + emergency),
                "system_status": "EMERGENCY" if emergency > 0 else "CRITICAL" if critical > 0 else "WARNING" if warning > 0 else "NORMAL"
            }
        except:
            return {"error": "Could not compute thermal status", "system_status": "UNKNOWN"}
    
    def estimate_time_to_damage(self, loading_percent: float) -> str:
        """Estimate time to equipment damage based on thermal overload"""
        if loading_percent <= 100:
            return "No immediate risk"
        elif loading_percent <= 110:
            return "~30 minutes before thermal limit"
        elif loading_percent <= 120:
            return "~15 minutes before thermal limit"
        elif loading_percent <= 130:
            return "~5 minutes before thermal limit"
        elif loading_percent <= 150:
            return "~1-2 minutes - IMMEDIATE ACTION REQUIRED"
        else:
            return "SECONDS - DAMAGE IMMINENT"
"""
Restoration Agent
Handles grid recovery and service restoration after outages
"""

import json
import copy
from grid_agent.agents.base_agent import BaseAgent
from typing import Dict, List, Tuple, Any, Optional
import pandapower as pp

# Optional ollama import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None


class RestorationAgent(BaseAgent):
    """
    Grid Restoration and Recovery Agent
    
    Capabilities:
    - Plans optimal restoration sequences after outages
    - Prioritizes critical loads for restoration
    - Manages network reconfiguration
    - Coordinates black-start procedures
    """
    
    def __init__(self, model: str = "mistral"):
        super().__init__("restoration", model)
        
        # Load priority categories
        self.load_priorities = {
            "critical": 1,      # Hospitals, emergency services
            "essential": 2,     # Water treatment, communication
            "commercial": 3,    # Business districts
            "residential": 4,   # Residential areas
            "industrial": 5     # Industrial loads
        }
        
        # Restoration constraints
        self.max_load_pickup_mw = 20  # Max MW per restoration step
        self.voltage_margin = 0.03    # Voltage margin during restoration
        self.loading_margin = 0.2     # Loading margin during restoration
    
    def assess_outage(self, net) -> Dict:
        """
        Assess the current outage situation
        """
        assessment = {
            "affected_buses": [],
            "affected_lines": [],
            "affected_generators": [],
            "total_load_lost_mw": 0,
            "available_generation_mw": 0,
            "restoration_feasibility": "UNKNOWN"
        }
        
        # Find out-of-service lines
        for line_idx in net.line.index:
            if not net.line.at[line_idx, "in_service"]:
                assessment["affected_lines"].append({
                    "line": int(line_idx),
                    "from_bus": int(net.line.at[line_idx, "from_bus"]),
                    "to_bus": int(net.line.at[line_idx, "to_bus"])
                })
        
        # Find out-of-service generators
        for gen_idx in net.gen.index:
            if not net.gen.at[gen_idx, "in_service"]:
                assessment["affected_generators"].append({
                    "generator": int(gen_idx),
                    "bus": int(net.gen.at[gen_idx, "bus"]),
                    "capacity_mw": float(net.gen.at[gen_idx, "p_mw"])
                })
        
        # Find isolated buses (no voltage)
        try:
            pp.runpp(net, enforce_q_lims=False)
            for bus_idx in net.res_bus.index:
                if net.res_bus.at[bus_idx, "vm_pu"] == 0 or net.res_bus.at[bus_idx, "vm_pu"] != net.res_bus.at[bus_idx, "vm_pu"]:  # NaN check
                    assessment["affected_buses"].append(int(bus_idx))
        except:
            # Power flow failed - significant outage
            assessment["affected_buses"] = list(net.bus.index)
        
        # Calculate lost load
        for load_idx in net.load.index:
            load_bus = net.load.at[load_idx, "bus"]
            if load_bus in assessment["affected_buses"]:
                assessment["total_load_lost_mw"] += net.load.at[load_idx, "p_mw"]
        
        # Calculate available generation
        for gen_idx in net.gen.index:
            if net.gen.at[gen_idx, "in_service"]:
                assessment["available_generation_mw"] += net.gen.at[gen_idx, "p_mw"]
        
        # Assess feasibility
        if not assessment["affected_buses"] and not assessment["affected_lines"]:
            assessment["restoration_feasibility"] = "NOT_NEEDED"
        elif assessment["available_generation_mw"] > assessment["total_load_lost_mw"]:
            assessment["restoration_feasibility"] = "FEASIBLE"
        else:
            assessment["restoration_feasibility"] = "PARTIAL"  # May need load shedding
        
        return assessment
    
    def plan_restoration_sequence(self, net, outage_assessment: Dict) -> Dict:
        """
        Plan optimal restoration sequence
        """
        plan = {
            "phases": [],
            "total_steps": 0,
            "estimated_time_minutes": 0,
            "priority_loads": [],
            "risks": []
        }
        
        if outage_assessment["restoration_feasibility"] == "NOT_NEEDED":
            plan["status"] = "NO_RESTORATION_NEEDED"
            return plan
        
        # Phase 1: Restore transmission backbone
        backbone_phase = {
            "phase": 1,
            "name": "Backbone Restoration",
            "description": "Restore main transmission paths",
            "actions": [],
            "estimated_time_minutes": 15
        }
        
        for line_info in outage_assessment["affected_lines"][:5]:
            backbone_phase["actions"].append({
                "action": "close_line",
                "target": line_info["line"],
                "description": f"Energize line {line_info['line']} ({line_info['from_bus']}-{line_info['to_bus']})"
            })
        
        if backbone_phase["actions"]:
            plan["phases"].append(backbone_phase)
        
        # Phase 2: Restore generators
        gen_phase = {
            "phase": 2,
            "name": "Generator Synchronization",
            "description": "Bring generators online",
            "actions": [],
            "estimated_time_minutes": 20
        }
        
        for gen_info in outage_assessment["affected_generators"]:
            gen_phase["actions"].append({
                "action": "start_generator",
                "target": gen_info["generator"],
                "description": f"Synchronize generator {gen_info['generator']} ({gen_info['capacity_mw']:.0f} MW)"
            })
        
        if gen_phase["actions"]:
            plan["phases"].append(gen_phase)
        
        # Phase 3: Load pickup (prioritized)
        load_phase = {
            "phase": 3,
            "name": "Load Restoration",
            "description": "Restore loads in priority order",
            "actions": [],
            "estimated_time_minutes": 30
        }
        
        # Get prioritized loads
        priority_loads = self._prioritize_loads(net, outage_assessment["affected_buses"])
        plan["priority_loads"] = priority_loads
        
        cumulative_load = 0
        for load_info in priority_loads:
            if cumulative_load + load_info["load_mw"] <= outage_assessment["available_generation_mw"]:
                load_phase["actions"].append({
                    "action": "restore_load",
                    "target": load_info["bus"],
                    "load_mw": load_info["load_mw"],
                    "priority": load_info["priority"],
                    "description": f"Restore {load_info['load_mw']:.1f} MW at bus {load_info['bus']} ({load_info['category']})"
                })
                cumulative_load += load_info["load_mw"]
        
        if load_phase["actions"]:
            plan["phases"].append(load_phase)
        
        # Calculate totals
        plan["total_steps"] = sum(len(p["actions"]) for p in plan["phases"])
        plan["estimated_time_minutes"] = sum(p["estimated_time_minutes"] for p in plan["phases"])
        
        # Get AI enhancement of plan
        plan = self._enhance_plan_with_ai(plan, outage_assessment)
        
        return plan
    
    def _prioritize_loads(self, net, affected_buses: List[int]) -> List[Dict]:
        """
        Prioritize loads for restoration based on criticality
        """
        loads = []
        
        for load_idx in net.load.index:
            load_bus = net.load.at[load_idx, "bus"]
            load_mw = net.load.at[load_idx, "p_mw"]
            
            if load_bus in affected_buses or not affected_buses:
                # Assign priority (in real system, this would be from database)
                # For demo, use load size as proxy
                if load_mw > 50:
                    category = "industrial"
                    priority = 5
                elif load_mw > 20:
                    category = "commercial"
                    priority = 3
                elif load_mw > 5:
                    category = "essential"
                    priority = 2
                else:
                    category = "residential"
                    priority = 4
                
                loads.append({
                    "load_idx": int(load_idx),
                    "bus": int(load_bus),
                    "load_mw": float(load_mw),
                    "category": category,
                    "priority": priority
                })
        
        # Sort by priority (lower = higher priority)
        loads.sort(key=lambda x: (x["priority"], -x["load_mw"]))
        
        return loads
    
    def _enhance_plan_with_ai(self, plan: Dict, assessment: Dict) -> Dict:
        """
        Use LLM to enhance restoration plan with additional insights
        """
        prompt = f"""Review this power grid restoration plan and identify risks or improvements.

OUTAGE ASSESSMENT:
- Lost load: {assessment.get('total_load_lost_mw', 0):.1f} MW
- Affected buses: {len(assessment.get('affected_buses', []))}
- Affected lines: {len(assessment.get('affected_lines', []))}

RESTORATION PLAN:
{json.dumps(plan['phases'], indent=2)}

Identify:
1. Top 3 risks during restoration
2. Any improvements to the sequence

Return JSON:
{{
    "risks": ["risk1", "risk2", "risk3"],
    "improvements": ["improvement1", "improvement2"],
    "confidence_score": 85
}}"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 512}
            )
            
            result = self._extract_json(response["message"]["content"])
            if result:
                plan["risks"] = result.get("risks", [])
                plan["improvements"] = result.get("improvements", [])
                self.last_confidence = result.get("confidence_score", 70) / 100.0
            
        except Exception as e:
            plan["risks"] = ["Could not perform AI risk analysis"]
        
        return plan
    
    def execute_restoration_step(self, net, step: Dict) -> Tuple[Any, bool, str]:
        """
        Execute a single restoration step
        
        Returns:
            (network, success, message)
        """
        action = step.get("action")
        target = step.get("target")
        
        try:
            if action == "close_line":
                if target in net.line.index:
                    net.line.at[target, "in_service"] = True
                    pp.runpp(net)
                    return net, True, f"Successfully energized line {target}"
            
            elif action == "start_generator":
                if target in net.gen.index:
                    net.gen.at[target, "in_service"] = True
                    pp.runpp(net)
                    return net, True, f"Successfully synchronized generator {target}"
            
            elif action == "restore_load":
                # Loads are typically restored by closing bus switches
                # In pandapower, we can adjust load scaling
                bus_loads = net.load[net.load.bus == target].index
                for load_idx in bus_loads:
                    net.load.at[load_idx, "scaling"] = 1.0  # Full load
                pp.runpp(net)
                return net, True, f"Successfully restored load at bus {target}"
            
            else:
                return net, False, f"Unknown action: {action}"
                
        except Exception as e:
            return net, False, f"Step failed: {str(e)}"
    
    def recommend_reconfiguration(self, net, isolated_buses: List[int]) -> List[Dict]:
        """
        Recommend network reconfiguration to restore isolated areas
        """
        recommendations = []
        
        # Find tie switches that could restore isolated buses
        for line_idx in net.line.index:
            if not net.line.at[line_idx, "in_service"]:
                from_bus = net.line.at[line_idx, "from_bus"]
                to_bus = net.line.at[line_idx, "to_bus"]
                
                # Check if this line could help
                if from_bus in isolated_buses or to_bus in isolated_buses:
                    recommendations.append({
                        "action": "close_tie_switch",
                        "line": int(line_idx),
                        "from_bus": int(from_bus),
                        "to_bus": int(to_bus),
                        "reason": f"Could restore connection to isolated buses"
                    })
        
        return recommendations[:5]  # Top 5 recommendations
    
    def get_restoration_status(self, plan: Dict, completed_steps: int) -> Dict:
        """
        Get current restoration progress
        """
        total_steps = plan.get("total_steps", 1)
        progress = min(100, completed_steps / total_steps * 100)
        
        current_phase = 1
        steps_in_phases = 0
        for phase in plan.get("phases", []):
            steps_in_phases += len(phase.get("actions", []))
            if completed_steps <= steps_in_phases:
                current_phase = phase.get("phase", 1)
                break
        
        return {
            "progress_percent": round(progress, 1),
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "current_phase": current_phase,
            "current_phase_name": plan.get("phases", [{}])[current_phase-1].get("name", "Unknown") if plan.get("phases") else "Unknown",
            "status": "COMPLETE" if progress >= 100 else "IN_PROGRESS"
        }

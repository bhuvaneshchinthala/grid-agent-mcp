"""
Digital Twin System
Creates a simulation replica of the real power grid
Enables safe testing of control strategies
"""

import pandapower as pp
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json
from pathlib import Path
import copy



class DigitalTwin:
    """
    Digital twin of the power grid
    Allows safe simulation of scenarios without affecting real operations
    """
    
    def __init__(self, base_network):
        self.base_network = copy.deepcopy(base_network)
        self.simulation_history = []
        self.scenario_results = {}
    
    def create_scenario(self, scenario_name: str, modifications: Dict) -> Tuple[bool, str, Any]:
        """
        Create a simulation scenario with modifications
        
        Args:
            scenario_name: Name of scenario
            modifications: Dictionary of network modifications
                {
                    "outages": [list of line/gen indices],
                    "load_increase": {bus: percentage},
                    "generation_change": {gen: MW_change}
                }
        
        Returns:
            (success, message, modified_network)
        """
        try:
            net = copy.deepcopy(self.base_network)
            
            # Apply outages
            for element_idx in modifications.get("outages", []):
                if element_idx in net.line.index:
                    net.line.at[element_idx, "in_service"] = False
                elif element_idx in net.gen.index:
                    net.gen.at[element_idx, "in_service"] = False
            
            # Apply load increases
            for bus, increase_percent in modifications.get("load_increase", {}).items():
                bus_loads = net.load[net.load.bus == bus].index
                for load_idx in bus_loads:
                    net.load.at[load_idx, "p_mw"] *= (1 + increase_percent / 100)
            
            # Apply generation changes
            for gen_idx, mw_change in modifications.get("generation_change", {}).items():
                if gen_idx in net.gen.index:
                    net.gen.at[gen_idx, "p_mw"] += mw_change
            
            return True, "Scenario created successfully", net
        
        except Exception as e:
            return False, f"Error creating scenario: {str(e)}", None
    
    def simulate_scenario(self, scenario_name: str, net, control_actions: List[Dict] = None) -> Dict:
        """
        Simulate a scenario with optional control actions
        
        Returns:
            Simulation results including violations and metrics
        """
        result = {
            "scenario": scenario_name,
            "timestamp": datetime.now().isoformat(),
            "before_control": {},
            "after_control": {},
            "improvement": {},
            "actions_applied": control_actions or [],
            "feasible": False
        }
        
        try:
            # Before control actions
            pp.runpp(net)
            result["before_control"] = self._extract_metrics(net)
            
            # Apply control actions if provided
            if control_actions:
                net = self._apply_actions(net, control_actions)
                pp.runpp(net)
                result["after_control"] = self._extract_metrics(net)
                
                # Calculate improvement
                result["improvement"] = self._calculate_improvement(
                    result["before_control"],
                    result["after_control"]
                )
            
            result["feasible"] = result["before_control"]["violations"] >= 0
            
            # Store result
            self.scenario_results[scenario_name] = result
            self.simulation_history.append(result)
            
            return result
        
        except Exception as e:
            result["error"] = str(e)
            return result
    
    def _extract_metrics(self, net) -> Dict:
        """Extract key metrics from network"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "voltage_violations": len([v for v in net.res_bus.vm_pu if v < 0.95 or v > 1.05]),
            "thermal_violations": len([l for l in net.res_line.loading_percent if l > 100]),
            "max_line_loading": float(net.res_line.loading_percent.max()) if len(net.res_line) > 0 else 0,
            "min_voltage": float(net.res_bus.vm_pu.min()),
            "max_voltage": float(net.res_bus.vm_pu.max()),
            "violations": 0,
            "power_balance_error": 0
        }
        
        metrics["violations"] = metrics["voltage_violations"] + metrics["thermal_violations"]
        
        return metrics
    
    def _apply_actions(self, net, actions: List[Dict]):
        """Apply control actions to network"""
        for action in actions:
            atype = action.get("action_type")
            
            if atype == "curtail_load":
                bus = action["target"]
                factor = action["value"]
                loads = net.load[net.load.bus == bus].index
                for l in loads:
                    net.load.at[l, "p_mw"] *= factor
            
            elif atype == "reduce_generation":
                gen = action["target"]
                factor = action["value"]
                if gen in net.gen.index:
                    net.gen.at[gen, "p_mw"] *= factor
            
            elif atype == "switch_line":
                line = action["target"]
                state = action["value"]
                if line in net.line.index:
                    net.line.at[line, "in_service"] = (state == "close")
        
        return net
    
    def _calculate_improvement(self, before: Dict, after: Dict) -> Dict:
        """Calculate improvement from before to after"""
        return {
            "voltage_improvement": before["voltage_violations"] - after["voltage_violations"],
            "thermal_improvement": before["thermal_violations"] - after["thermal_violations"],
            "total_improvement": before["violations"] - after["violations"],
            "improvement_percent": ((before["violations"] - after["violations"]) / 
                                   max(1, before["violations"])) * 100 if before["violations"] > 0 else 0,
            "voltage_margin_improvement": after["min_voltage"] - before["min_voltage"],
            "loading_reduction": before["max_line_loading"] - after["max_line_loading"]
        }
    
    def compare_strategies(self, scenario_name: str, net, 
                          strategy_a: List[Dict], 
                          strategy_b: List[Dict]) -> Dict:
        """
        Compare two different control strategies
        Helps operators choose best approach
        """
        net_a = copy.deepcopy(net)
        net_b = copy.deepcopy(net)

        
        result_a = self.simulate_scenario(f"{scenario_name}_strategy_a", net_a, strategy_a)
        result_b = self.simulate_scenario(f"{scenario_name}_strategy_b", net_b, strategy_b)
        
        return {
            "scenario": scenario_name,
            "strategy_a": {
                "actions": len(strategy_a),
                "improvements": result_a.get("improvement", {}),
                "final_violations": result_a.get("after_control", {}).get("violations", 0)
            },
            "strategy_b": {
                "actions": len(strategy_b),
                "improvements": result_b.get("improvement", {}),
                "final_violations": result_b.get("after_control", {}).get("violations", 0)
            },
            "recommendation": self._recommend_strategy(result_a, result_b),
            "comparison_metrics": self._compare_metrics(result_a, result_b)
        }
    
    def _recommend_strategy(self, result_a: Dict, result_b: Dict) -> str:
        """Recommend best strategy"""
        violations_a = result_a.get("after_control", {}).get("violations", 999)
        violations_b = result_b.get("after_control", {}).get("violations", 999)
        actions_a = len(result_a.get("actions_applied", []))
        actions_b = len(result_b.get("actions_applied", []))
        
        if violations_a < violations_b:
            return f"Strategy A is better ({violations_a} vs {violations_b} violations)"
        elif violations_b < violations_a:
            return f"Strategy B is better ({violations_b} vs {violations_a} violations)"
        else:
            if actions_a < actions_b:
                return f"Strategy A (fewer actions: {actions_a} vs {actions_b})"
            else:
                return f"Strategy B (fewer actions: {actions_b} vs {actions_a})"
    
    def _compare_metrics(self, result_a: Dict, result_b: Dict) -> Dict:
        """Compare detailed metrics"""
        return {
            "strategy_a": result_a.get("after_control", {}),
            "strategy_b": result_b.get("after_control", {}),
            "difference": {
                "violations": (result_a.get("after_control", {}).get("violations", 0) -
                             result_b.get("after_control", {}).get("violations", 0)),
                "actions_used": (len(result_a.get("actions_applied", [])) -
                               len(result_b.get("actions_applied", [])))
            }
        }
    
    def sensitivity_analysis(self, net, parameter: str, 
                            values: List[float]) -> Dict[float, Dict]:
        """
        Sensitivity analysis: how does changing a parameter affect violations?
        
        Args:
            net: Network to analyze
            parameter: "load_multiplier", "gen_reduction", etc.
            values: List of parameter values to test
        
        Returns:
            Dictionary mapping parameter values to results
        """
        results = {}
        
        for value in values:
            test_net = copy.deepcopy(net)
            
            # Apply parameter change
            if parameter == "load_multiplier":
                test_net.load.p_mw *= value
            elif parameter == "gen_reduction":
                test_net.gen.p_mw *= (1 - value)
            
            pp.runpp(test_net)
            results[value] = self._extract_metrics(test_net)
        
        return results
    
    def get_simulation_report(self) -> Dict:
        """Get report of all simulations run"""
        return {
            "total_scenarios": len(self.scenario_results),
            "total_simulations": len(self.simulation_history),
            "scenarios": list(self.scenario_results.keys()),
            "successful_scenarios": sum(1 for r in self.scenario_results.values() if r.get("feasible")),
            "average_improvement": self._calculate_avg_improvement()
        }
    
    def _calculate_avg_improvement(self) -> float:
        """Calculate average improvement across scenarios"""
        improvements = [r.get("improvement", {}).get("total_improvement", 0) 
                       for r in self.scenario_results.values()]
        return sum(improvements) / len(improvements) if improvements else 0
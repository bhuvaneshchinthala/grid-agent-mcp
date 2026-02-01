"""
N-1 Contingency Analysis Agent
Proactively tests "what if" scenarios to identify grid vulnerabilities
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


class ContingencyAgent(BaseAgent):
    """
    Advanced N-1 Contingency Analysis Agent
    
    Capabilities:
    - Simulates single component failures (N-1 analysis)
    - Identifies weak points before they fail
    - Ranks contingencies by severity
    - Recommends preventive strengthening actions
    """
    
    def __init__(self, model: str = "mistral"):
        super().__init__("contingency_analysis", model)
        
        # Contingency analysis settings
        self.max_contingencies = 50  # Max contingencies to analyze
        self.severity_threshold = 80  # Loading % to flag as critical
        self.voltage_min = 0.90  # Voltage limit for contingency analysis
        self.voltage_max = 1.10
    
    def run_n1_analysis(self, net, contingency_types: List[str] = None) -> Dict:
        """
        Run comprehensive N-1 contingency analysis
        
        Args:
            net: pandapower network
            contingency_types: List of ["line", "gen", "trafo"] to analyze
            
        Returns:
            Complete contingency analysis results
        """
        if contingency_types is None:
            contingency_types = ["line", "gen"]
        
        results = {
            "summary": {},
            "critical_contingencies": [],
            "all_contingencies": [],
            "most_vulnerable_components": [],
            "recommendations": []
        }
        
        contingencies_analyzed = 0
        critical_count = 0
        
        # Analyze line contingencies
        if "line" in contingency_types:
            for line_idx in net.line.index[:self.max_contingencies]:
                if not net.line.at[line_idx, "in_service"]:
                    continue
                
                cont_result = self._simulate_contingency(net, "line", line_idx)
                results["all_contingencies"].append(cont_result)
                contingencies_analyzed += 1
                
                if cont_result["severity"] >= 7:
                    results["critical_contingencies"].append(cont_result)
                    critical_count += 1
        
        # Analyze generator contingencies
        if "gen" in contingency_types:
            for gen_idx in net.gen.index[:20]:  # Limit generator contingencies
                if not net.gen.at[gen_idx, "in_service"]:
                    continue
                
                cont_result = self._simulate_contingency(net, "gen", gen_idx)
                results["all_contingencies"].append(cont_result)
                contingencies_analyzed += 1
                
                if cont_result["severity"] >= 7:
                    results["critical_contingencies"].append(cont_result)
                    critical_count += 1
        
        # Analyze transformer contingencies
        if "trafo" in contingency_types and hasattr(net, 'trafo') and len(net.trafo) > 0:
            for trafo_idx in net.trafo.index[:10]:
                if not net.trafo.at[trafo_idx, "in_service"]:
                    continue
                
                cont_result = self._simulate_contingency(net, "trafo", trafo_idx)
                results["all_contingencies"].append(cont_result)
                contingencies_analyzed += 1
                
                if cont_result["severity"] >= 7:
                    results["critical_contingencies"].append(cont_result)
                    critical_count += 1
        
        # Sort by severity
        results["critical_contingencies"] = sorted(
            results["critical_contingencies"],
            key=lambda x: -x["severity"]
        )[:10]  # Keep top 10
        
        # Identify most vulnerable components
        results["most_vulnerable_components"] = self._identify_vulnerable_components(
            results["all_contingencies"]
        )
        
        # Generate summary
        results["summary"] = {
            "total_contingencies_analyzed": contingencies_analyzed,
            "critical_contingencies": critical_count,
            "n1_secure": critical_count == 0,
            "worst_case_severity": max([c["severity"] for c in results["all_contingencies"]], default=0),
            "average_severity": sum([c["severity"] for c in results["all_contingencies"]]) / max(1, contingencies_analyzed)
        }
        
        # Get AI recommendations
        results["recommendations"] = self._get_ai_recommendations(results)
        
        self.last_confidence = 0.9 if critical_count == 0 else 0.7
        
        return results
    
    def _simulate_contingency(self, net, element_type: str, element_idx: int) -> Dict:
        """
        Simulate a single contingency (N-1 event)
        """
        # Create copy of network
        net_copy = copy.deepcopy(net)
        
        # Apply contingency
        original_state = None
        element_name = f"{element_type}_{element_idx}"
        
        try:
            if element_type == "line":
                original_state = net_copy.line.at[element_idx, "in_service"]
                net_copy.line.at[element_idx, "in_service"] = False
                from_bus = net.line.at[element_idx, "from_bus"]
                to_bus = net.line.at[element_idx, "to_bus"]
                element_name = f"Line {element_idx} ({from_bus}-{to_bus})"
            
            elif element_type == "gen":
                original_state = net_copy.gen.at[element_idx, "in_service"]
                net_copy.gen.at[element_idx, "in_service"] = False
                gen_bus = net.gen.at[element_idx, "bus"]
                gen_power = net.gen.at[element_idx, "p_mw"]
                element_name = f"Gen {element_idx} (Bus {gen_bus}, {gen_power:.0f} MW)"
            
            elif element_type == "trafo":
                original_state = net_copy.trafo.at[element_idx, "in_service"]
                net_copy.trafo.at[element_idx, "in_service"] = False
                element_name = f"Transformer {element_idx}"
            
            # Run power flow
            pp.runpp(net_copy, enforce_q_lims=False)
            
            # Analyze results
            violations = self._check_violations(net_copy)
            severity = self._calculate_severity(violations)
            
            return {
                "contingency_id": f"{element_type}_{element_idx}",
                "element_type": element_type,
                "element_idx": element_idx,
                "element_name": element_name,
                "converged": True,
                "severity": severity,
                "violations": violations,
                "max_loading": float(net_copy.res_line.loading_percent.max()),
                "min_voltage": float(net_copy.res_bus.vm_pu.min()),
                "max_voltage": float(net_copy.res_bus.vm_pu.max()),
                "status": "CRITICAL" if severity >= 7 else "WARNING" if severity >= 4 else "OK"
            }
            
        except Exception as e:
            # Power flow did not converge - serious contingency!
            return {
                "contingency_id": f"{element_type}_{element_idx}",
                "element_type": element_type,
                "element_idx": element_idx,
                "element_name": element_name,
                "converged": False,
                "severity": 10,  # Maximum severity
                "violations": {"error": str(e)},
                "status": "BLACKOUT_RISK",
                "error": str(e)
            }
    
    def _check_violations(self, net) -> Dict:
        """Check for violations after contingency"""
        violations = {
            "voltage_low": [],
            "voltage_high": [],
            "thermal": [],
            "total": 0
        }
        
        # Voltage violations
        for bus_idx in net.res_bus.index:
            vm = net.res_bus.at[bus_idx, 'vm_pu']
            if vm < self.voltage_min:
                violations["voltage_low"].append({
                    "bus": int(bus_idx),
                    "voltage": float(vm)
                })
            elif vm > self.voltage_max:
                violations["voltage_high"].append({
                    "bus": int(bus_idx),
                    "voltage": float(vm)
                })
        
        # Thermal violations
        for line_idx in net.res_line.index:
            loading = net.res_line.at[line_idx, 'loading_percent']
            if loading > 100:
                violations["thermal"].append({
                    "line": int(line_idx),
                    "loading": float(loading)
                })
        
        violations["total"] = (
            len(violations["voltage_low"]) + 
            len(violations["voltage_high"]) + 
            len(violations["thermal"])
        )
        
        return violations
    
    def _calculate_severity(self, violations: Dict) -> float:
        """Calculate severity score (0-10) based on violations"""
        severity = 0
        
        # Voltage violations
        for v in violations.get("voltage_low", []):
            if v["voltage"] < 0.85:
                severity += 3
            elif v["voltage"] < 0.90:
                severity += 2
            else:
                severity += 1
        
        for v in violations.get("voltage_high", []):
            if v["voltage"] > 1.15:
                severity += 3
            elif v["voltage"] > 1.10:
                severity += 2
            else:
                severity += 1
        
        # Thermal violations
        for t in violations.get("thermal", []):
            if t["loading"] > 150:
                severity += 4
            elif t["loading"] > 120:
                severity += 2
            else:
                severity += 1
        
        return min(10, severity)
    
    def _identify_vulnerable_components(self, all_contingencies: List[Dict]) -> List[Dict]:
        """Identify which components are most affected by contingencies"""
        # Track which lines appear overloaded after contingencies
        line_vulnerability = {}
        
        for cont in all_contingencies:
            for thermal in cont.get("violations", {}).get("thermal", []):
                line_id = thermal.get("line")
                if line_id not in line_vulnerability:
                    line_vulnerability[line_id] = {"count": 0, "max_loading": 0}
                line_vulnerability[line_id]["count"] += 1
                line_vulnerability[line_id]["max_loading"] = max(
                    line_vulnerability[line_id]["max_loading"],
                    thermal.get("loading", 0)
                )
        
        # Sort by vulnerability
        vulnerable = [
            {"line_id": lid, **data}
            for lid, data in line_vulnerability.items()
        ]
        vulnerable.sort(key=lambda x: (-x["count"], -x["max_loading"]))
        
        return vulnerable[:5]  # Top 5 vulnerable
    
    def _get_ai_recommendations(self, results: Dict) -> List[str]:
        """Get AI-powered recommendations based on contingency analysis"""
        critical = results.get("critical_contingencies", [])
        vulnerable = results.get("most_vulnerable_components", [])
        
        if not critical and not vulnerable:
            return ["System is N-1 secure. No immediate action required."]
        
        prompt = f"""Based on N-1 contingency analysis results, provide recommendations.

CRITICAL CONTINGENCIES (would cause violations):
{json.dumps(critical[:5], indent=2)}

MOST VULNERABLE COMPONENTS (frequently overloaded after contingencies):
{json.dumps(vulnerable[:3], indent=2)}

Provide 3-5 specific, actionable recommendations to improve grid resilience.
Return JSON: {{"recommendations": ["rec1", "rec2", ...]}}"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 512}
            )
            
            result = self._extract_json(response["message"]["content"])
            if result and "recommendations" in result:
                return result["recommendations"]
            
        except Exception as e:
            pass
        
        # Fallback recommendations
        recommendations = []
        if critical:
            recommendations.append(f"Review {len(critical)} critical contingencies that could cause violations")
        if vulnerable:
            recommendations.append(f"Consider upgrading line {vulnerable[0]['line_id']} which is vulnerable to {vulnerable[0]['count']} contingencies")
        
        return recommendations if recommendations else ["No specific recommendations"]
    
    def get_n1_security_status(self, net) -> Dict:
        """Quick N-1 security check (fewer contingencies for speed)"""
        # Run limited analysis
        results = self.run_n1_analysis(net, ["line"])
        
        return {
            "n1_secure": results["summary"]["n1_secure"],
            "critical_count": results["summary"]["critical_contingencies"],
            "worst_contingency": results["critical_contingencies"][0] if results["critical_contingencies"] else None,
            "status": "SECURE" if results["summary"]["n1_secure"] else "VULNERABLE"
        }
    
    def recommend_preventive_actions(self, critical_contingencies: List[Dict], net) -> List[Dict]:
        """Recommend actions to mitigate critical contingencies"""
        actions = []
        
        for cont in critical_contingencies[:3]:  # Top 3
            # Analyze the contingency
            violations = cont.get("violations", {})
            
            # For thermal violations, recommend generation redispatch
            if violations.get("thermal"):
                worst_thermal = max(violations["thermal"], key=lambda x: x.get("loading", 0))
                actions.append({
                    "contingency": cont["element_name"],
                    "action": "preventive_redispatch",
                    "description": f"Pre-position generation to prevent line {worst_thermal['line']} overload",
                    "priority": 1
                })
            
            # For voltage violations, recommend reactive support
            if violations.get("voltage_low"):
                worst_voltage = min(violations["voltage_low"], key=lambda x: x.get("voltage", 1.0))
                actions.append({
                    "contingency": cont["element_name"],
                    "action": "add_reactive_support",
                    "description": f"Add capacitor bank at bus {worst_voltage['bus']} for voltage support",
                    "priority": 2
                })
        
        return actions

"""
Intelligent Alarm Prioritization System
Ranks violations by severity and impact
Reduces operator overload with smart alerting
"""

from typing import Dict, List, Any
from enum import Enum
from datetime import datetime
import json


class AlarmSeverity(Enum):
    """Alarm severity levels"""
    CRITICAL = 5    # Immediate action required
    HIGH = 4        # Action needed soon
    MEDIUM = 3      # Monitor and plan action
    LOW = 2         # Informational
    INFO = 1        # FYI


class AlarmPrioritizer:
    """
    Prioritizes violations based on severity and impact
    Provides intelligent alerts to operators
    """
    
    def __init__(self):
        self.active_alarms = []
        self.alarm_history = []
        self.suppressed_alarms = set()
    
    def generate_alarms(self, violations: Dict, predictions: Dict = None) -> List[Dict]:
        """
        Generate prioritized alarms from violations and predictions
        
        Args:
            violations: Current violations from power flow
            predictions: Predicted future violations (optional)
        
        Returns:
            List of alarms sorted by priority
        """
        alarms = []
        
        # Process voltage violations
        for bus in violations.get("voltage", []):
            alarm = self._create_voltage_alarm(bus, violations)
            alarms.append(alarm)
        
        # Process thermal violations
        for line in violations.get("thermal", []):
            alarm = self._create_thermal_alarm(line, violations)
            alarms.append(alarm)
        
        # Add predictive alarms if available
        if predictions and predictions.get("status") == "ok":
            pred_alarms = self._create_predictive_alarms(predictions)
            alarms.extend(pred_alarms)
        
        # Sort by severity (highest first)
        alarms = sorted(alarms, key=lambda x: x["priority_score"], reverse=True)
        
        # Remove suppressed alarms
        alarms = [a for a in alarms if a["alarm_id"] not in self.suppressed_alarms]
        
        # Update active alarms
        self.active_alarms = alarms[:10]  # Keep top 10
        
        return alarms
    
    def _create_voltage_alarm(self, bus: int, violations: Dict) -> Dict:
        """Create alarm for voltage violation"""
        return {
            "alarm_id": f"VOLTAGE_{bus}",
            "type": "voltage_violation",
            "target": bus,
            "timestamp": datetime.now().isoformat(),
            "severity": AlarmSeverity.HIGH.name,
            "priority_score": 85,
            "message": f"Bus {bus}: Voltage out of nominal range (0.95-1.05 pu)",
            "recommended_actions": [
                "Adjust voltage control devices (capacitors, regulators)",
                "Reduce load or increase generation in area",
                "Check for network contingency"
            ],
            "impact": "Affects equipment lifespan and stability",
            "duration_seconds": 0
        }
    
    def _create_thermal_alarm(self, line: int, violations: Dict) -> Dict:
        """Create alarm for thermal violation"""
        return {
            "alarm_id": f"THERMAL_{line}",
            "type": "thermal_violation",
            "target": line,
            "timestamp": datetime.now().isoformat(),
            "severity": AlarmSeverity.CRITICAL.name,
            "priority_score": 95,
            "message": f"Line {line}: Loading exceeds 100% (thermal limit reached)",
            "recommended_actions": [
                "Reduce power flow on line (redirect or reduce demand)",
                "Switch alternative line if available",
                "Shed non-critical load in downstream area",
                "Reduce generation in upstream area"
            ],
            "impact": "Risk of line damage and cascading failures",
            "duration_seconds": 0
        }
    
    def _create_predictive_alarms(self, predictions: Dict) -> List[Dict]:
        """Create alarms based on predicted future violations"""
        alarms = []
        
        predicted_v = predictions["predicted_violations"].get("voltage", {})
        predicted_t = predictions["predicted_violations"].get("thermal", {})
        
        # Predictive voltage alarm
        if predicted_v and predicted_v.get("predicted_count", 0) > 0:
            if predicted_v.get("trend") == "increasing":
                alarms.append({
                    "alarm_id": "PRED_VOLTAGE_TREND",
                    "type": "predictive_voltage",
                    "timestamp": datetime.now().isoformat(),
                    "severity": AlarmSeverity.MEDIUM.name,
                    "priority_score": 65,
                    "message": f"Voltage violations trending upward - {predicted_v['predicted_count']} expected in {predictions['steps_ahead']} steps",
                    "confidence": predicted_v.get("confidence", 0.5),
                    "recommended_actions": [
                        "Implement preventive voltage control",
                        "Schedule maintenance or load transfer"
                    ],
                    "impact": "Proactive action can prevent cascading violations",
                    "duration_seconds": 0
                })
        
        # Predictive thermal alarm
        if predicted_t and predicted_t.get("predicted_count", 0) > 0:
            if predicted_t.get("trend") == "increasing":
                alarms.append({
                    "alarm_id": "PRED_THERMAL_TREND",
                    "type": "predictive_thermal",
                    "timestamp": datetime.now().isoformat(),
                    "severity": AlarmSeverity.HIGH.name,
                    "priority_score": 75,
                    "message": f"Thermal violations trending upward - {predicted_t['predicted_count']} expected in {predictions['steps_ahead']} steps",
                    "confidence": predicted_t.get("confidence", 0.5),
                    "recommended_actions": [
                        "Prepare line switching or load shedding plan",
                        "Increase generation capacity in area",
                        "Coordinate with adjacent operators"
                    ],
                    "impact": "Early action can prevent emergency conditions",
                    "duration_seconds": 0
                })
        
        return alarms
    
    def suppress_alarm(self, alarm_id: str, duration_minutes: int = 30):
        """
        Suppress recurring alarm temporarily
        (e.g., when acknowledged by operator)
        """
        self.suppressed_alarms.add(alarm_id)
        print(f"Suppressed alarm {alarm_id} for {duration_minutes} minutes")
    
    def acknowledge_alarm(self, alarm_id: str) -> bool:
        """Acknowledge alarm as reviewed by operator"""
        for alarm in self.active_alarms:
            if alarm["alarm_id"] == alarm_id:
                alarm["acknowledged"] = True
                alarm["acknowledged_at"] = datetime.now().isoformat()
                self.alarm_history.append(alarm)
                return True
        return False
    
    def get_active_alarms_summary(self) -> Dict[str, Any]:
        """Get summary of current active alarms"""
        if not self.active_alarms:
            return {
                "total_alarms": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "alarms": []
            }
        
        critical = sum(1 for a in self.active_alarms if a["severity"] == AlarmSeverity.CRITICAL.name)
        high = sum(1 for a in self.active_alarms if a["severity"] == AlarmSeverity.HIGH.name)
        medium = sum(1 for a in self.active_alarms if a["severity"] == AlarmSeverity.MEDIUM.name)
        
        return {
            "total_alarms": len(self.active_alarms),
            "critical": critical,
            "high": high,
            "medium": medium,
            "alarms": [
                {
                    "id": a["alarm_id"],
                    "type": a["type"],
                    "severity": a["severity"],
                    "message": a["message"],
                    "target": a.get("target"),
                    "priority": a["priority_score"]
                }
                for a in self.active_alarms[:5]  # Top 5
            ]
        }
    
    def get_recommended_actions(self) -> List[str]:
        """
        Get all recommended actions from top alarms
        Prioritized by severity
        """
        actions = []
        seen = set()
        
        for alarm in self.active_alarms[:5]:  # Top 5 alarms
            for action in alarm.get("recommended_actions", []):
                if action not in seen:
                    actions.append({
                        "action": action,
                        "from_alarm": alarm["alarm_id"],
                        "priority": alarm["priority_score"]
                    })
                    seen.add(action)
        
        return sorted(actions, key=lambda x: x["priority"], reverse=True)
    
    def clear_resolved_alarms(self, resolved_ids: List[str]):
        """Remove resolved alarms from active list"""
        self.active_alarms = [a for a in self.active_alarms if a["alarm_id"] not in resolved_ids]
    
    def get_alarm_statistics(self) -> Dict[str, Any]:
        """Get historical alarm statistics"""
        total = len(self.alarm_history)
        
        if total == 0:
            return {
                "total_alarms_ever": 0,
                "by_type": {},
                "by_severity": {}
            }
        
        by_type = {}
        by_severity = {}
        
        for alarm in self.alarm_history:
            atype = alarm["type"]
            by_type[atype] = by_type.get(atype, 0) + 1
            
            severity = alarm["severity"]
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_alarms_ever": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "average_response_time": self._calculate_avg_response_time()
        }
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average time from alarm to acknowledgement"""
        times = []
        
        for alarm in self.alarm_history:
            if alarm.get("acknowledged_at"):
                try:
                    created = datetime.fromisoformat(alarm["timestamp"])
                    acked = datetime.fromisoformat(alarm["acknowledged_at"])
                    times.append((acked - created).total_seconds())
                except:
                    pass
        
        return sum(times) / len(times) if times else 0
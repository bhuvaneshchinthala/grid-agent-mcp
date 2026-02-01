"""
Grid Agent - Multi-Agent System for Power Grid Control
All imports are optional with graceful fallbacks
"""

# Core base agent
try:
    from grid_agent.agents.base_agent import BaseAgent
except ImportError:
    BaseAgent = None

# Voltage control
try:
    from grid_agent.agents.voltage_control_agent import VoltageControlAgent
except ImportError:
    VoltageControlAgent = None

# Thermal control
try:
    from grid_agent.agents.thermal_control_agent import ThermalControlAgent
except ImportError:
    ThermalControlAgent = None

# Violation predictor
try:
    from grid_agent.agents.violation_predictor import ViolationPredictor
except ImportError:
    ViolationPredictor = None

# Alarm prioritizer
try:
    from grid_agent.agents.alarm_prioritizer import AlarmPrioritizer
except ImportError:
    AlarmPrioritizer = None

# Digital twin
try:
    from grid_agent.agents.digital_twin import DigitalTwin
except ImportError:
    DigitalTwin = None

# New specialized agents
try:
    from grid_agent.agents.contingency_agent import ContingencyAgent
except ImportError:
    ContingencyAgent = None

try:
    from grid_agent.agents.restoration_agent import RestorationAgent
except ImportError:
    RestorationAgent = None

__all__ = [
    "BaseAgent",
    "VoltageControlAgent",
    "ThermalControlAgent",
    "ViolationPredictor",
    "AlarmPrioritizer",
    "DigitalTwin",
    "ContingencyAgent",
    "RestorationAgent",
]

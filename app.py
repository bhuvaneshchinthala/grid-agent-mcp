"""
Enhanced VoltAI Grid Control UI
Includes predictive violations, smart alarms, digital twin scenarios, and more
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandapower.networks as pn
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Import all enhanced components
from grid_agent.core.power_flow_solver import PowerFlowSolver
from grid_agent.core.orchestrator import Orchestrator
from grid_agent.utils.graph_visualizer import plot_network
from violation_predictor import ViolationPredictor
from alarm_prioritizer import AlarmPrioritizer
from safety_constraint_validator import SafetyConstraintValidator
from digital_twin import DigitalTwin
from voltage_control_agent import VoltageControlAgent
from thermal_control_agent import ThermalControlAgent

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="VoltAI Advanced – LLM-Powered Smart Grid Control",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ VoltAI Advanced – AI-Powered Smart Grid Control")

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if "solver" not in st.session_state:
    st.session_state.solver = PowerFlowSolver()
    st.session_state.orchestrator = Orchestrator()
    st.session_state.predictor = ViolationPredictor()
    st.session_state.alarm_prioritizer = AlarmPrioritizer()
    st.session_state.safety_validator = SafetyConstraintValidator()
    st.session_state.digital_twin = None
    st.session_state.voltage_agent = VoltageControlAgent()
    st.session_state.thermal_agent = ThermalControlAgent()

# ============================================================================
# SIDEBAR - MAIN CONTROLS
# ============================================================================
st.sidebar.header("🛠 Control Panel")

tab_mode = st.sidebar.radio(
    "Select Tab",
    ["🏠 Home", "⚡ Real-Time Monitoring", "🔮 Predictive Analytics",
     "🚨 Alarm Management", "🤖 Specialized Agents", 
     "📊 Digital Twin", "💾 System Memory"]
)

# Network selection
use_distribution_demo = st.sidebar.checkbox("Use 33-Bus Distribution", value=True)
simulate_outage = st.sidebar.checkbox("Simulate Line Outage", value=True)

run_btn = st.sidebar.button("▶ Run System Analysis", key="run_btn")
reset_btn = st.sidebar.button("🔄 Reset & Clear Memory")

if reset_btn:
    st.session_state.predictor.historical_data = []
    st.session_state.alarm_prioritizer.active_alarms = []
    st.session_state.orchestrator.planner.successful_actions = []
    st.session_state.orchestrator.planner.failed_actions = []
    st.success("✓ System reset successfully")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_network():
    """Load and configure network"""
    if use_distribution_demo:
        net = pn.case33bw()
        if simulate_outage and len(net.line) > 11:
            net.line.at[10, "in_service"] = False
            net.line.at[11, "in_service"] = False
        return net, "33-Bus Distribution Network (with simulated outage)" if simulate_outage else "33-Bus Distribution"
    else:
        return pn.case30(), "30-Bus Transmission Network"

# ============================================================================
# HOME TAB
# ============================================================================
if tab_mode == "🏠 Home":
    st.markdown("""
    ## Welcome to VoltAI Advanced
    
    **Next-Generation AI-Powered Power Grid Control System**
    
    ### Key Features:
    
    🔮 **Predictive Violation Detection** - Forecasts future violations before they occur
    
    🚨 **Intelligent Alarms** - Prioritized alerts based on severity and impact
    
    🤖 **Specialized Agents** - Dedicated voltage, thermal, and restoration control
    
    🛡️ **Safety-Constrained Execution** - Validates all AI actions against physical limits
    
    📊 **Digital Twin** - Simulate scenarios without affecting real operations
    
    💡 **Explainability** - Understand why each control decision was made
    
    💾 **Learning System** - Agents learn from successful actions for continuous improvement
    
    ---
    
    ### Quick Start:
    1. Configure network in sidebar
    2. Click **"▶ Run System Analysis"**
    3. Navigate through tabs to explore analysis results
    4. Try Digital Twin for scenario testing
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("System Status", "🟢 Ready", "0 violations")
    with col2:
        st.metric("Agents", "4 Active", "Monitoring grid")
    with col3:
        st.metric("Prediction", "Enabled", "24-hour forecast")

# ============================================================================
# REAL-TIME MONITORING TAB
# ============================================================================
elif tab_mode == "⚡ Real-Time Monitoring" and run_btn:
    st.header("Real-Time Grid Monitoring")
    
    # Load network
    net, net_name = load_network()
    st.info(f"📍 Network: {net_name}")
    
    # Run power flow
    net = st.session_state.solver.run(net)
    violations = st.session_state.solver.detect_violations(net)
    st.session_state.digital_twin = DigitalTwin(net)
    
    # Dashboard metrics
    st.subheader("📊 Network Health Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    health = violations["summary"]["network_health_percent"]
    severity = violations["summary"]["severity_score"]
    total_violations = violations["summary"]["total_violations"]
    voltage_violations = len(violations.get("voltage", []))
    thermal_violations = len(violations.get("thermal", []))
    
    with col1:
        st.metric("Network Health", f"{health:.1f}%", 
                 "🟢 Excellent" if health > 80 else "🟡 Fair" if health > 50 else "🔴 Poor")
    with col2:
        st.metric("Severity Score", f"{severity:.1f}", 
                 delta=f"{severity:.1f}" if severity > 0 else "No issues")
    with col3:
        st.metric("Total Violations", total_violations, 
                 delta=f"{total_violations}" if total_violations > 0 else "Clear")
    with col4:
        st.metric("System Status", "Analyzing", "All systems operational")
    
    # Violations breakdown
    st.subheader("⚠️ Current Violations")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"🔴 **Voltage**: {voltage_violations} buses")
    with col2:
        st.error(f"🔴 **Thermal**: {thermal_violations} lines")
    with col3:
        st.warning(f"⚠️ **Severity**: {severity:.1f}")
    
    # Network visualization
    st.subheader("🗺️ Network Topology")
    fig = plot_network(net, violations)
    st.pyplot(fig)
    plt.close(fig)
    
    # Detailed violations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Voltage Violations")
        if violations["voltage"]:
            st.json({"buses": violations["voltage"]})
        else:
            st.success("✓ No voltage violations")
    
    with col2:
        st.subheader("Thermal Violations")
        if violations["thermal"]:
            st.json({"lines": violations["thermal"]})
        else:
            st.success("✓ No thermal violations")

# ============================================================================
# PREDICTIVE ANALYTICS TAB
# ============================================================================
elif tab_mode == "🔮 Predictive Analytics" and run_btn:
    st.header("Predictive Violation Detection")
    st.markdown("AI-powered forecasting of future grid violations")
    
    net, net_name = load_network()
    net = st.session_state.solver.run(net)
    violations = st.session_state.solver.detect_violations(net)
    
    # Add observation to history
    st.session_state.predictor.add_observation(net, violations)
    
    # Make predictions
    steps_ahead = st.slider("Predict steps ahead", 1, 5, 2)
    predictions = st.session_state.predictor.predict_violations(net, steps_ahead)
    
    # Display predictions
    st.subheader(f"📈 Predictions ({steps_ahead} steps ahead)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction Confidence", 
                 f"{predictions['confidence']*100:.1f}%",
                 "High" if predictions['confidence'] > 0.7 else "Medium")
    with col2:
        pred_v = predictions['predicted_violations'].get('voltage', {})
        st.metric("Predicted Voltage Violations", 
                 pred_v.get('predicted_count', 0),
                 f"Trend: {pred_v.get('trend', 'unknown')}")
    with col3:
        pred_t = predictions['predicted_violations'].get('thermal', {})
        st.metric("Predicted Thermal Violations",
                 pred_t.get('predicted_count', 0),
                 f"Trend: {pred_t.get('trend', 'unknown')}")
    
    # Warnings
    if predictions['warnings']:
        st.subheader("⚠️ Predictive Warnings")
        for warning in predictions['warnings']:
            st.warning(warning)
    
    # Detailed predictions
    st.subheader("📊 Detailed Predictions")
    st.json(predictions['predicted_violations'])
    
    # Confidence visualization
    st.subheader("📉 Prediction Confidence Over Time")
    confidence_data = pd.DataFrame({
        'Data Points': range(len(st.session_state.predictor.historical_data)),
        'Confidence': [min(1.0, (i+1)/st.session_state.predictor.history_window) 
                      for i in range(len(st.session_state.predictor.historical_data))]
    })
    
    fig = px.line(confidence_data, x='Data Points', y='Confidence',
                 title="Prediction Confidence Growth",
                 markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ALARM MANAGEMENT TAB
# ============================================================================
elif tab_mode == "🚨 Alarm Management" and run_btn:
    st.header("Intelligent Alarm Management System")
    
    net, net_name = load_network()
    net = st.session_state.solver.run(net)
    violations = st.session_state.solver.detect_violations(net)
    
    # Get predictions for alarm context
    st.session_state.predictor.add_observation(net, violations)
    predictions = st.session_state.predictor.predict_violations(net)
    
    # Generate alarms
    alarms = st.session_state.alarm_prioritizer.generate_alarms(violations, predictions)
    
    # Alarm summary
    st.subheader("🚨 Active Alarms")
    summary = st.session_state.alarm_prioritizer.get_active_alarms_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Alarms", summary['total_alarms'],
                 delta=f"{summary['total_alarms']} active" if summary['total_alarms'] > 0 else "Clear")
    with col2:
        st.metric("🔴 CRITICAL", summary['critical'],
                 delta="Requires immediate action" if summary['critical'] > 0 else "None")
    with col3:
        st.metric("🟠 HIGH", summary['high'])
    with col4:
        st.metric("🟡 MEDIUM", summary['medium'])
    
    # Top alarms
    if alarms:
        st.subheader("📋 Top Priority Alarms")
        for alarm in alarms[:5]:
            with st.expander(f"**{alarm['severity']}**: {alarm['message']}", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Type**: {alarm['type']}")
                    st.write(f"**Target**: {alarm.get('target', 'N/A')}")
                    st.write(f"**Impact**: {alarm.get('impact', 'Unknown')}")
                    st.write(f"**Recommended Actions**:")
                    for action in alarm.get('recommended_actions', []):
                        st.write(f"  • {action}")
                with col2:
                    if st.button(f"Acknowledge", key=f"ack_{alarm['alarm_id']}"):
                        st.session_state.alarm_prioritizer.acknowledge_alarm(alarm['alarm_id'])
                        st.success("✓ Alarm acknowledged")
    else:
        st.success("✓ No active alarms - System operating normally")
    
    # Recommended actions
    st.subheader("💡 AI-Recommended Actions")
    actions = st.session_state.alarm_prioritizer.get_recommended_actions()
    if actions:
        for i, action_info in enumerate(actions[:5], 1):
            st.info(f"{i}. {action_info['action']} (Priority: {action_info['priority']}/100)")
    
    # Statistics
    st.subheader("📊 Alarm Statistics")
    stats = st.session_state.alarm_prioritizer.get_alarm_statistics()
    col1, col2 = st.columns(2)
    with col1:
        st.json(stats)

# ============================================================================
# SPECIALIZED AGENTS TAB
# ============================================================================
elif tab_mode == "🤖 Specialized Agents" and run_btn:
    st.header("Specialized Multi-Agent Control System")
    
    net, net_name = load_network()
    net = st.session_state.solver.run(net)
    violations = st.session_state.solver.detect_violations(net)
    
    st.markdown("""
    Three specialized agents work in coordination:
    - **Voltage Control Agent**: Manages voltage regulation and reactive power
    - **Thermal Control Agent**: Manages line loading and power flow
    - **Restoration Agent**: Manages network recovery and reconfiguration
    """)
    
    agent_tab1, agent_tab2, agent_tab3 = st.tabs(
        ["⚡ Voltage Control", "🔥 Thermal Control", "🔧 Restoration"]
    )
    
    # Network context
    network_context = {
        "total_buses": len(net.bus),
        "voltage_devices": list(range(min(5, len(net.bus)))),
        "critical_buses": list(violations.get("voltage", [])[:3]),
        "alternative_lines": list(range(min(3, len(net.line)))),
        "flexible_generators": len(net.gen),
        "sheddable_load_mw": net.load.p_mw.sum() * 0.3,  # 30% of load
        "reactive_power_mvar": len(net.gen) * 50  # Estimate
    }
    
    # Voltage Control Agent
    with agent_tab1:
        st.subheader("⚡ Voltage Control Agent")
        
        if len(violations.get("voltage", [])) > 0:
            st.warning(f"🔴 {len(violations['voltage'])} voltage violations detected")
            
            with st.spinner("Voltage control agent planning..."):
                v_actions = st.session_state.voltage_agent.plan_voltage_control(
                    violations, network_context
                )
            
            st.subheader("Proposed Voltage Control Actions")
            if v_actions:
                for i, action in enumerate(v_actions, 1):
                    st.write(f"**Action {i}**: {action}")
            
            st.subheader("Agent Reasoning")
            st.code(st.session_state.voltage_agent.last_reasoning)
            
            # Memory stats
            col1, col2, col3 = st.columns(3)
            memory = st.session_state.voltage_agent.get_memory_summary()
            with col1:
                st.metric("Successful Actions", memory['total_successful_actions'])
            with col2:
                st.metric("Failed Actions", memory['total_failed_actions'])
            with col3:
                st.metric("Success Rate", f"{memory['success_rate']*100:.1f}%")
        else:
            st.success("✓ No voltage violations - Agent on standby")
    
    # Thermal Control Agent
    with agent_tab2:
        st.subheader("🔥 Thermal Control Agent")
        
        if len(violations.get("thermal", [])) > 0:
            st.error(f"🔴 {len(violations['thermal'])} thermal violations detected")
            
            with st.spinner("Thermal control agent planning..."):
                t_actions = st.session_state.thermal_agent.plan_thermal_control(
                    violations, network_context
                )
            
            st.subheader("Proposed Thermal Control Actions")
            if t_actions:
                for i, action in enumerate(t_actions, 1):
                    st.write(f"**Action {i}**: {action}")
            
            st.subheader("Agent Reasoning")
            st.code(st.session_state.thermal_agent.last_reasoning)
            
            # Thermal status
            thermal_status = st.session_state.thermal_agent.get_thermal_status_summary(net)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Normal Lines", thermal_status.get('normal_lines', 0))
            with col2:
                st.metric("Warning Lines", thermal_status.get('warning_lines', 0))
            with col3:
                st.metric("Max Loading", f"{thermal_status.get('max_loading', 0):.1f}%")
        else:
            st.success("✓ All lines within thermal limits - Agent on standby")
    
    # Restoration Agent
    with agent_tab3:
        st.subheader("🔧 Restoration & Recovery Agent")
        st.info("Handles network reconfiguration and service restoration after outages")
        st.write("Status: Monitoring system for restoration opportunities")

# ============================================================================
# DIGITAL TWIN TAB
# ============================================================================
elif tab_mode == "📊 Digital Twin" and run_btn:
    st.header("Digital Twin – Scenario Simulation & Analysis")
    
    net, net_name = load_network()
    net = st.session_state.solver.run(net)
    violations = st.session_state.solver.detect_violations(net)
    
    if st.session_state.digital_twin is None:
        st.session_state.digital_twin = DigitalTwin(net)
    
    st.markdown("""
    The Digital Twin creates a simulation replica of your grid, allowing you to:
    - Test control strategies safely without affecting real operations
    - Simulate outages and contingencies
    - Compare alternative solutions
    - Train AI agents on realistic scenarios
    """)
    
    # Scenario creation
    st.subheader("🎯 Create Simulation Scenario")
    
    col1, col2 = st.columns(2)
    with col1:
        scenario_name = st.text_input("Scenario Name", "Custom_Scenario_1")
        outage_lines = st.multiselect("Simulate Line Outages", 
                                      range(min(10, len(net.line))),
                                      default=[])
    with col2:
        load_increase_pct = st.number_input("Load Increase (%)", 0, 100, 20)
        gen_reduction_mw = st.number_input("Generation Reduction (MW)", 0, 100, 10)
    
    if st.button("Create & Simulate Scenario"):
        modifications = {
            "outages": outage_lines,
            "load_increase": {0: load_increase_pct},
            "generation_change": {0: -gen_reduction_mw}
        }
        
        success, msg, scenario_net = st.session_state.digital_twin.create_scenario(
            scenario_name, modifications
        )
        
        if success:
            st.success(f"✓ {msg}")
            
            # Simulate
            result = st.session_state.digital_twin.simulate_scenario(
                scenario_name, scenario_net
            )
            
            st.subheader("📊 Scenario Results")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Control Actions**")
                st.json(result["before_control"])
            with col2:
                st.write("**After Control Actions**")
                st.json(result["after_control"])
            
            if result.get("improvement"):
                st.subheader("📈 Improvement")
                st.json(result["improvement"])
        else:
            st.error(f"❌ {msg}")
    
    # Strategy comparison
    st.subheader("⚖️ Strategy Comparison")
    st.write("Compare two different control strategies on the same scenario")
    
    if st.button("Run Strategy Comparison"):
        strategy_a = [
            {"action_type": "reduce_generation", "target": 0, "value": 0.8},
            {"action_type": "curtail_load", "target": 1, "value": 0.9}
        ]
        strategy_b = [
            {"action_type": "curtail_load", "target": 0, "value": 0.85}
        ]
        
        comparison = st.session_state.digital_twin.compare_strategies(
            "comparison_1", net, strategy_a, strategy_b
        )
        
        st.json(comparison)

# ============================================================================
# SYSTEM MEMORY TAB
# ============================================================================
elif tab_mode == "💾 System Memory" and run_btn:
    st.header("Agent Memory & Learning System")
    
    st.markdown("""
    The system learns from past control actions, improving decision quality over time.
    """)
    
    # Agent memory overview
    st.subheader("🧠 Agent Memory Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    planner_memory = st.session_state.orchestrator.planner.get_memory_summary()
    voltage_memory = st.session_state.voltage_agent.get_memory_summary()
    thermal_memory = st.session_state.thermal_agent.get_memory_summary()
    
    with col1:
        st.metric("Planner Agent",
                 f"{planner_memory['total_successful_actions']}",
                 f"Success: {planner_memory['success_rate']*100:.1f}%")
    with col2:
        st.metric("Voltage Agent",
                 f"{voltage_memory['total_successful_actions']}",
                 f"Success: {voltage_memory['success_rate']*100:.1f}%")
    with col3:
        st.metric("Thermal Agent",
                 f"{thermal_memory['total_successful_actions']}",
                 f"Success: {thermal_memory['success_rate']*100:.1f}%")
    with col4:
        st.metric("System Learning", 
                 "Active",
                 "Continuous improvement")
    
    # Detailed memory views
    tab_mem1, tab_mem2, tab_mem3 = st.tabs(["Planner", "Voltage", "Thermal"])
    
    with tab_mem1:
        st.subheader("Planner Agent Memory")
        st.json(planner_memory)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Successful Actions (Planner)"):
                st.json(st.session_state.orchestrator.planner.successful_actions[-5:])
        with col2:
            if st.button("Clear Planner Memory"):
                st.session_state.orchestrator.planner.clear_memory()
                st.success("Planner memory cleared")
    
    with tab_mem2:
        st.subheader("Voltage Agent Memory")
        st.json(voltage_memory)
    
    with tab_mem3:
        st.subheader("Thermal Agent Memory")
        st.json(thermal_memory)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("VoltAI Advanced v2.0 • Next-Gen AI Grid Control • " +
          "Pandapower + Streamlit + Mistral + Advanced ML")
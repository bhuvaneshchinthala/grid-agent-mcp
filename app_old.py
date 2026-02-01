# -------------------------------
# IMPORT SAFETY (IMPORTANT)
# -------------------------------
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandapower.networks as pn
import json
import matplotlib.pyplot as plt

from grid_agent.core.power_flow_solver import PowerFlowSolver
from grid_agent.core.orchestrator import Orchestrator
from grid_agent.utils.graph_visualizer import plot_network

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="VoltAI – Grid Control",
    layout="wide"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("⚡ VoltAI – LLM-Powered Power Grid Control")
st.markdown(
    """
This demo shows an **LLM-driven multi-agent system** for power-grid monitoring,
violation detection, and autonomous control using **Mistral (Ollama)**.

**Features**
- Power-flow analysis (pandapower)
- Violation detection
- LLM-based planning
- AutoMode & Interactive Mode
"""
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("🛠 Control Panel")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["AutoMode", "Interactive Mode"]
)

use_distribution_demo = st.sidebar.checkbox(
    "Use Distribution Outage Demo (33-bus)",
    value=True
)

run_btn = st.sidebar.button("▶ Run System")

st.sidebar.markdown("---")
st.sidebar.info("Run the system to analyze and control the grid")

# -------------------------------------------------
# BACKEND OBJECTS
# -------------------------------------------------
solver = PowerFlowSolver()
orchestrator = Orchestrator()

# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------
if run_btn:
    st.success("Power-flow analysis started")

    # ---------------------------------------------
    # LOAD NETWORK (SAFE NETWORKS ONLY)
    # ---------------------------------------------
    if use_distribution_demo:
        net = pn.case33bw()

        # Simulate outage
        if len(net.line) > 11:
            net.line.at[10, "in_service"] = False
            net.line.at[11, "in_service"] = False

        st.warning(
            "Simulated outage: lines 10 & 11 opened (33-bus distribution network)"
        )
    else:
        net = pn.case30()

    # ---------------------------------------------
    # RUN POWER FLOW
    # ---------------------------------------------
    net = solver.run(net)
    violations = solver.detect_violations(net)

    # ---------------------------------------------
    # DASHBOARD
    # ---------------------------------------------
    st.subheader("📊 Network Health Dashboard")

    voltage_count = len(violations.get("voltage", []))
    thermal_count = len(violations.get("thermal", []))
    disconnected_count = len(violations.get("disconnected", []))

    severity = voltage_count * 2 + thermal_count * 3 + disconnected_count * 4
    health = max(0, 100 - severity * 3)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Network Health (%)", f"{health:.1f}")

    with col2:
        st.metric("Severity Score", f"{severity:.2f}")

    with col3:
        st.metric(
            "Total Violations",
            voltage_count + thermal_count + disconnected_count
        )

    st.progress(min(100, int(severity)))

    # ---------------------------------------------
    # NETWORK GRAPH (FIXED – NO ZERO BUG)
    # ---------------------------------------------
    st.subheader("🗺️ Network Topology")

    fig = plot_network(net, violations)
    _ = st.pyplot(fig)   # DISCARD RETURN VALUE (IMPORTANT)
    plt.close(fig)

    # ---------------------------------------------
    # INITIAL VIOLATIONS
    # ---------------------------------------------
    st.subheader("⚠️ Initial Violations")
    st.json(violations)

    # =================================================
    # AUTOMODE
    # =================================================
    if mode == "AutoMode":
        st.subheader("🤖 AutoMode Execution")

        with st.spinner("Agents are resolving violations..."):
            final_violations = orchestrator.run(net)

        st.subheader("🧠 Agent Reasoning (Last Iteration)")
        st.code(orchestrator.planner.last_reasoning)

        st.subheader("✅ Final Violations")
        st.json(final_violations)

    # =================================================
    # INTERACTIVE MODE
    # =================================================
    else:
        st.subheader("🧑‍💼 Interactive Mode")

        plan = orchestrator.planner.plan(violations)

        st.subheader("🧠 Agent Proposed Plan")
        st.json(json.loads(json.dumps(plan)))

        st.subheader("🧠 Agent Reasoning")
        st.code(orchestrator.planner.last_reasoning)

        approve = st.button("✔ Approve & Execute Plan")

        if approve:
            st.success("Executing approved plan")

            net = orchestrator.executor.execute(net, plan)
            net = solver.run(net)
            new_violations = solver.detect_violations(net)

            st.subheader("📉 Violations After Execution")
            st.json(new_violations)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption(
    "VoltAI Demo • LLM-powered multi-agent grid control • "
    "Pandapower + Streamlit + Mistral (Ollama)"
)

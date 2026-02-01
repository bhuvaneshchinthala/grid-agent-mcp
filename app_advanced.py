"""
⚡ Faraday Power Grid Dashboard - Advanced Edition
AI-Powered Electrical Network Violation Analysis & Resolution
With Workflow Results, Interactive Action Plan, and Battery Storage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandapower.networks as pn
import pandapower as pp
import pandas as pd
import numpy as np
import json
import time
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

# Import grid components
from grid_agent.core.power_flow_solver import PowerFlowSolver
from grid_agent.core.orchestrator import Orchestrator
from grid_agent.agents.thermal_control_agent import ThermalControlAgent
from grid_agent.agents.voltage_control_agent import VoltageControlAgent

try:
    from grid_agent.agents.battery_storage import BatteryStorageAgent
    BATTERY_AVAILABLE = True
except:
    BATTERY_AVAILABLE = False

try:
    from grid_agent.agents.contingency_agent import ContingencyAgent
    CONTINGENCY_AVAILABLE = True
except:
    CONTINGENCY_AVAILABLE = False

try:
    from grid_agent.agents.restoration_agent import RestorationAgent
    RESTORATION_AVAILABLE = True
except:
    RESTORATION_AVAILABLE = False


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Faraday Power Grid Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE - Extended
# ============================================================================
if "solver" not in st.session_state:
    st.session_state.solver = PowerFlowSolver()
    st.session_state.orchestrator = Orchestrator()
    st.session_state.voltage_agent = VoltageControlAgent()
    st.session_state.thermal_agent = ThermalControlAgent()
    if BATTERY_AVAILABLE:
        st.session_state.battery_agent = BatteryStorageAgent()
    st.session_state.activity_log = []
    st.session_state.net = None
    st.session_state.net_original = None
    st.session_state.net_initial = None
    st.session_state.net_final = None
    st.session_state.violations = None
    st.session_state.violations_initial = None
    st.session_state.violations_final = None
    st.session_state.mode = "auto"
    st.session_state.start_time = None
    st.session_state.network_loaded = False
    st.session_state.actions_in_plan = 0
    st.session_state.actions_executed = 0
    st.session_state.theme = "light"
    st.session_state.workflow_complete = False
    st.session_state.action_plan = []
    st.session_state.executed_actions = []
    st.session_state.show_debug = False
    st.session_state.network_name = ""
    st.session_state.drop_line = None


# ============================================================================
# THEME STYLES
# ============================================================================
def get_theme_css(theme):
    if theme == "dark":
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            * { font-family: 'Inter', sans-serif; }
            .stApp { background: #0f1419 !important; }
            .main-header {
                background: linear-gradient(135deg, #1a365d 0%, #2563eb 100%);
                color: white; padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;
            }
            .main-header h1 { margin: 0; font-size: 1.5rem; font-weight: 600; }
            .main-header p { margin: 5px 0 0 0; font-size: 0.85rem; opacity: 0.8; }
            .section-card {
                background: #1a1f2e; border-radius: 10px; padding: 20px;
                margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); border: 1px solid #2d3748;
            }
            .section-title { font-size: 1rem; font-weight: 600; color: #60a5fa; margin-bottom: 15px; }
            .metric-box { text-align: center; padding: 10px; }
            .metric-label { font-size: 0.75rem; color: #9ca3af; margin-bottom: 5px; }
            .metric-value { font-size: 1.8rem; font-weight: 600; color: #f1f5f9; }
            .metric-delta-positive { color: #22c55e; font-size: 0.8rem; }
            .metric-delta-negative { color: #ef4444; font-size: 0.8rem; }
            .status-healthy { color: #22c55e; font-weight: 500; }
            .status-warning { color: #eab308; font-weight: 500; }
            .status-critical { color: #ef4444; font-weight: 500; }
            .success-banner {
                background: #064e3b; border-left: 4px solid #10b981;
                padding: 12px 16px; border-radius: 0 8px 8px 0; color: #a7f3d0;
            }
            .action-card {
                background: #1e293b; border: 1px solid #334155; border-radius: 8px;
                padding: 15px; margin-bottom: 10px;
            }
            h1, h2, h3, h4, h5, h6, p, span, div, label { color: #f1f5f9 !important; }
            [data-testid="stSidebar"] { background: #111827 !important; border-right: 1px solid #2d3748; }
            [data-testid="stSidebar"] * { color: #e5e7eb !important; }
            #MainMenu, footer, header {visibility: hidden;}
        </style>
        """
    else:
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            * { font-family: 'Inter', sans-serif; }
            .stApp { background: #f8f9fa; }
            .main-header {
                background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                color: white; padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;
            }
            .main-header h1 { margin: 0; font-size: 1.5rem; font-weight: 600; color: white !important; }
            .main-header p { margin: 5px 0 0 0; font-size: 0.85rem; opacity: 0.8; color: white !important; }
            .section-card {
                background: white; border-radius: 10px; padding: 20px;
                margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .section-title { font-size: 1rem; font-weight: 600; color: #1e3a5f; margin-bottom: 15px; }
            .metric-box { text-align: center; padding: 10px; }
            .metric-label { font-size: 0.75rem; color: #6c757d !important; margin-bottom: 5px; }
            .metric-value { font-size: 1.8rem; font-weight: 600; color: #1e3a5f !important; }
            .metric-delta-positive { color: #28a745; font-size: 0.8rem; }
            .metric-delta-negative { color: #dc3545; font-size: 0.8rem; }
            .status-healthy { color: #28a745; font-weight: 500; }
            .status-warning { color: #ffc107; font-weight: 500; }
            .status-critical { color: #dc3545; font-weight: 500; }
            .success-banner {
                background: #d1fae5; border-left: 4px solid #10b981;
                padding: 12px 16px; border-radius: 0 8px 8px 0; color: #065f46;
            }
            .action-card {
                background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
                padding: 15px; margin-bottom: 10px;
            }
            h1, h2, h3 { color: #1e3a5f !important; }
            #MainMenu, footer, header {visibility: hidden;}
        </style>
        """

st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def log_activity(msg, agent="System"):
    st.session_state.activity_log.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "message": msg
    })

def load_network(network_name, drop_line=None):
    """Load network with optional line outage"""
    networks = {
        "IEEE 69-Bus": pn.case69,
        "IEEE 33-Bus": pn.case33bw,
        "IEEE 30-Bus": pn.case30,
        "IEEE 14-Bus": pn.case14,
        "IEEE 4-Bus": pn.case4gs,
    }
    net = networks.get(network_name, pn.case33bw)()
    if drop_line is not None and drop_line < len(net.line):
        net.line.at[drop_line, "in_service"] = False
    return net

def create_network_visualization(net, violations, theme="light", title="Network State"):
    """Create network topology visualization with voltage coloring"""
    bg_color = '#0f1419' if theme == 'dark' else 'white'
    text_color = '#f1f5f9' if theme == 'dark' else '#2c3e50'
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    G = nx.Graph()
    n_buses = len(net.bus)
    
    for i in net.bus.index:
        G.add_node(i)
    
    for line_idx in net.line.index:
        from_bus = int(net.line.at[line_idx, "from_bus"])
        to_bus = int(net.line.at[line_idx, "to_bus"])
        in_service = net.line.at[line_idx, "in_service"]
        G.add_edge(from_bus, to_bus, in_service=in_service, line_idx=line_idx)
    
    try:
        pos = nx.bfs_layout(G, list(G.nodes())[0], align='horizontal')
        pos = {node: (y, -x) for node, (x, y) in pos.items()}
    except:
        pos = nx.spring_layout(G, k=2, seed=42)
    
    voltage_violations = {v.get("bus") for v in violations.get("voltage", [])}
    thermal_violations = {t.get("line") for t in violations.get("thermal", [])}
    
    # Get voltages for coloring
    voltages = {}
    for i in net.bus.index:
        try:
            vm = net.res_bus.at[i, 'vm_pu'] if not net.res_bus.empty else 1.0
            voltages[i] = vm if not pd.isna(vm) else 1.0
        except:
            voltages[i] = 1.0
    
    # Draw edges
    for edge in G.edges(data=True):
        from_bus, to_bus, data = edge
        in_service = data.get("in_service", True)
        line_idx = data.get("line_idx")
        x = [pos[from_bus][0], pos[to_bus][0]]
        y = [pos[from_bus][1], pos[to_bus][1]]
        
        if not in_service:
            ax.plot(x, y, 'k--', linewidth=1.5, alpha=0.3, zorder=1)
        elif line_idx in thermal_violations:
            ax.plot(x, y, color='#e74c3c', linewidth=3, zorder=1)
        else:
            ax.plot(x, y, color='#3498db', linewidth=2, zorder=1)
    
    # Draw nodes with voltage coloring
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=0.9, vmax=1.1)
    
    for node in G.nodes():
        x, y = pos[node]
        vm = voltages.get(node, 1.0)
        color = cmap(norm(vm))
        ax.scatter(x, y, s=150, c=[color], edgecolors='white', linewidths=1.5, zorder=2)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=15, pad=0.02)
    cbar.set_label('V (p.u.)', fontsize=9, color=text_color)
    cbar.ax.tick_params(labelsize=7, colors=text_color)
    
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left', color=text_color)
    ax.axis('off')
    plt.tight_layout()
    return fig


def create_voltage_comparison_chart(net_initial, net_final, theme="light"):
    """Create bus voltage comparison scatter plot"""
    bg_color = '#0f1419' if theme == 'dark' else 'white'
    text_color = '#f1f5f9' if theme == 'dark' else '#2c3e50'
    
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    buses = list(net_initial.bus.index)
    v_initial = [net_initial.res_bus.at[b, 'vm_pu'] if not net_initial.res_bus.empty else 1.0 for b in buses]
    v_final = [net_final.res_bus.at[b, 'vm_pu'] if not net_final.res_bus.empty else 1.0 for b in buses]
    
    ax.scatter(buses, v_initial, c='#3498db', label='Initial', s=40, alpha=0.7, marker='o')
    ax.scatter(buses, v_final, c='#2ecc71', label='Final', s=40, alpha=0.7, marker='s')
    
    ax.axhline(y=0.95, color='#e74c3c', linestyle='--', alpha=0.5, label='Min (0.95)')
    ax.axhline(y=1.05, color='#e74c3c', linestyle='--', alpha=0.5, label='Max (1.05)')
    
    ax.set_xlabel('Bus', color=text_color)
    ax.set_ylabel('Voltage (p.u.)', color=text_color)
    ax.set_title('Bus Voltage Comparison', color=text_color, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    
    plt.tight_layout()
    return fig


def create_thermal_chart(violations, theme="light"):
    """Create thermal violations bar chart"""
    bg_color = '#0f1419' if theme == 'dark' else 'white'
    text_color = '#f1f5f9' if theme == 'dark' else '#2c3e50'
    
    thermal = violations.get("thermal", [])
    if not thermal:
        fig, ax = plt.subplots(figsize=(10, 3), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        ax.text(0.5, 0.5, 'No thermal violations', ha='center', va='center', color=text_color)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 3), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    lines = [f"L{t.get('line', 0)}" for t in thermal]
    loadings = [t.get("loading_percent", 0) for t in thermal]
    colors = ['#dc3545' if l > 120 else '#ffc107' if l > 100 else '#28a745' for l in loadings]
    
    ax.bar(lines, loadings, color=colors)
    ax.axhline(y=100, color='#e74c3c', linestyle='--', alpha=0.7)
    ax.set_ylabel('Loading %', color=text_color)
    ax.set_title('Line Loading Violations', color=text_color, fontweight='bold')
    ax.tick_params(colors=text_color)
    
    plt.tight_layout()
    return fig


def execute_action(net, action):
    """Execute a control action on the network"""
    atype = action.get("action_type", "")
    target = action.get("target")
    value = action.get("value", 1.0)
    
    try:
        if "close" in atype.lower() or atype == "update_switch_status" and action.get("params", {}).get("status") == "close":
            if target is not None and target < len(net.line):
                net.line.at[target, "in_service"] = True
                return True, f"Closed line {target}"
        
        elif "open" in atype.lower():
            if target is not None and target < len(net.line):
                net.line.at[target, "in_service"] = False
                return True, f"Opened line {target}"
        
        elif atype == "reduce_generation" and target is not None:
            if target in net.gen.index:
                old_val = net.gen.at[target, "p_mw"]
                new_val = max(0, old_val - value)
                net.gen.at[target, "p_mw"] = new_val
                return True, f"Reduced gen {target}: {old_val:.1f} → {new_val:.1f} MW"
        
        elif atype == "increase_generation" and target is not None:
            if target in net.gen.index:
                old_val = net.gen.at[target, "p_mw"]
                max_p = net.gen.at[target, "max_p_mw"] if "max_p_mw" in net.gen.columns else old_val + 50
                new_val = min(max_p, old_val + value)
                net.gen.at[target, "p_mw"] = new_val
                return True, f"Increased gen {target}: {old_val:.1f} → {new_val:.1f} MW"
        
        elif atype in ["curtail_load", "reduce_load"] and target is not None:
            bus_loads = net.load[net.load.bus == target].index if atype == "reduce_load" else [target]
            if len(bus_loads) > 0:
                reduction = value if 0 <= value <= 1 else 0.1
                for load_idx in bus_loads:
                    if load_idx in net.load.index:
                        net.load.at[load_idx, "p_mw"] *= (1 - reduction)
                return True, f"Reduced load at bus {target} by {reduction*100:.0f}%"
        
        elif atype == "add_battery" and target is not None and BATTERY_AVAILABLE:
            result = st.session_state.battery_agent.add_battery(net, target, value)
            if result.get("success"):
                return True, f"Added {value} MW battery at bus {target}"
        
        elif atype == "switch_capacitor" and target is not None:
            if target in net.bus.index:
                pp.create_shunt(net, target, q_mvar=-value, p_mw=0)
                return True, f"Added {value} MVAR capacitor at bus {target}"
        
        return False, f"Unknown action: {atype}"
    except Exception as e:
        return False, str(e)


def run_auto_resolution(net, max_iter=5):
    """Run automatic violation resolution"""
    history = []
    
    for i in range(max_iter):
        violations = st.session_state.solver.detect_violations(net)
        total_v = violations.get("summary", {}).get("total_violations", 0)
        voltage_v = violations.get("summary", {}).get("voltage_violations", 0)
        thermal_v = violations.get("summary", {}).get("thermal_violations", 0)
        
        history.append({"iter": i+1, "violations": total_v, "voltage": voltage_v, "thermal": thermal_v})
        log_activity(f"Iter {i+1}: {total_v} violations ({voltage_v}V, {thermal_v}T)", "Auto")
        
        if total_v == 0:
            break
        
        try:
            plan = st.session_state.orchestrator.plan_control_actions(net, violations)
            actions = plan.get("actions", [])
        except:
            actions = generate_fallback_actions(violations)
        
        if not actions:
            actions = generate_fallback_actions(violations)
        
        for action in actions[:5]:
            success, msg = execute_action(net, action)
            if success:
                st.session_state.executed_actions.append({"action": action, "result": msg})
                log_activity(msg, "Auto")
        
        try:
            net = st.session_state.solver.run(net)
        except:
            break
    
    return net, history


def generate_fallback_actions(violations):
    """Generate rule-based fallback actions"""
    actions = []
    
    for v in violations.get("voltage", [])[:3]:
        bus_id = v.get("bus")
        voltage = v.get("voltage_pu", 1.0)
        if voltage < 0.95:
            actions.append({"action_type": "switch_capacitor", "target": bus_id, "value": 0.5, "priority": 2})
        elif voltage > 1.05:
            actions.append({"action_type": "switch_capacitor", "target": bus_id, "value": -0.5, "priority": 2})
    
    for t in violations.get("thermal", [])[:3]:
        from_bus = t.get("from_bus", 0)
        loading = t.get("loading_percent", 0)
        if loading > 100:
            actions.append({"action_type": "reduce_load", "target": from_bus, "value": min(0.2, (loading-100)/200), "priority": 1})
    
    return actions

"""
⚡ VoltAI Power Grid Dashboard
AI-Powered Electrical Network Violation Analysis & Resolution
With Dark/Light Theme Toggle
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandapower.networks as pn
import pandas as pd
import numpy as np
import json
import time
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

# Import grid components
from grid_agent.core.power_flow_solver import PowerFlowSolver
from grid_agent.core.orchestrator import Orchestrator
from grid_agent.agents.thermal_control_agent import ThermalControlAgent
from grid_agent.agents.voltage_control_agent import VoltageControlAgent

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
    page_title="VoltAI Power Grid Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE
# ============================================================================
if "solver" not in st.session_state:
    st.session_state.solver = PowerFlowSolver()
    st.session_state.orchestrator = Orchestrator()
    st.session_state.voltage_agent = VoltageControlAgent()
    st.session_state.thermal_agent = ThermalControlAgent()
    st.session_state.activity_log = []
    st.session_state.net = None
    st.session_state.net_original = None
    st.session_state.violations = None
    st.session_state.mode = "auto"
    st.session_state.start_time = None
    st.session_state.actions_in_plan = 0
    st.session_state.actions_executed = 0
    st.session_state.theme = "dark"  # Default theme
    
    # Configuration defaults
    if "v_min" not in st.session_state:
        st.session_state.v_min = 0.95
    if "v_max" not in st.session_state:
        st.session_state.v_max = 1.05
    if "thermal_limit" not in st.session_state:
        st.session_state.thermal_limit = 100
    if "show_impact" not in st.session_state:
        st.session_state.show_impact = False
    if "current_plan" not in st.session_state:
        st.session_state.current_plan = []
    if "plan_generated" not in st.session_state:
        st.session_state.plan_generated = False



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
                color: white;
                padding: 20px 30px;
                border-radius: 12px;
                margin-bottom: 20px;
            }
            
            .main-header h1 { margin: 0; font-size: 1.5rem; font-weight: 600; }
            .main-header p { margin: 5px 0 0 0; font-size: 0.85rem; opacity: 0.8; }
            
            .section-card {
                background: #1a1f2e;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                border: 1px solid #2d3748;
            }
            
            .section-title {
                font-size: 1rem;
                font-weight: 600;
                color: #60a5fa;
                margin-bottom: 15px;
            }
            
            .metric-box { text-align: center; padding: 10px; }
            .metric-label { font-size: 0.75rem; color: #9ca3af; margin-bottom: 5px; }
            .metric-value { font-size: 1.8rem; font-weight: 600; color: #f1f5f9; }
            
            .status-healthy { color: #22c55e; font-weight: 500; }
            .status-healthy::before { content: '●'; margin-right: 6px; }
            
            .status-warning { color: #eab308; font-weight: 500; }
            .status-warning::before { content: '●'; margin-right: 6px; }
            
            .status-critical { color: #ef4444; font-weight: 500; }
            .status-critical::before { content: '●'; margin-right: 6px; }
            
            .health-bar { background: #374151; border-radius: 4px; height: 8px; margin: 10px 0; }
            .health-bar-fill { height: 100%; border-radius: 4px; }
            
            .alert-banner {
                background: #422006;
                border-left: 4px solid #f59e0b;
                padding: 10px 15px;
                border-radius: 0 6px 6px 0;
                font-size: 0.85rem;
                color: #fcd34d;
            }
            
            .auto-progress {
                background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                margin-top: 15px;
            }
            
            [data-testid="stSidebar"] {
                background: #111827 !important;
                border-right: 1px solid #2d3748;
            }
            
            [data-testid="stSidebar"] * { color: #e5e7eb !important; }
            
            [data-testid="stSidebar"] .stButton button {
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
            }
            
            h1, h2, h3, h4, h5, h6, p, span, div, label { color: #f1f5f9 !important; }
            
            .stTabs [data-baseweb="tab-list"] { background: #1f2937; border-radius: 8px; }
            .stTabs [data-baseweb="tab"] { color: #9ca3af; }
            .stTabs [aria-selected="true"] { color: #60a5fa; background: #374151; }
            
            .stDataFrame { background: #1a1f2e; }
            .stMetric { background: #1a1f2e; padding: 10px; border-radius: 8px; }
            .stMetric label { color: #9ca3af !important; }
            .stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """
    else:  # Light theme
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            * { font-family: 'Inter', sans-serif; }
            
            .stApp { background: #f8f9fa; }
            
            .main-header {
                background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                color: white;
                padding: 20px 30px;
                border-radius: 12px;
                margin-bottom: 20px;
            }
            
            .main-header h1 { margin: 0; font-size: 1.5rem; font-weight: 600; color: white !important; }
            .main-header p { margin: 5px 0 0 0; font-size: 0.85rem; opacity: 0.8; color: white !important; }
            
            .section-card {
                background: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            .section-title {
                font-size: 1rem;
                font-weight: 600;
                color: #1e3a5f;
                margin-bottom: 15px;
            }
            
            .metric-box { text-align: center; padding: 10px; }
            .metric-label { font-size: 0.75rem; color: #6c757d !important; margin-bottom: 5px; }
            .metric-value { font-size: 1.8rem; font-weight: 600; color: #1e3a5f !important; }
            
            .status-healthy { color: #28a745; font-weight: 500; }
            .status-healthy::before { content: '●'; margin-right: 6px; }
            
            .status-warning { color: #ffc107; font-weight: 500; }
            .status-warning::before { content: '●'; margin-right: 6px; }
            
            .status-critical { color: #dc3545; font-weight: 500; }
            .status-critical::before { content: '●'; margin-right: 6px; }
            
            .health-bar { background: #e9ecef; border-radius: 4px; height: 8px; margin: 10px 0; }
            .health-bar-fill { height: 100%; border-radius: 4px; }
            
            .alert-banner {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 10px 15px;
                border-radius: 0 6px 6px 0;
                font-size: 0.85rem;
                color: #856404 !important;
            }
            
            .auto-progress {
                background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                margin-top: 15px;
            }
            
            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid #e9ecef;
            }
            
            [data-testid="stSidebar"] .stButton button {
                background: #1e3a5f;
                color: white;
                border: none;
                border-radius: 6px;
            }
            
            h1, h2, h3 { color: #1e3a5f !important; }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """

# Apply theme CSS
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
    """Load network"""
    networks = {
        "IEEE 33-Bus": pn.case33bw,
        "IEEE 30-Bus": pn.case30,
        "IEEE 14-Bus": pn.case14,
        "IEEE 4-Bus": pn.case4gs,
    }
    
    print(f"DEBUG: Attempting to load network {network_name}")
    try:
        net = networks.get(network_name, pn.case33bw)()
        print(f"DEBUG: Network loaded successfully: {net}")
        print(f"DEBUG: Bus count: {len(net.bus)}")
        print(f"DEBUG: Gen count: {len(net.gen)}")
    except Exception as e:
        print(f"DEBUG: Failed to load network: {e}")
        return None
    
    if drop_line is not None and drop_line < len(net.line):
        net.line.at[drop_line, "in_service"] = False
    
    return net


# Auto-load network on startup (placed here to ensure load_network is defined)
if st.session_state.net is None:
    print("DEBUG: Auto-loading IEEE 33-Bus on startup")
    st.session_state.net = load_network("IEEE 33-Bus")
    st.session_state.net_original = copy.deepcopy(st.session_state.net)
    st.session_state.violations = st.session_state.solver.detect_violations(st.session_state.net)
    st.session_state.network_loaded = True
    st.session_state.start_time = datetime.now()
    log_activity("Auto-loaded IEEE 33-Bus", "System")

def create_tree_topology(net, violations, theme="light"):
    """Create tree-style network topology"""
    bg_color = '#0f1419' if theme == 'dark' else 'white'
    text_color = '#f1f5f9' if theme == 'dark' else '#2c3e50'
    
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    G = nx.Graph()
    n_buses = len(net.bus)
    
    for i in range(n_buses):
        G.add_node(i)
    
    for line_idx in net.line.index:
        from_bus = int(net.line.at[line_idx, "from_bus"])
        to_bus = int(net.line.at[line_idx, "to_bus"])
        in_service = net.line.at[line_idx, "in_service"]
        G.add_edge(from_bus, to_bus, in_service=in_service, line_idx=line_idx)
    
    try:
        pos = nx.bfs_layout(G, 0, align='horizontal')
        pos = {node: (y, -x) for node, (x, y) in pos.items()}
    except:
        pos = nx.spring_layout(G, k=2, seed=42)
    
    voltage_violations = {v.get("bus") for v in violations.get("voltage", [])}
    thermal_violations = {t.get("line") for t in violations.get("thermal", [])}
    
    # Get voltages
    voltages = {}
    for i in range(n_buses):
        try:
            vm = net.res_bus.at[i, 'vm_pu'] if not net.res_bus.empty else 1.0
            voltages[i] = vm if not pd.isna(vm) else None
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
    
    # Draw nodes
    for node in G.nodes():
        x, y = pos[node]
        vm = voltages.get(node)
        
        if vm is None:
            color = '#95a5a6'
        elif node in voltage_violations:
            color = '#e74c3c'
        elif 0.95 <= vm <= 1.05:
            color = '#2ecc71'
        else:
            color = '#f39c12'
        
        ax.scatter(x, y, s=200, c=color, edgecolors='white', linewidths=2, zorder=2)
        ax.annotate(str(node), (x, y), ha='center', va='center', fontsize=7, 
                    fontweight='bold', color='white', zorder=3)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0.9, vmax=1.1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Voltage (p.u.)', fontsize=10, color=text_color)
    cbar.ax.tick_params(labelsize=8, colors=text_color)
    
    ax.set_title("Network Topology", fontsize=11, fontweight='bold', loc='left', color=text_color)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def create_voltage_profile_chart(net, violations):
    """Create interactive voltage profile chart using Plotly"""
    # Extract voltage data
    buses = []
    voltages = []
    status = []
    colors = []
    
    v_min = st.session_state.v_min
    v_max = st.session_state.v_max
    
    for idx in net.res_bus.index:
        bus_id = idx
        vm = net.res_bus.at[idx, 'vm_pu']
        
        buses.append(f"Bus {bus_id}")
        
        if pd.isna(vm):
            voltages.append(0)
            status.append("Disconnected")
            colors.append("#ef4444") # Red
        else:
            voltages.append(vm)
            if vm < v_min:
                status.append(f"Undervoltage ({vm:.4f} < {v_min})")
                colors.append("#eab308") # Yellow/Orange
            elif vm > v_max:
                status.append(f"Overvoltage ({vm:.4f} > {v_max})")
                colors.append("#eab308")
            else:
                status.append("Healthy")
                colors.append("#22c55e") # Green
    
    df = pd.DataFrame({
        "Bus": buses,
        "Voltage (p.u.)": voltages,
        "Status": status,
        "Color": colors
    })
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=df["Bus"],
        y=df["Voltage (p.u.)"],
        marker_color=df["Color"],
        text=df["Voltage (p.u.)"].apply(lambda x: f"{x:.3f}"),
        textposition='auto',
        hovertemplate="<b>%{x}</b><br>Voltage: %{y:.4f} p.u.<br>Status: %{customdata}<extra></extra>",
        customdata=df["Status"]
    ))
    
    # Add threshold lines
    fig.add_hline(y=v_min, line_dash="dash", line_color="white", annotation_text="Min Limit")
    fig.add_hline(y=v_max, line_dash="dash", line_color="white", annotation_text="Max Limit")
    fig.add_hline(y=1.0, line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Bus Voltage Profile",
        xaxis_title="Bus ID",
        yaxis_title="Voltage (p.u.)",
        yaxis_range=[0.8, 1.2],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#9ca3af"),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_thermal_loading_chart(net, violations):
    """Create interactive thermal loading chart using Plotly"""
    if net.res_line.empty:
        return go.Figure()
        
    lines = []
    loading = []
    status = []
    colors = []
    
    limit = st.session_state.thermal_limit
    
    for idx in net.res_line.index:
        l_load = net.res_line.at[idx, 'loading_percent']
        lines.append(f"Line {idx}")
        loading.append(l_load)
        
        if l_load > limit:
            status.append(f"Overloaded ({l_load:.1f}% > {limit}%)")
            colors.append("#ef4444") # Red
        elif l_load > 80:
            status.append("High Loading")
            colors.append("#eab308") # Orange
        else:
            status.append("Normal")
            colors.append("#22c55e") # Green
            
    df = pd.DataFrame({
        "Line": lines,
        "Loading (%)": loading,
        "Status": status,
        "Color": colors
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["Line"],
        y=df["Loading (%)"],
        marker_color=df["Color"],
        hovertemplate="<b>%{x}</b><br>Loading: %{y:.1f}%<br>Status: %{customdata}<extra></extra>",
        customdata=df["Status"]
    ))
    
    fig.add_hline(y=limit, line_dash="dash", line_color="red", annotation_text="Thermal Limit")
    
    fig.update_layout(
        title="Line Thermal Loading",
        xaxis_title="Line ID",
        yaxis_title="Loading (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#9ca3af"),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def execute_action(net, action):
    """Execute a control action on the network"""
    atype = action.get("action_type", "")
    target = action.get("target")
    value = action.get("value", 1.0)
    
    try:
        # Line operations
        if "close" in atype and target is not None:
            if target < len(net.line):
                net.line.at[target, "in_service"] = True
                return True, f"Closed line {target}"
        
        elif "open" in atype and target is not None:
            if target < len(net.line):
                net.line.at[target, "in_service"] = False
                return True, f"Opened line {target}"
        
        # Generation control
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
        
        # Load control
        elif atype == "curtail_load" and target is not None:
            if target in net.load.index:
                old_load = net.load.at[target, "p_mw"]
                reduction_factor = value if 0 <= value <= 1 else 0.1
                net.load.at[target, "p_mw"] *= (1 - reduction_factor)
                return True, f"Curtailed load {target}: {old_load:.1f} → {old_load*(1-reduction_factor):.1f} MW"
        
        elif atype == "reduce_load" and target is not None:
            # Find loads at bus and reduce them
            bus_loads = net.load[net.load.bus == target].index
            if len(bus_loads) > 0:
                reduction_factor = value if 0 <= value <= 1 else 0.1
                for load_idx in bus_loads:
                    old_load = net.load.at[load_idx, "p_mw"]
                    net.load.at[load_idx, "p_mw"] *= (1 - reduction_factor)
                return True, f"Reduced loads at bus {target} by {reduction_factor*100:.0f}%"
        
        elif atype == "emergency_shed" and target is not None:
            # Emergency load shedding at bus
            bus_loads = net.load[net.load.bus == target].index
            total_shed = 0
            for load_idx in bus_loads:
                old_load = net.load.at[load_idx, "p_mw"]
                shed_amount = min(old_load, value / max(1, len(bus_loads)))
                net.load.at[load_idx, "p_mw"] -= shed_amount
                total_shed += shed_amount
            if total_shed > 0:
                return True, f"Emergency shed at bus {target}: {total_shed:.1f} MW"
        
        # Voltage control
        elif atype == "switch_capacitor" and target is not None:
            if target in net.bus.index:
                import pandapower as pp
                pp.create_shunt(net, target, q_mvar=-value, p_mw=0)
                return True, f"Added {value} MVAR capacitor at bus {target}"
        
        elif atype == "adjust_gen_voltage" and target is not None:
            if target in net.gen.index:
                old_val = net.gen.at[target, "vm_pu"]
                net.gen.at[target, "vm_pu"] = value
                return True, f"Changed gen {target} voltage: {old_val:.3f} → {value:.3f} pu"
        
        elif atype == "adjust_tap" and target is not None:
            if hasattr(net, 'trafo') and len(net.trafo) > 0:
                # Find transformer connected to target bus
                for trafo_idx in net.trafo.index:
                    if net.trafo.at[trafo_idx, "hv_bus"] == target or net.trafo.at[trafo_idx, "lv_bus"] == target:
                        net.trafo.at[trafo_idx, "tap_pos"] = value
                        return True, f"Adjusted transformer {trafo_idx} tap to {value}"
        
        return False, f"Unknown or unsupported action: {atype}"
    except Exception as e:
        return False, str(e)


def run_auto_resolution(net, max_iter=5):
    """
    Run automatic violation resolution with improved action handling.
    Uses fallback actions if orchestrator/LLM response is slow or fails.
    """
    history = []
    
    for i in range(max_iter):
        # Detect violations
        violations = st.session_state.solver.detect_violations(net)
        total_v = violations.get("summary", {}).get("total_violations", 0)
        voltage_v = violations.get("summary", {}).get("voltage_violations", 0)
        thermal_v = violations.get("summary", {}).get("thermal_violations", 0)
        
        history.append({
            "iter": i+1, 
            "violations": total_v,
            "voltage": voltage_v,
            "thermal": thermal_v
        })
        
        # Stop if no violations
        if total_v == 0:
            log_activity(f"All violations resolved in iteration {i+1}", "Auto")
            break
        
        log_activity(f"Iteration {i+1}: {total_v} violations ({voltage_v} voltage, {thermal_v} thermal)", "Auto")
        
        # Get actions from orchestrator (may be slow due to LLM)
        try:
            plan = st.session_state.orchestrator.plan_control_actions(net, violations)
            actions = plan.get("actions", [])
        except Exception as e:
            log_activity(f"Orchestrator error: {str(e)[:50]}", "Auto")
            actions = []
        
        # Generate fallback actions if no actions returned (faster than LLM)
        if not actions:
            actions = _generate_fallback_actions(violations)
            log_activity(f"Using {len(actions)} fallback actions", "Auto")
        
        # Execute up to 5 actions per iteration
        executed_count = 0
        for action in actions[:5]:
            success, msg = execute_action(net, action)
            if success:
                executed_count += 1
                log_activity(msg, "Auto")
        
        if executed_count == 0:
            log_activity("No actions could be executed", "Auto")
        
        # Run power flow after applying actions
        try:
            net = st.session_state.solver.run(net)
        except Exception as e:
            log_activity(f"Power flow failed: {str(e)[:50]}", "Auto")
            break
    
    return net, history


def _generate_fallback_actions(violations):
    """Generate simple rule-based fallback actions without LLM"""
    actions = []
    
    # Handle voltage violations
    for v in violations.get("voltage", [])[:3]:
        bus_id = v.get("bus")
        voltage = v.get("voltage_pu", 1.0)
        
        if voltage < 0.95:
            # Low voltage - try switching in capacitor
            actions.append({
                "action_type": "switch_capacitor",
                "target": bus_id,
                "value": 0.5,  # Add 0.5 MVAR
                "priority": 2,
                "reason": f"Boost voltage at bus {bus_id}"
            })
        elif voltage > 1.05:
            # High voltage - reduce capacitor
            actions.append({
                "action_type": "switch_capacitor",
                "target": bus_id,
                "value": -0.5,  # Remove 0.5 MVAR
                "priority": 2,
                "reason": f"Reduce voltage at bus {bus_id}"
            })
    
    # Handle thermal violations  
    for t in violations.get("thermal", [])[:3]:
        line_id = t.get("line")
        loading = t.get("loading_percent", 0)
        from_bus = t.get("from_bus", 0)
        
        if loading > 100:
            # Try load reduction first
            overload_pct = loading - 100
            reduction_factor = min(0.2, overload_pct / 200)  # Up to 20% reduction
            
            actions.append({
                "action_type": "reduce_load",
                "target": from_bus,
                "value": reduction_factor,
                "priority": 1,
                "reason": f"Reduce overload on line {line_id}"
            })
    
    return actions


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    show_settings = st.toggle("⚙️ Show Sidebar Controls", value=True)
    
    if show_settings:
        st.markdown("### ⚙️ Configuration")
        
        # Theme toggle
        st.markdown("##### 🎨 Theme")
        theme_choice = st.radio(
            "Select Theme",
            ["☀️ Light Mode", "🌙 Dark Mode"],
            index=0 if st.session_state.theme == "light" else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        new_theme = "light" if "Light" in theme_choice else "dark"
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("##### 🔌 Network Selection")
        network_name = st.selectbox(
            "Network",
            ["IEEE 33-Bus", "IEEE 30-Bus", "IEEE 14-Bus", "IEEE 4-Bus"],
            index=0
        )
        
        drop_line = st.selectbox(
            "Drop Line (create outage)",
            [None, 3, 5, 10, 12, 15],
            index=0,
            format_func=lambda x: "None" if x is None else f"Line {x}"
        )
        
        st.markdown("---")
        
        st.markdown("##### ⚡ Voltage Thresholds")
        col1, col2 = st.columns(2)
        with col1:
            v_min = st.number_input("Min (p.u.)", 0.90, 1.00, 0.95, 0.01)
        with col2:
            v_max = st.number_input("Max (p.u.)", 1.00, 1.10, 1.05, 0.01)
        
        if st.button("Update Thresholds"):
            st.success("Updated")
        
        st.markdown("---")
        
        st.markdown("##### 📊 Session Info")
        st.text(f"Start Time: {datetime.now().strftime('%H:%M:%S')}")
        st.text(f"Network Loaded: {st.session_state.network_loaded}")
        st.text(f"Network: {network_name}")
        st.text(f"Mode: {st.session_state.mode}")
        st.text(f"Actions in Plan: {st.session_state.actions_in_plan}")
        st.text(f"Actions Executed: {st.session_state.actions_executed}")
        st.text(f"Activities Logged: {len(st.session_state.activity_log)}")
        
        st.markdown("---")
        show_dashboard = st.checkbox("Show Dashboard", value=True)
        st.checkbox("🐛 Debug Mode")
    else:
        # Defaults if hidden to prevent NameErrors
        network_name = "IEEE 33-Bus"
        drop_line = None
        show_dashboard = True # Keep dashboard visible by default if sidebar hidden
        v_min, v_max = 0.95, 1.05


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>⚡ VoltAI Power Grid Dashboard</h1>
    <p>AI-Powered Electrical Network Violation Analysis & Resolution</p>
</div>
""", unsafe_allow_html=True)

# Load network button
if st.button("🔄 Load Network & Analyze", use_container_width=True):
    st.session_state.net = load_network(network_name, drop_line)
    st.session_state.net_original = copy.deepcopy(st.session_state.net)
    st.session_state.violations = st.session_state.solver.detect_violations(st.session_state.net)
    st.session_state.network_loaded = True
    st.session_state.start_time = datetime.now()
    log_activity(f"Loaded {network_name}", "System")

print("DEBUG: Starting VoltAI Dashboard v2...")

# Main content
if show_dashboard and st.session_state.network_loaded and st.session_state.net is not None:
    net = st.session_state.net
    violations = st.session_state.violations
    summary = violations.get("summary", {})
    
    print(f"DEBUG: Displaying metrics. Net: {net}")
    print(f"DEBUG: Res Gen Empty? {net.res_gen.empty}")
    if not net.res_gen.empty:
        print(f"DEBUG: Total Gen: {net.res_gen.p_mw.sum()}")
    
    # Network Health Overview
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Network Health Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Total Buses</div><div class="metric-value">{len(net.bus)}</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Total Lines</div><div class="metric-value">{len(net.line)}</div></div>', unsafe_allow_html=True)
    
    with col3:
        v_violations = len(violations.get("voltage", []))
        status = "healthy" if v_violations == 0 else "warning" if v_violations < 5 else "critical"
        text = "Healthy" if v_violations == 0 else f"{v_violations} Issues"
        st.markdown(f'<div class="metric-box"><div class="metric-label">Voltage Status</div><div class="status-{status}">{text}</div></div>', unsafe_allow_html=True)
    
    with col4:
        t_violations = len(violations.get("thermal", []))
        status = "healthy" if t_violations == 0 else "warning" if t_violations < 5 else "critical"
        text = "Healthy" if t_violations == 0 else f"{t_violations} Issues"
        st.markdown(f'<div class="metric-box"><div class="metric-label">Thermal Status</div><div class="status-{status}">{text}</div></div>', unsafe_allow_html=True)
    
    with col5:
        try:
            disconnected = len(net.res_bus[net.res_bus['vm_pu'].isna()])
        except:
            disconnected = 0
        status = "healthy" if disconnected == 0 else "critical"
        text = "Connected" if disconnected == 0 else f"{disconnected} Disconnected"
        st.markdown(f'<div class="metric-box"><div class="metric-label">Connectivity</div><div class="status-{status}">{text}</div></div>', unsafe_allow_html=True)
    
    # Health bar
    health = summary.get("network_health_percent", 100)
    bar_color = "#28a745" if health > 80 else "#ffc107" if health > 50 else "#dc3545"
    st.markdown(f'<div class="health-bar"><div class="health-bar-fill" style="width: {health}%; background: {bar_color};"></div></div>', unsafe_allow_html=True)
    
    total_v = summary.get("total_violations", 0)
    if total_v > 0:
        st.markdown(f'<div class="alert-banner">⚠️ Network has {total_v} violations requiring attention.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Network Visualization
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Network Visualization</div>', unsafe_allow_html=True)
    
    view_tab1, view_tab2, view_tab3 = st.tabs(["🗺️ Network Map", "📊 Voltage Profile", "🔥 Thermal Loading"])
    
    with view_tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            plot_type = st.selectbox("Plot Type", ["Network Topology"], label_visibility="collapsed")
        with col2:
            color_scheme = st.selectbox("Color Scheme", ["Default", "Voltage"], label_visibility="collapsed")
        with col3:
            show_labels = st.checkbox("Show Labels", value=True)
        with col4:
            interactive = st.checkbox("Interactive Mode", value=False)
        
        fig = create_tree_topology(net, violations, st.session_state.theme)
        st.pyplot(fig)
        plt.close(fig)
        
    with view_tab2:
        st.plotly_chart(create_voltage_profile_chart(net, violations), use_container_width=True)
        
    with view_tab3:
        st.plotly_chart(create_thermal_loading_chart(net, violations), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Impact Analysis (Comparison)
    if st.session_state.net_original is not None and st.session_state.get('show_impact', False):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚖️ Impact Analysis (Before vs After)</div>', unsafe_allow_html=True)
        
        tab_visual, tab_metrics = st.tabs(["🖼️ Visual Comparison", "📊 Detailed Metrics"])
        
        with tab_visual:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Before (Original State)**")
                # Use a cached/simple plot for original to save time
                fig_orig = create_tree_topology(st.session_state.net_original, {}, st.session_state.theme)
                st.pyplot(fig_orig)
                plt.close(fig_orig)
                
            with col2:
                st.markdown("**After (Current State)**")
                fig_curr = create_tree_topology(net, violations, st.session_state.theme)
                st.pyplot(fig_curr)
                plt.close(fig_curr)
        
        with tab_metrics:
            # Calculate metrics
            def get_net_metrics(n):
                return {
                    "load_mw": n.res_load.p_mw.sum() if not n.res_load.empty else 0,
                    "gen_mw": (n.res_gen.p_mw.sum() if not n.res_gen.empty else 0) + (n.res_ext_grid.p_mw.sum() if not n.res_ext_grid.empty else 0),
                    "min_voltage": n.res_bus.vm_pu.min() if not n.res_bus.empty else 0,
                    "max_loading": n.res_line.loading_percent.max() if not n.res_line.empty else 0,
                    "losses": (n.res_gen.p_mw.sum() + n.res_ext_grid.p_mw.sum()) - n.res_load.p_mw.sum()
                }
            
            m_orig = get_net_metrics(st.session_state.net_original)
            m_curr = get_net_metrics(net)
            
            # Create comparison table
            comp_data = {
                "Metric": ["Min Voltage (p.u.)", "Max Line Loading (%)", "Total Losses (MW)", "Total Generation (MW)"],
                "Before": [f"{m_orig['min_voltage']:.4f}", f"{m_orig['max_loading']:.1f}%", f"{m_orig['losses']:.2f}", f"{m_orig['gen_mw']:.2f}"],
                "After": [f"{m_curr['min_voltage']:.4f}", f"{m_curr['max_loading']:.1f}%", f"{m_curr['losses']:.2f}", f"{m_curr['gen_mw']:.2f}"],
                "Change": [
                    f"{m_curr['min_voltage'] - m_orig['min_voltage']:+.4f}",
                    f"{m_curr['max_loading'] - m_orig['max_loading']:+.1f}%",
                    f"{m_curr['losses'] - m_orig['losses']:+.2f}",
                    f"{m_curr['gen_mw'] - m_orig['gen_mw']:+.2f}"
                ]
            }
            st.table(pd.DataFrame(comp_data))
            
            # Analysis Text
            st.markdown("### 📝 Executive Summary")
            
            summary_points = []
            
            # Voltage Analysis
            v_diff = m_curr['min_voltage'] - m_orig['min_voltage']
            if v_diff > 0.01:
                summary_points.append(f"The minimum system voltage has significantly improved by **{v_diff:.4f} p.u.**, enhancing overall grid stability.")
            elif v_diff > 0:
                summary_points.append(f"The minimum system voltage saw a slight improvement of **{v_diff:.4f} p.u.**.")
            elif v_diff < -0.01:
                summary_points.append(f"⚠️ Warning: Minimum voltage has degraded by **{abs(v_diff):.4f} p.u.**.")
            
            # Loading Analysis
            l_diff = m_orig['max_loading'] - m_curr['max_loading']
            if l_diff > 10:
                summary_points.append(f"Critical thermal congestion was relieved, with peak line loading reducing by **{l_diff:.1f}%**.")
            elif l_diff > 0:
                summary_points.append(f"Line loading margins improved by **{l_diff:.1f}%**.")
                
            # Losses Analysis
            loss_diff = m_orig['losses'] - m_curr['losses']
            if loss_diff > 0.1:
                summary_points.append(f"Efficiency improvements led to a **{loss_diff:.2f} MW** reduction in total system losses.")
            
            # Overall Conclusion
            if not violations.get('voltage') and not violations.get('thermal') and (len(st.session_state.violations.get('voltage', [])) > 0 or len(st.session_state.violations.get('thermal', [])) > 0):
                summary_points.append("✅ **All previous network violations have been successfully resolved.**")
            
            if summary_points:
                st.info(" ".join(summary_points))
            else:
                st.info("No significant changes detected in key network metrics.")

        st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics tabs
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "⚡ Power Flow", "⚠️ Violations", "📈 Trends"])
    
    with tab1:
        st.markdown("### Network Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Buses", len(net.bus))
            st.metric("Total Lines", len(net.line))
            st.metric("Total Transformers", len(net.trafo) if hasattr(net, 'trafo') else 0)
        
        with col2:
            total_load = net.res_load.p_mw.sum() if not net.res_load.empty else 0
            
            gen_mw = net.res_gen.p_mw.sum() if not net.res_gen.empty else 0
            ext_mw = net.res_ext_grid.p_mw.sum() if not net.res_ext_grid.empty else 0
            total_gen = gen_mw + ext_mw
            
            st.metric("Total Load (MW)", f"{total_load:.2f}")
            st.metric("Total Generation (MW)", f"{total_gen:.2f}")
            st.metric("Network Losses (MW)", f"{total_gen - total_load:.2f}")
        
        with col3:
            st.metric("Voltage Violations", len(violations.get("voltage", [])))
            st.metric("Thermal Violations", len(violations.get("thermal", [])))
            st.metric("Health Score", f"{health:.1f}%")
    
    with tab2:
        st.markdown("### Power Flow Results")
        if not net.res_bus.empty:
            st.dataframe(net.res_bus.head(10), use_container_width=True)
    
    with tab3:
        st.markdown("### Violations")
        if violations.get("voltage"):
            for v in violations["voltage"][:10]:
                v_type = v.get('violation_type', 'unknown').replace('_', ' ').title()
                val = v.get('voltage_pu', 0)
                reason = f"{v_type}: {val:.4f} p.u."
                if v_type == "Disconnected":
                    reason = "Disconnected (Voltage Collapse/Islanded)"
                
                st.warning(f"**Bus {v.get('bus')}** | {reason}")
        
        if violations.get("thermal"):
            for t in violations["thermal"][:10]:
                loading = t.get('loading_percent', 0)
                st.error(f"**Line {t.get('line')}** | Thermal Overload: {loading:.1f}% (Limit: 100%)")
        if not violations.get("voltage") and not violations.get("thermal"):
            st.success("No violations detected!")
    
    with tab4:
        st.info("Trends will be available after multiple analyses")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Operation Mode
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎮 Operation Mode</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🤖 Auto Mode\n\nFully automated resolution", 
                     use_container_width=True,
                     type="primary" if st.session_state.mode == "auto" else "secondary"):
            st.session_state.mode = "auto"
            st.rerun()
    
    with col2:
        if st.button("👤 Interactive Mode\n\nStep-by-step with approval",
                     use_container_width=True,
                     type="primary" if st.session_state.mode == "interactive" else "secondary"):
            st.session_state.mode = "interactive"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mode-specific content
    if st.session_state.mode == "auto":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ Automatic Mode</div>', unsafe_allow_html=True)
        
        if st.button("▶️ Start Automatic Resolution", use_container_width=True, type="primary"):
            with st.spinner("Running automatic resolution..."):
                net, history = run_auto_resolution(net, max_iter=5)
                st.session_state.net = net
                st.session_state.violations = st.session_state.solver.detect_violations(net)
            
            final_v = history[-1]["violations"] if history else 0
            if final_v == 0:
                st.success(f"✅ All violations resolved in {len(history)} iterations!")
            else:
                st.warning(f"⚠️ {final_v} violations remaining")
            
            st.session_state.show_impact = True
            st.rerun()
        
        st.markdown('<div class="auto-progress"><strong>Status:</strong> Ready to start</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Interactive Mode</div>', unsafe_allow_html=True)
        
        # Step 1: Generate Plan
        if not st.session_state.plan_generated:
            st.info("Click 'Generate Fix Plan' to analyze the grid and propose control actions.")
            if st.button("📝 Generate Fix Plan", type="primary", use_container_width=True):
                with st.spinner("Analyzing grid and generating solutions..."):
                    try:
                        plan = st.session_state.orchestrator.plan_control_actions(net, violations)
                        st.session_state.current_plan = plan.get("actions", [])
                        st.session_state.plan_generated = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating plan: {e}")
        
        # Step 2: Review & Modify
        else:
            st.markdown("### 🛠️ Review & Modify Action Plan")
            st.markdown("Edit, reorder, or remove actions below. You can also add new actions.")
            
            # Action List Editor
            if not st.session_state.current_plan:
                st.warning("No actions in current plan.")
            else:
                for i, action in enumerate(st.session_state.current_plan):
                    with st.expander(f"Action {i+1}: {action.get('action_type', 'Unknown')} - {action.get('target', 'N/A')}", expanded=True):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        # Edit Type
                        new_type = col1.selectbox("Type", 
                                                ["switch_line", "switch_capacitor", "adjust_gen_voltage", "adjust_transformer", "shed_load", "add_battery"],
                                                index=["switch_line", "switch_capacitor", "adjust_gen_voltage", "adjust_transformer", "shed_load", "add_battery"].index(action.get('action_type', 'switch_line')) if action.get('action_type') in ["switch_line", "switch_capacitor", "adjust_gen_voltage", "adjust_transformer", "shed_load", "add_battery"] else 0,
                                                key=f"type_{i}")
                        
                        # Edit Value
                        if new_type in ["switch_capacitor", "adjust_gen_voltage", "shed_load", "add_battery"]:
                            new_val = col2.number_input("Value", value=float(action.get("value", 0.0)), key=f"val_{i}")
                            action["value"] = new_val
                        
                        # Delete Button
                        if col3.button("🗑️ Remove", key=f"del_{i}"):
                            st.session_state.current_plan.pop(i)
                            st.rerun()
                        
                        action["action_type"] = new_type
            
            # Add New Action Section
            st.markdown("#### ➕ Add New Action")
            with st.form("add_action_form"):
                c1, c2, c3 = st.columns(3)
                new_act_type = c1.selectbox("Action Type", ["switch_line", "switch_capacitor", "adjust_gen_voltage", "shed_load"])
                new_act_target = c2.number_input("Target ID (Bus/Line)", min_value=0, step=1)
                new_act_val = c3.number_input("Value (MW/MVAR/p.u.)", value=0.0)
                
                if st.form_submit_button("Add to Plan"):
                    st.session_state.current_plan.append({
                        "action_type": new_act_type,
                        "target": int(new_act_target),
                        "value": new_act_val,
                        "reason": "User added manually"
                    })
                    st.rerun()

                    st.rerun()

            # Analysis Sections
            st.markdown("### 📊 Analysis & Insights")
            with st.expander("🔍 Detailed Violation Analysis", expanded=False):
                if violations.get("voltage"):
                    st.markdown("**Voltage Violations:**")
                    for v in violations["voltage"]:
                        st.write(f"- Bus {v['bus']}: {v['voltage_pu']:.4f} p.u. ({v['violation_type']})")
                if violations.get("thermal"):
                    st.markdown("**Thermal Violations:**")
                    for t in violations["thermal"]:
                        st.write(f"- Line {t['line']}: {t['loading_percent']:.1f}% Loading")
            
            with st.expander("📚 Referral Analysis (Context)", expanded=False):
                st.info("Here you would see relevant excerpts from technical documentation or past similar cases.")
                st.markdown("""
                *   **IEEE Standard 1547**: Voltage regulation requirements for distributed resources.
                *   **Operator Guide**: "In case of thermal overload on Line 3, prioritize generation shifting over load shedding."
                """)

            # Step 3: Execute
            st.markdown("---")
            c1, c2 = st.columns(2)
            
            if c1.button("✅ Execute Plan", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                for i, action in enumerate(st.session_state.current_plan):
                    success, msg = execute_action(net, action)
                    if success:
                        st.session_state.actions_executed += 1
                        log_activity(msg, "Interactive")
                    progress_bar.progress((i + 1) / len(st.session_state.current_plan))
                
                # Re-run power flow
                try:
                    net = st.session_state.solver.run(net)
                    st.session_state.net = net
                    st.session_state.violations = st.session_state.solver.detect_violations(net)
                    st.session_state.show_impact = True
                    st.session_state.plan_generated = False # Reset for next cycle
                    st.session_state.current_plan = []
                except:
                    pass
                st.rerun()
            
            if c2.button("❌ Discard Plan", type="secondary", use_container_width=True):
                st.session_state.plan_generated = False
                st.session_state.current_plan = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="section-card" style="text-align: center; padding: 60px;">
        <h2>Welcome to VoltAI</h2>
        <p>Select a network and click "Load Network & Analyze" to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        selected_net = st.selectbox("Select Network to Analyze", 
                                  ["IEEE 33-Bus", "IEEE 30-Bus", "IEEE 14-Bus", "IEEE 4-Bus"],
                                  index=0) # Default to 33-bus to match screenshot
        
        if st.button("🚀 Load Network & Start Analysis", use_container_width=True, type="primary"):
            st.session_state.net = load_network(selected_net)
            st.session_state.net_original = copy.deepcopy(st.session_state.net)
            st.session_state.violations = st.session_state.solver.detect_violations(st.session_state.net)
            st.session_state.network_loaded = True
            st.session_state.start_time = datetime.now()
            st.session_state.show_impact = False
            log_activity(f"Loaded {selected_net}", "System")
            st.rerun()
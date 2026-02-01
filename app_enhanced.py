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
import networkx as nx

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
    st.session_state.network_loaded = False
    st.session_state.actions_in_plan = 0
    st.session_state.actions_executed = 0
    st.session_state.theme = "light"  # Default theme

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
    
    net = networks.get(network_name, pn.case33bw)()
    
    if drop_line is not None and drop_line < len(net.line):
        net.line.at[drop_line, "in_service"] = False
    
    return net


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
        index=0,
        label_visibility="collapsed"
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
    st.checkbox("🐛 Debug Mode")


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

# Main content
if st.session_state.network_loaded and st.session_state.net is not None:
    net = st.session_state.net
    violations = st.session_state.violations
    summary = violations.get("summary", {})
    
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
            total_load = net.load.p_mw.sum() if len(net.load) > 0 else 0
            total_gen = net.gen.p_mw.sum() if len(net.gen) > 0 else 0
            st.metric("Total Load (MW)", f"{total_load:.1f}")
            st.metric("Total Generation (MW)", f"{total_gen:.1f}")
            st.metric("Power Balance", f"{total_gen - total_load:.1f}")
        
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
            for v in violations["voltage"][:5]:
                st.warning(f"Bus {v.get('bus')}: {v.get('voltage_pu', 0):.4f} p.u.")
        if violations.get("thermal"):
            for t in violations["thermal"][:5]:
                st.error(f"Line {t.get('line')}: {t.get('loading_percent', 0):.1f}%")
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
            st.rerun()
        
        st.markdown('<div class="auto-progress"><strong>Status:</strong> Ready to start</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Interactive Mode</div>', unsafe_allow_html=True)
        
        plan = st.session_state.orchestrator.plan_control_actions(net, violations)
        actions = plan.get("actions", [])
        st.session_state.actions_in_plan = len(actions)
        
        if actions:
            st.markdown("**Proposed Actions:**")
            selected = []
            for i, action in enumerate(actions[:5]):
                if st.checkbox(f"{action.get('action_type', 'Unknown')} - Target: {action.get('target')}", key=f"act_{i}"):
                    selected.append(action)
            
            if st.button("✅ Execute Selected", use_container_width=True, type="primary"):
                for action in selected:
                    success, msg = execute_action(net, action)
                    if success:
                        st.session_state.actions_executed += 1
                        log_activity(msg, "Interactive")
                
                try:
                    net = st.session_state.solver.run(net)
                    st.session_state.net = net
                    st.session_state.violations = st.session_state.solver.detect_violations(net)
                except:
                    pass
                st.rerun()
        else:
            st.success("No actions needed")
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="section-card" style="text-align: center; padding: 60px;">
        <h2>Welcome to VoltAI</h2>
        <p>Select a network and click "Load Network & Analyze" to begin.</p>
    </div>
    """, unsafe_allow_html=True)
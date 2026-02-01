# ⚡ Faraday - AI-Powered Smart Grid Control System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![AI](https://img.shields.io/badge/AI-Ollama%20%2B%20Mistral-purple.svg)

**Next-Generation AI-Powered Power Grid Violation Analysis & Resolution**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Agents](#-ai-agents)

</div>

---

## 🎯 Overview

Faraday is an advanced AI-powered smart grid control system that uses multi-agent architecture with LLM reasoning to detect, analyze, and automatically resolve power grid violations. Built with pandapower for electrical network simulation and Streamlit for real-time visualization.

## 🌟 Features

### 🔍 **Real-Time Monitoring**
- Live power flow analysis with Newton-Raphson solver
- Voltage and thermal violation detection
- Network health scoring and visualization
- Interactive network topology with color-coded voltages

### 🤖 **AI-Powered Resolution**
- Multi-agent orchestration system
- LLM-based chain-of-thought reasoning (Mistral via Ollama)
- Automatic and interactive resolution modes
- Conflict resolution between competing actions

### 📊 **Visualization & Analytics**
- Before/After workflow comparison
- Side-by-side network state visualization
- Bus voltage comparison charts
- Thermal loading heatmaps
- Detailed metrics tabs (Overview/Voltage/Thermal)

### ⚡ **Control Actions**
- Generation redispatch
- Load curtailment/shedding
- Topology switching (open/close lines)
- Capacitor bank switching
- **Battery storage integration**

### 🔋 **Battery Storage**
- Add battery units at any bus
- Configurable capacity (1-10 MW)
- Charge/discharge scheduling
- Grid support during violations

## 🛠 Installation

### Prerequisites
- Python 3.10+
- Ollama with Mistral model

### Setup

```bash
# Clone the repository
git clone https://github.com/bhuvaneshchinthala/grid-agent-mcp.git
cd grid-agent-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in separate terminal)
ollama serve

# Pull Mistral model
ollama pull mistral
```

## 🚀 Usage

### Start the Dashboard

```bash
streamlit run app_enhanced.py
```

Visit `http://localhost:8501` in your browser.

### Quick Start

1. **Select Network**: Choose IEEE 33-Bus, 69-Bus, etc.
2. **Simulate Outage**: Optionally drop a line
3. **Load & Analyze**: Click to run power flow
4. **Choose Mode**:
   - 🤖 **Auto Mode**: Fully automated resolution
   - 👤 **Interactive Mode**: Review and approve actions

## 🏗 Architecture

```
grid-agent-mcp/
├── app_enhanced.py        # Main Streamlit dashboard
├── grid_agent/
│   ├── core/
│   │   ├── power_flow_solver.py   # Newton-Raphson solver
│   │   └── orchestrator.py        # Multi-agent coordinator
│   └── agents/
│       ├── base_agent.py          # LLM-powered base agent
│       ├── thermal_control_agent.py
│       ├── voltage_control_agent.py
│       ├── contingency_agent.py
│       ├── restoration_agent.py
│       └── battery_storage.py
└── config/
    ├── thresholds.yaml
    └── agent_config.yaml
```

## 🤖 AI Agents

| Agent | Role |
|-------|------|
| **Orchestrator** | Coordinates all agents, resolves conflicts |
| **Thermal Control** | Manages line overloads, generation redispatch |
| **Voltage Control** | Handles voltage violations, reactive power |
| **Contingency** | N-1 security analysis |
| **Restoration** | Grid restoration after outages |
| **Battery Storage** | Energy storage management |

## 📈 Supported Networks

- IEEE 4-Bus
- IEEE 14-Bus
- IEEE 30-Bus
- IEEE 33-Bus
- IEEE 69-Bus

## 🎨 Theme Support

Toggle between light and dark themes in the sidebar.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

---

<div align="center">
Made with ⚡ by the Faraday Team
</div>

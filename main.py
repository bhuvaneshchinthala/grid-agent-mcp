print("🔥 MAIN.PY FILE EXECUTED 🔥")

import pandapower.networks as pn
from grid_agent.core.orchestrator import Orchestrator

print("STEP A: Imports successful")

def main():
    print("STEP B: Inside main() function")

    print("Loading IEEE-30 grid...")
    net = pn.case30()
    print("Grid loaded")

    orchestrator = Orchestrator()
    print("Orchestrator created")

    print("Running orchestrator...")
    final_violations = orchestrator.run(net)

    print("FINAL VIOLATIONS:")
    print(final_violations)

if __name__ == "__main__":
    print("STEP C: __main__ block reached")
    main()

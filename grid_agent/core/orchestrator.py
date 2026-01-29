from grid_agent.agents.planner_agent import PlannerAgent
from grid_agent.agents.executor_agent import ExecutorAgent
from grid_agent.agents.validator_agent import ValidatorAgent
from grid_agent.core.power_flow_solver import PowerFlowSolver


class Orchestrator:
    def __init__(self):
        print("[Orchestrator] Initializing components...")
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.validator = ValidatorAgent()
        self.solver = PowerFlowSolver()

    def run(self, net):
        print("[Orchestrator] Running power flow...")
        net = self.solver.run(net)

        violations = self.solver.detect_violations(net)
        print("[Orchestrator] Initial violations:")
        print(violations)

        for step in range(1, 4):
            print(f"\n[Iteration {step}] Calling PlannerAgent (Mistral)...")
            actions = self.planner.plan(violations)
            print("[PlannerAgent] Suggested actions:")
            print(actions)

            print("[ExecutorAgent] Executing actions...")
            new_net = self.executor.execute(net, actions)

            new_violations = self.solver.detect_violations(new_net)
            print("[ValidatorAgent] New violations:")
            print(new_violations)

            if self.validator.is_better(violations, new_violations):
                print("✔ Improvement accepted")
                net = new_net
                violations = new_violations
            else:
                print("✖ No improvement, stopping")
                break

        return violations

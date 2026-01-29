import pandapower as pp

class PowerFlowSolver:
    def run(self, net):
        pp.runpp(net)
        return net

    def detect_violations(self, net):
        violations = {"voltage": [], "thermal": [], "disconnected": []}

        for bus in net.bus.index:
            v = net.res_bus.vm_pu[bus]
            if v < 0.95 or v > 1.05:
                violations["voltage"].append(int(bus))

        for line in net.line.index:
            if net.res_line.loading_percent[line] > 100:
                violations["thermal"].append(int(line))

        return violations

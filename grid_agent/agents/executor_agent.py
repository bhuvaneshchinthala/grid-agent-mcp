import pandapower as pp

class ExecutorAgent:
    def execute(self, net, actions):
        for action in actions:
            atype = action["action_type"]

            if atype == "curtail_load":
                bus = action["target"]
                factor = action["value"]

                loads = net.load[net.load.bus == bus].index
                for l in loads:
                    net.load.at[l, "p_mw"] *= factor

            elif atype == "reduce_generation":
                gen = action["target"]
                factor = action["value"]
                net.gen.at[gen, "p_mw"] *= factor

            elif atype == "switch_line":
                line = action["target"]
                state = action["value"]

                net.line.at[line, "in_service"] = (state == "close")

        return net

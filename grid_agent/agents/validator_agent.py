class ValidatorAgent:
    def is_better(self, old_v, new_v):
        return len(new_v["voltage"]) <= len(old_v["voltage"])

# possible base conditions: time and space; runtime conditions: hardware, communication (or other metrics), etc
# other sophisticated conditions (eg, val loss or other statistics) can be arg max of these
# account for partial backward

class generalized_optimizer(): # wraps torch.optim
    def __init__(self, config):
        super(generalized_optimizer, self).__init__(optimizer, condition=None) # optimizer is required 
        self.optimizer = optimizer
        self.parsed_condition = self._parse_condition(condition) # string of boolean cnf having both activate and deactivate boundaries

    def _parse_condition(condition):
        # split and assert condition validity
        parsed_condition = condition.split(' ')
        return parsed_condition

class generalized_optimizer_engine():
    def __init__(self, config):
        super(generalized_optimizer_engine, self).__init__(generalized_optimizers=[])
        self.generalized_optimizers = generalized_optimizers # torch or custom

    def check_condition(generalized_optimizer):
        for c in generalized_optimizer.parsed_condition[c]:
            if not exec(c):
                return False
        return True
        #if self.global_step > 30:
        #    return True

    def step():
        for o in self.generalized_optimizers:
            if self.check_condition(o):
                o.step()

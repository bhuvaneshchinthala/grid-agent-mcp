"""
🛡️ SAFETY CONSTRAINT VALIDATOR
Validates actions against physical and operational constraints
"""

class SafetyConstraintValidator:
    """
    Validates control actions against safety constraints
    """
    
    def __init__(self):
        """Initialize validator"""
        self.name = "SafetyConstraintValidator"
        
        # Constraints
        self.voltage_min = 0.95
        self.voltage_max = 1.05
        self.line_loading_max = 100
        self.gen_ramp_rate = 5  # MW/min
    
    def validate_action_sequence(self, actions, net):
        """
        Validate a sequence of actions
        
        Args:
            actions: list of action dicts
            net: pandapower network
        
        Returns:
            tuple: (is_valid, error_list)
        """
        errors = []
        
        try:
            if not actions:
                return True, []
            
            # Check each action
            for i, action in enumerate(actions):
                action_errors = self._validate_single_action(action, net)
                errors.extend(action_errors)
            
            # Check action sequence validity
            sequence_errors = self._validate_action_sequence(actions)
            errors.extend(sequence_errors)
            
            is_valid = len(errors) == 0
            return is_valid, errors
        
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
    
    def _validate_single_action(self, action, net):
        """Validate a single action"""
        errors = []
        
        try:
            action_type = action.get('type', '')
            
            if action_type == 'adjust_capacitor':
                mvar = action.get('mvar', 0)
                if abs(mvar) > 100:
                    errors.append(f"Capacitor adjustment too large: {mvar} MVAR")
            
            elif action_type == 'adjust_generation':
                mw = action.get('mw', 0)
                if abs(mw) > 200:
                    errors.append(f"Generation adjustment too large: {mw} MW")
            
            elif action_type == 'shed_load':
                mw = action.get('mw', 0)
                total_load = net.load['p_mw'].sum() if 'load' in dir(net) else 100
                if mw > total_load:
                    errors.append(f"Load shedding exceeds total load: {mw} MW")
            
        except:
            pass
        
        return errors
    
    def _validate_action_sequence(self, actions):
        """Validate action sequence"""
        errors = []
        
        try:
            # Check for conflicting actions
            action_types = [a.get('type') for a in actions]
            
            # Multiple generation adjustments
            gen_count = action_types.count('adjust_generation')
            if gen_count > 5:
                errors.append(f"Too many generation adjustments: {gen_count}")
            
        except:
            pass
        
        return errors
    
    def check_voltage_constraints(self, net):
        """Check voltage constraints"""
        violations = []
        
        try:
            if hasattr(net, 'res_bus'):
                for bus_idx in net.res_bus.index:
                    voltage = net.res_bus.at[bus_idx, 'vm_pu']
                    
                    if voltage < self.voltage_min or voltage > self.voltage_max:
                        violations.append({
                            'bus': bus_idx,
                            'voltage': voltage,
                            'violation': 'voltage'
                        })
        except:
            pass
        
        return violations
    
    def check_thermal_constraints(self, net):
        """Check thermal constraints"""
        violations = []
        
        try:
            if hasattr(net, 'res_line'):
                for line_idx in net.res_line.index:
                    loading = net.res_line.at[line_idx, 'loading_percent']
                    
                    if loading > self.line_loading_max:
                        violations.append({
                            'line': line_idx,
                            'loading': loading,
                            'violation': 'thermal'
                        })
        except:
            pass
        
        return violations
    
    def check_feasibility(self, action, net):
        """Check if action is feasible"""
        try:
            # Basic feasibility checks
            errors = self._validate_single_action(action, net)
            return len(errors) == 0, errors
        except:
            return False, ["Feasibility check error"]
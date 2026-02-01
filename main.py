"""
🔥 FARADAY ADVANCED - MAIN EXECUTION FILE
Complete power grid control system with AI agents
"""

print("=" * 80)
print("🔥 FARADAY ADVANCED - MAIN.PY FILE EXECUTED 🔥")
print("=" * 80)

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandapower as pp
import pandapower.networks as pn
from grid_agent.core.orchestrator import Orchestrator
from grid_agent.core.power_flow_solver import PowerFlowSolver
import json
from datetime import datetime

print("\n✓ STEP A: Imports successful")

def run_analysis(net, network_name="Network", stress_test=False):
    """
    Run complete grid analysis with AI orchestrator
    
    Args:
        net: pandapower network
        network_name: Name of network
        stress_test: Whether to apply stress (line outage)
    
    Returns:
        dict: Analysis results
    """
    print(f"\n{'=' * 80}")
    print(f"ANALYZING: {network_name}")
    print(f"{'=' * 80}")
    
    # Apply stress if requested
    if stress_test:
        print("\n⚠️  APPLYING STRESS: Opening lines 10 & 11")
        net.line.at[10, 'in_service'] = False
        net.line.at[11, 'in_service'] = False
    
    # STEP 1: Run initial power flow
    print("\n[1/7] Running initial power flow analysis...")
    solver = PowerFlowSolver()
    try:
        net = solver.run(net)
        print(f"✓ Power flow converged")
    except Exception as e:
        print(f"✗ Power flow failed: {e}")
        return None
    
    # STEP 2: Detect violations
    print("\n[2/7] Detecting violations...")
    try:
        violations = solver.detect_violations(net)
        num_voltage = len(violations.get('voltage', []))
        num_thermal = len(violations.get('thermal', []))
        print(f"✓ Violations detected: {num_voltage} voltage, {num_thermal} thermal")
        if num_voltage + num_thermal == 0:
            print("  → Grid is healthy!")
    except Exception as e:
        print(f"✗ Violation detection failed: {e}")
        return None
    
    # STEP 3: Calculate network health
    print("\n[3/7] Calculating network health...")
    try:
        total_violations = num_voltage + num_thermal
        max_buses = len(net.bus)
        max_lines = len(net.line)
        max_possible_violations = max_buses + max_lines
        
        health_score = max(0, 100 - (total_violations / max_possible_violations * 100))
        severity_score = total_violations * 5  # Simple severity calculation
        
        print(f"✓ Network Health: {health_score:.1f}%")
        print(f"✓ Severity Score: {severity_score:.1f}")
    except Exception as e:
        print(f"✗ Health calculation failed: {e}")
        health_score = 0
        severity_score = 100
    
    # STEP 4: Create orchestrator and plan control actions
    print("\n[4/7] Creating orchestrator and planning actions...")
    try:
        orchestrator = Orchestrator()
        print("✓ Orchestrator created")
        
        # Get agent plan
        control_plan = orchestrator.plan_control_actions(net, violations)
        print(f"✓ Control plan generated")
        print(f"  → Proposed actions: {len(control_plan.get('actions', []))}")
        
        for i, action in enumerate(control_plan.get('actions', [])[:3], 1):
            print(f"    {i}. {action.get('description', 'Action')}")
    except Exception as e:
        print(f"⚠️  Orchestrator warning: {e}")
        control_plan = {'actions': [], 'reasoning': str(e)}
    
    # STEP 5: Validate actions with safety constraints
    print("\n[5/7] Validating actions with safety constraints...")
    try:
        from grid_agent.agents.safety_constraint_validator import SafetyConstraintValidator
        validator = SafetyConstraintValidator()
        
        actions = control_plan.get('actions', [])
        is_valid, errors = validator.validate_action_sequence(actions, net)
        
        if is_valid:
            print(f"✓ All {len(actions)} actions passed safety validation")
        else:
            print(f"⚠️  {len(errors)} safety constraint(s) violated")
            for error in errors[:3]:
                print(f"  → {error}")
    except Exception as e:
        print(f"⚠️  Safety validation warning: {e}")
    
    # STEP 6: Get alarm recommendations
    print("\n[6/7] Generating intelligent alarms...")
    try:
        from grid_agent.agents.alarm_prioritizer import AlarmPrioritizer
        prioritizer = AlarmPrioritizer()
        
        alarms = prioritizer.generate_alarms(violations, {})
        critical_alarms = [a for a in alarms if a.get('severity') == 'CRITICAL']
        high_alarms = [a for a in alarms if a.get('severity') == 'HIGH']
        
        print(f"✓ Alarms generated: {len(critical_alarms)} CRITICAL, {len(high_alarms)} HIGH")
        
        if critical_alarms:
            print("\n  🔴 CRITICAL ALARMS:")
            for alarm in critical_alarms[:3]:
                print(f"    → {alarm.get('message', 'Critical issue')}")
    except Exception as e:
        print(f"⚠️  Alarm generation warning: {e}")
    
    # STEP 7: Make predictions
    print("\n[7/7] Making predictive forecasts...")
    try:
        from grid_agent.agents.violation_predictor import ViolationPredictor
        predictor = ViolationPredictor()
        predictor.add_observation(net, violations)
        
        predictions = predictor.predict_violations(net, steps_ahead=2)
        confidence = predictions.get('confidence', 0) * 100
        
        print(f"✓ Predictions made with {confidence:.1f}% confidence")
        print(f"  → Predicted voltage violations: {len(predictions.get('voltage_violations', []))}")
        print(f"  → Predicted thermal violations: {len(predictions.get('thermal_violations', []))}")
    except Exception as e:
        print(f"⚠️  Prediction warning: {e}")
        predictions = {}
    
    # Compile final results
    print(f"\n{'=' * 80}")
    print("ANALYSIS RESULTS")
    print(f"{'=' * 80}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'network': network_name,
        'network_stats': {
            'buses': len(net.bus),
            'lines': len(net.line),
            'generators': len(net.gen),
            'loads': len(net.load)
        },
        'health': {
            'network_health_percent': round(health_score, 2),
            'severity_score': round(severity_score, 2)
        },
        'violations': {
            'voltage_violations': num_voltage,
            'thermal_violations': num_thermal,
            'total': total_violations
        },
        'control_actions': {
            'proposed': len(control_plan.get('actions', [])),
            'actions': control_plan.get('actions', [])[:5]
        },
        'alarms': {
            'critical': len(critical_alarms) if 'critical_alarms' in locals() else 0,
            'high': len(high_alarms) if 'high_alarms' in locals() else 0
        },
        'predictions': {
            'confidence_percent': round(confidence, 2) if 'confidence' in locals() else 0,
            'voltage_violations_predicted': len(predictions.get('voltage_violations', [])),
            'thermal_violations_predicted': len(predictions.get('thermal_violations', []))
        }
    }
    
    print(f"\n✓ Network Health: {results['health']['network_health_percent']:.1f}%")
    print(f"✓ Violations: {results['violations']['total']} detected")
    print(f"✓ Control Actions: {results['control_actions']['proposed']} proposed")
    print(f"✓ Alarms: {results['alarms']['critical']} CRITICAL, {results['alarms']['high']} HIGH")
    print(f"✓ Predictions: {results['predictions']['confidence_percent']:.1f}% confidence")
    
    return results

def main():
    """Main execution function"""
    print("\n✓ STEP B: Inside main() function")
    
    results_summary = {
        'execution_time': datetime.now().isoformat(),
        'analyses': []
    }
    
    # Test 1: Normal Grid (IEEE-30)
    print("\n" + "=" * 80)
    print("TEST 1: NORMAL GRID (IEEE-30)")
    print("=" * 80)
    
    try:
        print("Loading IEEE-30 grid...")
        net1 = pn.case30()
        print(f"✓ Grid loaded: {len(net1.bus)} buses, {len(net1.line)} lines")
        
        results1 = run_analysis(net1, "IEEE-30 Normal Grid", stress_test=False)
        if results1:
            results_summary['analyses'].append(results1)
            print("\n✓ TEST 1 COMPLETED SUCCESSFULLY")
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
    
    # Test 2: Stressed Grid (IEEE-30 with outage)
    print("\n" + "=" * 80)
    print("TEST 2: STRESSED GRID (IEEE-30 with Line Outage)")
    print("=" * 80)
    
    try:
        print("Loading IEEE-30 grid with stress...")
        net2 = pn.case30()
        print(f"✓ Grid loaded: {len(net2.bus)} buses, {len(net2.line)} lines")
        
        results2 = run_analysis(net2, "IEEE-30 Stressed Grid", stress_test=True)
        if results2:
            results_summary['analyses'].append(results2)
            print("\n✓ TEST 2 COMPLETED SUCCESSFULLY")
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
    
    # Test 3: Distribution Network (IEEE-33)
    print("\n" + "=" * 80)
    print("TEST 3: DISTRIBUTION NETWORK (IEEE-33)")
    print("=" * 80)
    
    try:
        print("Loading IEEE-33 grid...")
        net3 = pn.case33bw()
        print(f"✓ Grid loaded: {len(net3.bus)} buses, {len(net3.line)} lines")
        
        results3 = run_analysis(net3, "IEEE-33 Distribution Network", stress_test=False)
        if results3:
            results_summary['analyses'].append(results3)
            print("\n✓ TEST 3 COMPLETED SUCCESSFULLY")
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
    
    # Summary Report
    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    
    print(f"\nTotal Analyses Run: {len(results_summary['analyses'])}")
    
    for i, analysis in enumerate(results_summary['analyses'], 1):
        print(f"\n[Analysis {i}] {analysis['network']}")
        print(f"  Network Health: {analysis['health']['network_health_percent']:.1f}%")
        print(f"  Violations: {analysis['violations']['total']}")
        print(f"  Actions Proposed: {analysis['control_actions']['proposed']}")
        print(f"  Prediction Confidence: {analysis['predictions']['confidence_percent']:.1f}%")
    
    # Save results to JSON
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    try:
        output_file = Path("analysis_results.json")
        with open(output_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"✓ Results saved to: {output_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    print("\n" + "=" * 80)
    print("✅ MAIN.PY EXECUTION COMPLETE")
    print("=" * 80)
    print("\nTo view results in dashboard, run:")
    print("  streamlit run app_enhanced.py")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print("\n✓ STEP C: __main__ block reached")
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
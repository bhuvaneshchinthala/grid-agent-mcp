import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings

# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class BusType(Enum):
    """Bus type enumeration"""
    SLACK = 1   # Reference bus (V and δ fixed)
    PV = 2      # Generator bus (P and V specified)
    PQ = 3      # Load bus (P and Q specified)

# >>> ADDED: Status tracking for PV/PQ switching
class BusStatus(Enum):
    """Tracks original vs current bus type for restoration logic"""
    ORIGINAL = 1
    SWITCHED = 2

@dataclass
class Bus:
    """Bus data container"""
    bus_id: int
    bus_type: BusType
    Pd: float = 0.0      # Active power demand (MW)
    Qd: float = 0.0      # Reactive power demand (MVAr)
    Gs: float = 0.0      # Shunt conductance (MW at V=1.0)
    Bs: float = 0.0      # Shunt susceptance (MVAr at V=1.0)
    Vm: float = 1.0      # Voltage magnitude (p.u.)
    Va: float = 0.0      # Voltage angle (radians)
    baseKV: float = 132.0
    # >>> ADDED: Fields for Q-limit switching
    original_type: BusType = None
    status: BusStatus = BusStatus.ORIGINAL
    
    def __post_init__(self):
        # >>> ADDED: Store original type for restoration
        if self.original_type is None:
            self.original_type = self.bus_type

@dataclass
class Generator:
    """Generator data container"""
    bus_id: int
    Pg: float            # Active power generation (MW)
    Qg: float = 0.0      # Reactive power generation (MVAr)
    Qmax: float = 999.0  # Maximum reactive power (MVAr)
    Qmin: float = -999.0 # Minimum reactive power (MVAr)
    Vg: float = 1.0      # Voltage setpoint (p.u.)
    status: int = 1      # 1 = in service
    # >>> ADDED: Track Q limit violations
    at_Qmax: bool = False
    at_Qmin: bool = False

@dataclass
class Branch:
    """Branch (transmission line/transformer) data container"""
    from_bus: int
    to_bus: int
    r: float             # Resistance (p.u.)
    x: float             # Reactance (p.u.)
    b: float = 0.0       # Total line charging susceptance (p.u.)
    rateA: float = 999.0 # MVA rating
    ratio: float = 1.0   # Transformer tap ratio (0 or 1 for lines)
    angle: float = 0.0   # Phase shift angle (degrees)
    status: int = 1      # 1 = in service

# =============================================================================
# >>> ADDED: CONVERGENCE LOGGER
# =============================================================================

@dataclass
class ConvergenceLog:
    """Stores convergence history for analysis"""
    iteration: int
    P_mismatch_inf: float
    Q_mismatch_inf: float
    P_mismatch_2: float
    Q_mismatch_2: float
    max_dV: float
    max_ddelta: float
    damping: float
    pv_count: int
    pq_count: int
    switched_buses: List[int] = field(default_factory=list)

# =============================================================================
# MAIN SOLVER CLASS
# =============================================================================

class NewtonRaphsonPowerFlow:
    """
    Industrial-grade Newton-Raphson Power Flow Solver
    Implements all features comparable to pandapower
    """
    
    def __init__(self, 
                 buses: List[Bus], 
                 generators: List[Generator], 
                 branches: List[Branch],
                 baseMVA: float = 100.0):
        
        self.buses = {b.bus_id: b for b in buses}
        self.generators = generators
        self.branches = branches
        self.baseMVA = baseMVA
        self.n_bus = len(buses)
        
        # Create bus ID to index mapping
        self.bus_ids = sorted(self.buses.keys())
        self.bus_id_to_idx = {bid: idx for idx, bid in enumerate(self.bus_ids)}
        self.idx_to_bus_id = {idx: bid for bid, idx in self.bus_id_to_idx.items()}
        
        # Build admittance matrix
        self.Ybus = self._build_ybus()
        
        # >>> ADDED: Generator lookup by bus
        self.gen_at_bus: Dict[int, Generator] = {}
        for gen in self.generators:
            if gen.status == 1:
                self.gen_at_bus[gen.bus_id] = gen
        
        # >>> ADDED: Convergence history
        self.convergence_history: List[ConvergenceLog] = []
        
        # >>> ADDED: Current bus types (can change during solution)
        self.current_bus_types: Dict[int, BusType] = {
            b.bus_id: b.bus_type for b in buses
        }
        
    def _build_ybus(self) -> np.ndarray:
        """Build the bus admittance matrix (Y-bus)"""
        n = self.n_bus
        Ybus = np.zeros((n, n), dtype=complex)
        
        for branch in self.branches:
            if branch.status == 0:
                continue
                
            i = self.bus_id_to_idx[branch.from_bus]
            j = self.bus_id_to_idx[branch.to_bus]
            
            # Series admittance
            if branch.x == 0 and branch.r == 0:
                # >>> ADDED: Handle zero impedance branches
                y_series = complex(1e5, 0)  # Very large admittance
            else:
                z_series = complex(branch.r, branch.x)
                y_series = 1.0 / z_series
            
            # Shunt admittance (line charging)
            y_shunt = complex(0, branch.b / 2.0)
            
            # >>> MODIFIED: Proper transformer modeling
            if branch.ratio != 0 and branch.ratio != 1.0:
                tap = branch.ratio
                shift = np.radians(branch.angle)
                tap_complex = tap * np.exp(1j * shift)
                
                # Pi-equivalent for transformer
                Ybus[i, i] += y_series / (tap ** 2) + y_shunt
                Ybus[j, j] += y_series + y_shunt
                Ybus[i, j] -= y_series / np.conj(tap_complex)
                Ybus[j, i] -= y_series / tap_complex
            else:
                # Standard transmission line
                Ybus[i, i] += y_series + y_shunt
                Ybus[j, j] += y_series + y_shunt
                Ybus[i, j] -= y_series
                Ybus[j, i] -= y_series
        
        # Add shunt elements at buses
        for bus_id, bus in self.buses.items():
            idx = self.bus_id_to_idx[bus_id]
            # Gs and Bs are in MW/MVAr at V=1.0, convert to p.u.
            Ybus[idx, idx] += complex(bus.Gs, bus.Bs) / self.baseMVA
            
        return Ybus
    
    # >>> ADDED: Complete method for calculating bus power injections
    def _calculate_power_injection(self, V: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate P and Q injections at all buses
        
        P_i = V_i * Σ_j V_j * (G_ij * cos(δ_i - δ_j) + B_ij * sin(δ_i - δ_j))
        Q_i = V_i * Σ_j V_j * (G_ij * sin(δ_i - δ_j) - B_ij * cos(δ_i - δ_j))
        """
        n = self.n_bus
        P = np.zeros(n)
        Q = np.zeros(n)
        
        G = self.Ybus.real
        B = self.Ybus.imag
        
        for i in range(n):
            for j in range(n):
                angle_diff = delta[i] - delta[j]
                P[i] += V[i] * V[j] * (G[i, j] * np.cos(angle_diff) + 
                                        B[i, j] * np.sin(angle_diff))
                Q[i] += V[i] * V[j] * (G[i, j] * np.sin(angle_diff) - 
                                        B[i, j] * np.cos(angle_diff))
        
        return P, Q
    
    # >>> ADDED: Method to get specified power at each bus
    def _get_specified_power(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get specified P and Q at each bus (generation - load)"""
        n = self.n_bus
        P_spec = np.zeros(n)
        Q_spec = np.zeros(n)
        
        # Subtract loads
        for bus_id, bus in self.buses.items():
            idx = self.bus_id_to_idx[bus_id]
            P_spec[idx] = -bus.Pd / self.baseMVA  # Convert to p.u.
            Q_spec[idx] = -bus.Qd / self.baseMVA
        
        # Add generation
        for gen in self.generators:
            if gen.status == 1:
                idx = self.bus_id_to_idx[gen.bus_id]
                P_spec[idx] += gen.Pg / self.baseMVA
                Q_spec[idx] += gen.Qg / self.baseMVA
        
        return P_spec, Q_spec
    
    # >>> ADDED: Build indices for reduced Jacobian
    def _build_index_sets(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Build index sets for Jacobian construction
        
        Returns:
            pv_indices: Indices of PV buses (for δ equations)
            pq_indices: Indices of PQ buses (for δ and V equations)
            non_slack_indices: All non-slack bus indices
        """
        slack_indices = []
        pv_indices = []
        pq_indices = []
        
        for bus_id in self.bus_ids:
            idx = self.bus_id_to_idx[bus_id]
            bus_type = self.current_bus_types[bus_id]
            
            if bus_type == BusType.SLACK:
                slack_indices.append(idx)
            elif bus_type == BusType.PV:
                pv_indices.append(idx)
            else:  # PQ
                pq_indices.append(idx)
        
        non_slack_indices = pv_indices + pq_indices
        
        return pv_indices, pq_indices, non_slack_indices
    
    # >>> ADDED: Complete Jacobian calculation with correct formulation
    def _build_jacobian(self, V: np.ndarray, delta: np.ndarray, 
                        P_calc: np.ndarray, Q_calc: np.ndarray) -> np.ndarray:
        """
        Build the reduced Jacobian matrix for Newton-Raphson
        
        J = | J1  J2 |   where J1 = ∂P/∂δ,  J2 = ∂P/∂|V|
            | J3  J4 |         J3 = ∂Q/∂δ,  J4 = ∂Q/∂|V|
        
        Dimensions:
            J1: (n_pv + n_pq) × (n_pv + n_pq)
            J2: (n_pv + n_pq) × n_pq
            J3: n_pq × (n_pv + n_pq)
            J4: n_pq × n_pq
        """
        n = self.n_bus
        G = self.Ybus.real
        B = self.Ybus.imag
        
        # Get index sets
        pv_idx, pq_idx, non_slack_idx = self._build_index_sets()
        
        n_non_slack = len(non_slack_idx)
        n_pq = len(pq_idx)
        
        # Full Jacobian submatrices
        J1_full = np.zeros((n, n))  # ∂P/∂δ
        J2_full = np.zeros((n, n))  # ∂P/∂|V|
        J3_full = np.zeros((n, n))  # ∂Q/∂δ
        J4_full = np.zeros((n, n))  # ∂Q/∂|V|
        
        for i in range(n):
            for j in range(n):
                angle_diff = delta[i] - delta[j]
                
                if i == j:
                    # Diagonal elements
                    # J1[i,i] = -Q_i - B_ii * V_i^2
                    J1_full[i, i] = -Q_calc[i] - B[i, i] * V[i] ** 2
                    
                    # J2[i,i] = P_i/V_i + G_ii * V_i
                    J2_full[i, i] = P_calc[i] / V[i] + G[i, i] * V[i]
                    
                    # J3[i,i] = P_i - G_ii * V_i^2
                    J3_full[i, i] = P_calc[i] - G[i, i] * V[i] ** 2
                    
                    # J4[i,i] = Q_i/V_i - B_ii * V_i
                    J4_full[i, i] = Q_calc[i] / V[i] - B[i, i] * V[i]
                else:
                    # Off-diagonal elements
                    # J1[i,j] = V_i * V_j * (G_ij * sin(δ_i - δ_j) - B_ij * cos(δ_i - δ_j))
                    J1_full[i, j] = V[i] * V[j] * (G[i, j] * np.sin(angle_diff) - 
                                                    B[i, j] * np.cos(angle_diff))
                    
                    # J2[i,j] = V_i * (G_ij * cos(δ_i - δ_j) + B_ij * sin(δ_i - δ_j))
                    J2_full[i, j] = V[i] * (G[i, j] * np.cos(angle_diff) + 
                                            B[i, j] * np.sin(angle_diff))
                    
                    # J3[i,j] = -V_i * V_j * (G_ij * cos(δ_i - δ_j) + B_ij * sin(δ_i - δ_j))
                    J3_full[i, j] = -V[i] * V[j] * (G[i, j] * np.cos(angle_diff) + 
                                                     B[i, j] * np.sin(angle_diff))
                    
                    # J4[i,j] = V_i * (G_ij * sin(δ_i - δ_j) - B_ij * cos(δ_i - δ_j))
                    J4_full[i, j] = V[i] * (G[i, j] * np.sin(angle_diff) - 
                                            B[i, j] * np.cos(angle_diff))
        
        # >>> ADDED: Build reduced Jacobian by removing slack bus
        # State vector is [Δδ_non_slack, Δ|V|_pq]
        # Mismatch vector is [ΔP_non_slack, ΔQ_pq]
        
        jacobian_size = n_non_slack + n_pq
        J = np.zeros((jacobian_size, jacobian_size))
        
        # Create index mappings for reduced matrix
        non_slack_to_reduced = {idx: i for i, idx in enumerate(non_slack_idx)}
        pq_to_reduced_v = {idx: i for i, idx in enumerate(pq_idx)}
        
        # Fill J1 block: ∂P/∂δ for non-slack buses
        for i, bus_i in enumerate(non_slack_idx):
            for j, bus_j in enumerate(non_slack_idx):
                J[i, j] = J1_full[bus_i, bus_j]
        
        # Fill J2 block: ∂P/∂|V| for non-slack P, PQ voltages
        for i, bus_i in enumerate(non_slack_idx):
            for j, bus_j in enumerate(pq_idx):
                J[i, n_non_slack + j] = J2_full[bus_i, bus_j]
        
        # Fill J3 block: ∂Q/∂δ for PQ buses
        for i, bus_i in enumerate(pq_idx):
            for j, bus_j in enumerate(non_slack_idx):
                J[n_non_slack + i, j] = J3_full[bus_i, bus_j]
        
        # Fill J4 block: ∂Q/∂|V| for PQ buses
        for i, bus_i in enumerate(pq_idx):
            for j, bus_j in enumerate(pq_idx):
                J[n_non_slack + i, n_non_slack + j] = J4_full[bus_i, bus_j]
        
        return J
    
    # >>> ADDED: PV to PQ switching logic with Q-limit enforcement
    def _enforce_q_limits(self, V: np.ndarray, delta: np.ndarray, 
                          Q_calc: np.ndarray) -> Tuple[bool, List[int]]:
        """
        Check Q limits at PV buses and switch to PQ if violated
        
        Returns:
            switched: True if any bus type changed
            switched_buses: List of bus IDs that switched
        """
        switched = False
        switched_buses = []
        
        for gen in self.generators:
            if gen.status == 0:
                continue
                
            bus_id = gen.bus_id
            bus = self.buses[bus_id]
            idx = self.bus_id_to_idx[bus_id]
            
            # Only check buses currently operating as PV
            if self.current_bus_types[bus_id] != BusType.PV:
                continue
            
            # Calculate generator Q output
            # Q_gen = Q_calc + Q_load (since Q_calc = Q_gen - Q_load)
            Q_gen = Q_calc[idx] * self.baseMVA + bus.Qd
            
            # Check Q limits
            if Q_gen > gen.Qmax:
                # Hit upper limit - switch to PQ with Q at Qmax
                print(f"  >>> Bus {bus_id}: Q_gen={Q_gen:.2f} > Qmax={gen.Qmax:.2f} → Switching PV→PQ")
                self.current_bus_types[bus_id] = BusType.PQ
                gen.Qg = gen.Qmax
                gen.at_Qmax = True
                gen.at_Qmin = False
                bus.status = BusStatus.SWITCHED
                switched = True
                switched_buses.append(bus_id)
                
            elif Q_gen < gen.Qmin:
                # Hit lower limit - switch to PQ with Q at Qmin
                print(f"  >>> Bus {bus_id}: Q_gen={Q_gen:.2f} < Qmin={gen.Qmin:.2f} → Switching PV→PQ")
                self.current_bus_types[bus_id] = BusType.PQ
                gen.Qg = gen.Qmin
                gen.at_Qmax = False
                gen.at_Qmin = True
                bus.status = BusStatus.SWITCHED
                switched = True
                switched_buses.append(bus_id)
        
        return switched, switched_buses
    
    # >>> ADDED: Check if switched PQ buses can return to PV
    def _check_pv_restoration(self, V: np.ndarray) -> Tuple[bool, List[int]]:
        """
        Check if any switched PQ bus can be restored to PV
        (voltage returns to setpoint with Q within limits)
        """
        restored = False
        restored_buses = []
        
        for gen in self.generators:
            if gen.status == 0:
                continue
                
            bus_id = gen.bus_id
            bus = self.buses[bus_id]
            
            # Only check switched buses
            if bus.status != BusStatus.SWITCHED:
                continue
            if bus.original_type != BusType.PV:
                continue
            if self.current_bus_types[bus_id] != BusType.PQ:
                continue
            
            idx = self.bus_id_to_idx[bus_id]
            V_current = V[idx]
            V_setpoint = gen.Vg
            
            # Check if voltage is trying to exceed setpoint (at Qmax)
            # or go below setpoint (at Qmin)
            if gen.at_Qmax and V_current > V_setpoint:
                print(f"  >>> Bus {bus_id}: V={V_current:.4f} > Vset={V_setpoint:.4f} → Restoring PQ→PV")
                self.current_bus_types[bus_id] = BusType.PV
                bus.Vm = V_setpoint
                V[idx] = V_setpoint
                gen.at_Qmax = False
                bus.status = BusStatus.ORIGINAL
                restored = True
                restored_buses.append(bus_id)
                
            elif gen.at_Qmin and V_current < V_setpoint:
                print(f"  >>> Bus {bus_id}: V={V_current:.4f} < Vset={V_setpoint:.4f} → Restoring PQ→PV")
                self.current_bus_types[bus_id] = BusType.PV
                bus.Vm = V_setpoint
                V[idx] = V_setpoint
                gen.at_Qmin = False
                bus.status = BusStatus.ORIGINAL
                restored = True
                restored_buses.append(bus_id)
        
        return restored, restored_buses
    
    # >>> ADDED: Adaptive damping factor calculation
    def _calculate_damping(self, iteration: int, mismatch_norm: float, 
                           prev_mismatch_norm: float, current_damping: float) -> float:
        """
        Calculate adaptive damping factor
        
        - Start conservative (α = 0.4)
        - Increase if converging well
        - Decrease if diverging
        """
        min_damping = 0.1
        max_damping = 1.0
        
        if iteration == 0:
            return 0.4  # Initial conservative damping
        
        if mismatch_norm < prev_mismatch_norm:
            # Converging - can increase damping
            new_damping = min(current_damping * 1.2, max_damping)
        else:
            # Diverging or stagnating - reduce damping
            new_damping = max(current_damping * 0.5, min_damping)
        
        return new_damping
    
    # >>> ADDED: Numerical regularization for singular Jacobian
    def _regularize_jacobian(self, J: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        Add small diagonal regularization to prevent singularity
        """
        n = J.shape[0]
        
        # Check condition number
        try:
            cond = np.linalg.cond(J)
            if cond > 1e12:
                print(f"  Warning: Ill-conditioned Jacobian (cond={cond:.2e}), adding regularization")
                J = J + epsilon * np.eye(n)
        except:
            J = J + epsilon * np.eye(n)
        
        return J
    
    # >>> MODIFIED: Main solve method with all industrial features
    def solve(self, 
              tol: float = 1e-6,
              max_iter: int = 50,
              enforce_q_limits: bool = True,
              initial_damping: float = 0.4,
              verbose: bool = True) -> Dict:
        """
        Solve power flow using Newton-Raphson method
        
        Args:
            tol: Convergence tolerance
            max_iter: Maximum iterations
            enforce_q_limits: Enable PV→PQ switching
            initial_damping: Starting damping factor
            verbose: Print iteration details
        
        Returns:
            Dictionary with solution results
        """
        print("="*70)
        print("NEWTON-RAPHSON POWER FLOW SOLVER")
        print("="*70)
        
        # Initialize state variables
        n = self.n_bus
        V = np.ones(n)      # Voltage magnitudes
        delta = np.zeros(n)  # Voltage angles
        
        # Set initial voltages from bus data and generator setpoints
        for bus_id, bus in self.buses.items():
            idx = self.bus_id_to_idx[bus_id]
            V[idx] = bus.Vm
            delta[idx] = bus.Va
        
        # Override with generator voltage setpoints
        for gen in self.generators:
            if gen.status == 1 and self.current_bus_types[gen.bus_id] == BusType.PV:
                idx = self.bus_id_to_idx[gen.bus_id]
                V[idx] = gen.Vg
        
        # Get specified power injections
        P_spec, Q_spec = self._get_specified_power()
        
        # Initialize tracking variables
        self.convergence_history = []
        damping = initial_damping
        prev_mismatch_norm = np.inf
        converged = False
        
        # >>> ADDED: Get slack bus index
        slack_idx = None
        for bus_id, bus_type in self.current_bus_types.items():
            if bus_type == BusType.SLACK:
                slack_idx = self.bus_id_to_idx[bus_id]
                break
        
        if verbose:
            print(f"\nSystem: {n} buses, BaseMVA: {self.baseMVA}")
            print(f"Tolerance: {tol}, Max iterations: {max_iter}")
            print(f"Initial damping: {initial_damping}")
            print("-"*70)
        
        # =====================================================================
        # MAIN ITERATION LOOP
        # =====================================================================
        for iteration in range(max_iter):
            
            # -----------------------------------------------------------------
            # Step 1: Calculate power injections
            # -----------------------------------------------------------------
            P_calc, Q_calc = self._calculate_power_injection(V, delta)
            
            # -----------------------------------------------------------------
            # Step 2: Calculate mismatches (ΔP = P_spec - P_calc)
            # -----------------------------------------------------------------
            dP = P_spec - P_calc
            dQ = Q_spec - Q_calc
            
            # -----------------------------------------------------------------
            # >>> ADDED: Step 3: Enforce Q limits (PV→PQ switching)
            # -----------------------------------------------------------------
            if enforce_q_limits and iteration > 0:
                switched, switched_buses = self._enforce_q_limits(V, delta, Q_calc)
                if switched:
                    # Update Q_spec for switched buses
                    P_spec, Q_spec = self._get_specified_power()
                    dQ = Q_spec - Q_calc
                
                # Check for PQ→PV restoration
                restored, restored_buses = self._check_pv_restoration(V)
                if restored:
                    P_spec, Q_spec = self._get_specified_power()
            
            # -----------------------------------------------------------------
            # Step 4: Get current index sets
            # -----------------------------------------------------------------
            pv_idx, pq_idx, non_slack_idx = self._build_index_sets()
            
            # -----------------------------------------------------------------
            # Step 5: Build mismatch vector (excluding slack bus)
            # -----------------------------------------------------------------
            # Mismatch vector: [ΔP for non-slack buses, ΔQ for PQ buses]
            mismatch_P = dP[non_slack_idx]
            mismatch_Q = dQ[pq_idx]
            mismatch = np.concatenate([mismatch_P, mismatch_Q])
            
            # -----------------------------------------------------------------
            # >>> ADDED: Step 6: Calculate convergence metrics
            # -----------------------------------------------------------------
            P_inf_norm = np.max(np.abs(mismatch_P)) if len(mismatch_P) > 0 else 0
            Q_inf_norm = np.max(np.abs(mismatch_Q)) if len(mismatch_Q) > 0 else 0
            P_2_norm = np.linalg.norm(mismatch_P)
            Q_2_norm = np.linalg.norm(mismatch_Q)
            total_inf_norm = max(P_inf_norm, Q_inf_norm)
            total_2_norm = np.linalg.norm(mismatch)
            
            # -----------------------------------------------------------------
            # >>> ADDED: Step 7: Adaptive damping
            # -----------------------------------------------------------------
            damping = self._calculate_damping(iteration, total_2_norm, 
                                              prev_mismatch_norm, damping)
            prev_mismatch_norm = total_2_norm
            
            # Log convergence
            n_pv = len(pv_idx)
            n_pq = len(pq_idx)
            
            if verbose:
                print(f"Iter {iteration:2d}: |ΔP|∞={P_inf_norm:.2e}, |ΔQ|∞={Q_inf_norm:.2e}, "
                      f"||mismatch||₂={total_2_norm:.2e}, α={damping:.3f}, "
                      f"PV={n_pv}, PQ={n_pq}")
            
            # -----------------------------------------------------------------
            # >>> MODIFIED: Step 8: Check convergence (both norms)
            # -----------------------------------------------------------------
            if total_inf_norm < tol and total_2_norm < tol * np.sqrt(len(mismatch) + 1):
                converged = True
                print("-"*70)
                print(f"✓ CONVERGED in {iteration} iterations")
                print(f"  Final |ΔP|∞ = {P_inf_norm:.2e}")
                print(f"  Final |ΔQ|∞ = {Q_inf_norm:.2e}")
                print(f"  Final ||mismatch||₂ = {total_2_norm:.2e}")
                break
            
            # -----------------------------------------------------------------
            # Step 9: Build Jacobian matrix
            # -----------------------------------------------------------------
            J = self._build_jacobian(V, delta, P_calc, Q_calc)
            
            # -----------------------------------------------------------------
            # >>> ADDED: Step 10: Regularize Jacobian if needed
            # -----------------------------------------------------------------
            J = self._regularize_jacobian(J)
            
            # -----------------------------------------------------------------
            # Step 11: Solve linear system J * Δx = mismatch
            # -----------------------------------------------------------------
            try:
                dx = np.linalg.solve(J, mismatch)
            except np.linalg.LinAlgError:
                print("  Error: Singular Jacobian matrix!")
                # Try pseudo-inverse as fallback
                dx = np.linalg.lstsq(J, mismatch, rcond=None)[0]
            
            # -----------------------------------------------------------------
            # >>> ADDED: Step 12: Apply damping to update
            # -----------------------------------------------------------------
            dx_damped = damping * dx
            
            # Extract angle and voltage updates
            n_non_slack = len(non_slack_idx)
            d_delta = dx_damped[:n_non_slack]
            d_V = dx_damped[n_non_slack:]
            
            # -----------------------------------------------------------------
            # Step 13: Update state variables
            # -----------------------------------------------------------------
            # Update angles for non-slack buses
            for i, bus_idx in enumerate(non_slack_idx):
                delta[bus_idx] += d_delta[i]
            
            # Update voltage magnitudes for PQ buses only
            for i, bus_idx in enumerate(pq_idx):
                V[bus_idx] += d_V[i]
                # >>> ADDED: Voltage magnitude limits
                V[bus_idx] = np.clip(V[bus_idx], 0.5, 1.5)
            
            # -----------------------------------------------------------------
            # >>> ADDED: Log iteration data
            # -----------------------------------------------------------------
            log_entry = ConvergenceLog(
                iteration=iteration,
                P_mismatch_inf=P_inf_norm,
                Q_mismatch_inf=Q_inf_norm,
                P_mismatch_2=P_2_norm,
                Q_mismatch_2=Q_2_norm,
                max_dV=np.max(np.abs(d_V)) if len(d_V) > 0 else 0,
                max_ddelta=np.max(np.abs(d_delta)) if len(d_delta) > 0 else 0,
                damping=damping,
                pv_count=n_pv,
                pq_count=n_pq
            )
            self.convergence_history.append(log_entry)
        
        # =====================================================================
        # POST-PROCESSING
        # =====================================================================
        if not converged:
            print("-"*70)
            print(f"✗ DID NOT CONVERGE after {max_iter} iterations")
            print(f"  Final |ΔP|∞ = {P_inf_norm:.2e}")
            print(f"  Final |ΔQ|∞ = {Q_inf_norm:.2e}")
        
        # Store final values back to bus objects
        for bus_id, bus in self.buses.items():
            idx = self.bus_id_to_idx[bus_id]
            bus.Vm = V[idx]
            bus.Va = delta[idx]
        
        # Calculate final power flows and generator outputs
        P_final, Q_final = self._calculate_power_injection(V, delta)
        
        # Update generator Q outputs
        for gen in self.generators:
            if gen.status == 1:
                idx = self.bus_id_to_idx[gen.bus_id]
                bus = self.buses[gen.bus_id]
                gen.Qg = (Q_final[idx] * self.baseMVA) + bus.Qd
        
        # Calculate slack bus power
        if slack_idx is not None:
            slack_bus_id = self.idx_to_bus_id[slack_idx]
            slack_P = P_final[slack_idx] * self.baseMVA + self.buses[slack_bus_id].Pd
            slack_Q = Q_final[slack_idx] * self.baseMVA + self.buses[slack_bus_id].Qd
        else:
            slack_P, slack_Q = 0, 0
        
        # Compile results
        results = {
            'converged': converged,
            'iterations': iteration + 1 if not converged else iteration,
            'V': V.copy(),
            'delta': delta.copy(),
            'delta_deg': np.degrees(delta),
            'P_gen': P_final * self.baseMVA,
            'Q_gen': Q_final * self.baseMVA,
            'slack_P': slack_P,
            'slack_Q': slack_Q,
            'convergence_history': self.convergence_history
        }
        
        return results
    
    # >>> ADDED: Results printing method
    def print_results(self, results: Dict):
        """Print formatted power flow results"""
        print("\n" + "="*70)
        print("POWER FLOW RESULTS")
        print("="*70)
        
        print(f"\n{'Bus':>4} {'Type':>6} {'|V| (pu)':>10} {'δ (deg)':>10} "
              f"{'P (MW)':>10} {'Q (MVAr)':>10}")
        print("-"*60)
        
        V = results['V']
        delta_deg = results['delta_deg']
        
        for bus_id in sorted(self.buses.keys()):
            bus = self.buses[bus_id]
            idx = self.bus_id_to_idx[bus_id]
            
            bus_type = self.current_bus_types[bus_id].name
            
            # Net injection at bus
            P_net = results['P_gen'][idx]
            Q_net = results['Q_gen'][idx]
            
            print(f"{bus_id:4d} {bus_type:>6} {V[idx]:10.4f} {delta_deg[idx]:10.2f} "
                  f"{P_net:10.2f} {Q_net:10.2f}")
        
        print("-"*60)
        print(f"\nSlack Bus Generation: P = {results['slack_P']:.2f} MW, "
              f"Q = {results['slack_Q']:.2f} MVAr")
        
        # Print generator summary
        print(f"\n{'Generator Summary':^60}")
        print("-"*60)
        print(f"{'Bus':>4} {'Pg (MW)':>10} {'Qg (MVAr)':>10} "
              f"{'Qmin':>8} {'Qmax':>8} {'Status':>10}")
        print("-"*60)
        
        for gen in self.generators:
            if gen.status == 1:
                status = "At Qmax" if gen.at_Qmax else ("At Qmin" if gen.at_Qmin else "Normal")
                print(f"{gen.bus_id:4d} {gen.Pg:10.2f} {gen.Qg:10.2f} "
                      f"{gen.Qmin:8.1f} {gen.Qmax:8.1f} {status:>10}")


# =============================================================================
# IEEE 30-BUS SYSTEM DATA
# =============================================================================

def create_ieee30_system() -> NewtonRaphsonPowerFlow:
    """Create IEEE 30-bus test system"""
    
    # Bus data: bus_id, type, Pd (MW), Qd (MVAr), Gs, Bs, Vm, Va
    buses = [
        Bus(1,  BusType.SLACK, 0.0,    0.0,    0, 0, 1.060, 0.0),
        Bus(2,  BusType.PV,    21.7,   12.7,   0, 0, 1.043, 0.0),
        Bus(3,  BusType.PQ,    2.4,    1.2,    0, 0, 1.0,   0.0),
        Bus(4,  BusType.PQ,    7.6,    1.6,    0, 0, 1.0,   0.0),
        Bus(5,  BusType.PV,    94.2,   19.0,   0, 0, 1.010, 0.0),
        Bus(6,  BusType.PQ,    0.0,    0.0,    0, 0, 1.0,   0.0),
        Bus(7,  BusType.PQ,    22.8,   10.9,   0, 0, 1.0,   0.0),
        Bus(8,  BusType.PV,    30.0,   30.0,   0, 0, 1.010, 0.0),
        Bus(9,  BusType.PQ,    0.0,    0.0,    0, 0, 1.0,   0.0),
        Bus(10, BusType.PQ,    5.8,    2.0,    0, 0, 1.0,   0.0),
        Bus(11, BusType.PV,    0.0,    0.0,    0, 0, 1.082, 0.0),
        Bus(12, BusType.PQ,    11.2,   7.5,    0, 0, 1.0,   0.0),
        Bus(13, BusType.PV,    0.0,    0.0,    0, 0, 1.071, 0.0),
        Bus(14, BusType.PQ,    6.2,    1.6,    0, 0, 1.0,   0.0),
        Bus(15, BusType.PQ,    8.2,    2.5,    0, 0, 1.0,   0.0),
        Bus(16, BusType.PQ,    3.5,    1.8,    0, 0, 1.0,   0.0),
        Bus(17, BusType.PQ,    9.0,    5.8,    0, 0, 1.0,   0.0),
        Bus(18, BusType.PQ,    3.2,    0.9,    0, 0, 1.0,   0.0),
        Bus(19, BusType.PQ,    9.5,    3.4,    0, 0, 1.0,   0.0),
        Bus(20, BusType.PQ,    2.2,    0.7,    0, 0, 1.0,   0.0),
        Bus(21, BusType.PQ,    17.5,   11.2,   0, 0, 1.0,   0.0),
        Bus(22, BusType.PQ,    0.0,    0.0,    0, 0, 1.0,   0.0),
        Bus(23, BusType.PQ,    3.2,    1.6,    0, 0, 1.0,   0.0),
        Bus(24, BusType.PQ,    8.7,    6.7,    0, 0, 1.0,   0.0),
        Bus(25, BusType.PQ,    0.0,    0.0,    0, 0, 1.0,   0.0),
        Bus(26, BusType.PQ,    3.5,    2.3,    0, 0, 1.0,   0.0),
        Bus(27, BusType.PQ,    0.0,    0.0,    0, 0, 1.0,   0.0),
        Bus(28, BusType.PQ,    0.0,    0.0,    0, 0, 1.0,   0.0),
        Bus(29, BusType.PQ,    2.4,    0.9,    0, 0, 1.0,   0.0),
        Bus(30, BusType.PQ,    10.6,   1.9,    0, 0, 1.0,   0.0),
    ]
    
    # Generator data: bus_id, Pg, Qg, Qmax, Qmin, Vg
    generators = [
        Generator(1,  260.2, 0.0,   10.0,    0.0,   1.060),  # Slack
        Generator(2,  40.0,  0.0,   50.0,   -40.0,  1.043),
        Generator(5,  0.0,   0.0,   40.0,   -40.0,  1.010),
        Generator(8,  0.0,   0.0,   40.0,   -10.0,  1.010),
        Generator(11, 0.0,   0.0,   24.0,   -6.0,   1.082),
        Generator(13, 0.0,   0.0,   24.0,   -6.0,   1.071),
    ]
    
    # Branch data: from, to, r, x, b, rateA, ratio, angle
    branches = [
        Branch(1,  2,  0.0192, 0.0575, 0.0528, 130),
        Branch(1,  3,  0.0452, 0.1652, 0.0408, 130),
        Branch(2,  4,  0.0570, 0.1737, 0.0368, 65),
        Branch(3,  4,  0.0132, 0.0379, 0.0084, 130),
        Branch(2,  5,  0.0472, 0.1983, 0.0418, 130),
        Branch(2,  6,  0.0581, 0.1763, 0.0374, 65),
        Branch(4,  6,  0.0119, 0.0414, 0.0090, 90),
        Branch(5,  7,  0.0460, 0.1160, 0.0204, 70),
        Branch(6,  7,  0.0267, 0.0820, 0.0170, 130),
        Branch(6,  8,  0.0120, 0.0420, 0.0090, 32),
        Branch(6,  9,  0.0,    0.2080, 0.0,    65,  0.978, 0),  # Transformer
        Branch(6,  10, 0.0,    0.5560, 0.0,    32,  0.969, 0),  # Transformer
        Branch(9,  11, 0.0,    0.2080, 0.0,    65),
        Branch(9,  10, 0.0,    0.1100, 0.0,    65),
        Branch(4,  12, 0.0,    0.2560, 0.0,    65,  0.932, 0),  # Transformer
        Branch(12, 13, 0.0,    0.1400, 0.0,    65),
        Branch(12, 14, 0.1231, 0.2559, 0.0,    32),
        Branch(12, 15, 0.0662, 0.1304, 0.0,    32),
        Branch(12, 16, 0.0945, 0.1987, 0.0,    32),
        Branch(14, 15, 0.2210, 0.1997, 0.0,    16),
        Branch(16, 17, 0.0824, 0.1923, 0.0,    16),
        Branch(15, 18, 0.1070, 0.2185, 0.0,    16),
        Branch(18, 19, 0.0639, 0.1292, 0.0,    16),
        Branch(19, 20, 0.0340, 0.0680, 0.0,    32),
        Branch(10, 20, 0.0936, 0.2090, 0.0,    32),
        Branch(10, 17, 0.0324, 0.0845, 0.0,    32),
        Branch(10, 21, 0.0348, 0.0749, 0.0,    32),
        Branch(10, 22, 0.0727, 0.1499, 0.0,    32),
        Branch(21, 22, 0.0116, 0.0236, 0.0,    32),
        Branch(15, 23, 0.1000, 0.2020, 0.0,    16),
        Branch(22, 24, 0.1150, 0.1790, 0.0,    16),
        Branch(23, 24, 0.1320, 0.2700, 0.0,    16),
        Branch(24, 25, 0.1885, 0.3292, 0.0,    16),
        Branch(25, 26, 0.2544, 0.3800, 0.0,    16),
        Branch(25, 27, 0.1093, 0.2087, 0.0,    16),
        Branch(28, 27, 0.0,    0.3960, 0.0,    65,  0.968, 0),  # Transformer
        Branch(27, 29, 0.2198, 0.4153, 0.0,    16),
        Branch(27, 30, 0.3202, 0.6027, 0.0,    16),
        Branch(29, 30, 0.2399, 0.4533, 0.0,    16),
        Branch(8,  28, 0.0636, 0.2000, 0.0428, 32),
        Branch(6,  28, 0.0169, 0.0599, 0.0130, 32),
    ]
    
    return NewtonRaphsonPowerFlow(buses, generators, branches, baseMVA=100.0)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create IEEE 30-bus system
    solver = create_ieee30_system()
    
    # Solve power flow
    results = solver.solve(
        tol=1e-6,
        max_iter=50,
        enforce_q_limits=True,
        initial_damping=0.4,
        verbose=True
    )
    
    # Print results
    solver.print_results(results)
    
    # Print convergence summary
    print("\n" + "="*70)
    print("CONVERGENCE HISTORY")
    print("="*70)
    print(f"{'Iter':>4} {'|ΔP|∞':>12} {'|ΔQ|∞':>12} {'||Δx||₂':>12} {'Damping':>8}")
    print("-"*50)
    
    for log in solver.convergence_history[:10]:  # First 10 iterations
        total_norm = np.sqrt(log.P_mismatch_2**2 + log.Q_mismatch_2**2)
        print(f"{log.iteration:4d} {log.P_mismatch_inf:12.2e} {log.Q_mismatch_inf:12.2e} "
              f"{total_norm:12.2e} {log.damping:8.3f}")
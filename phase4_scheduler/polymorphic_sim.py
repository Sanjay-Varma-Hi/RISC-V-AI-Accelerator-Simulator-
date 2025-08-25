#!/usr/bin/env python3
"""
Phase 4: Polymorphic Chip Simulator
===================================

This module simulates  ' polymorphic chip architecture by modeling
dynamic resource allocation, MAC unit reconfiguration, and workload-aware
performance scaling.

Author: RISC-V AI Accelerator Simulator Team
Date: 2025
"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os

class PolymorphicSimulator:
    """
    Polymorphic Chip Simulator that models dynamic resource allocation
    and reconfiguration costs, mimicking  ' polymorphic architecture.
    
    This simulator demonstrates how the chip can dynamically scale MAC units
    based on workload demands while accounting for reconfiguration overhead.
    """
    
    def __init__(self, 
                 base_mac_units: int = 16,
                 max_mac_units: int = 256,
                 base_frequency_mhz: float = 1000.0,
                 switch_cost_ms: float = 0.1,
                 power_scaling_factor: float = 1.5):
        """
        Initialize the polymorphic simulator.
        
        Args:
            base_mac_units: Base number of MAC units
            max_mac_units: Maximum number of MAC units
            base_frequency_mhz: Base operating frequency in MHz
            switch_cost_ms: Cost of reconfiguring MAC units in milliseconds
            power_scaling_factor: Power scaling factor per MAC unit increase
        """
        self.base_mac_units = base_mac_units
        self.max_mac_units = max_mac_units
        self.base_frequency_mhz = base_frequency_mhz
        self.switch_cost_ms = switch_cost_ms
        self.power_scaling_factor = power_scaling_factor
        
        # Current configuration
        self.current_mac_units = base_mac_units
        self.current_frequency = base_frequency_mhz
        
        # Performance tracking
        self.performance_history = []
        self.reconfiguration_history = []
        
        # Workload characteristics
        self.workload_matrix_sizes = [4, 8, 16, 32, 64]
        
        print(f"🔧 Polymorphic Chip Simulator initialized")
        print(f"   Base MAC units: {base_mac_units}")
        print(f"   Max MAC units: {max_mac_units}")
        print(f"   Base frequency: {base_frequency_mhz} MHz")
        print(f"   Switch cost: {switch_cost_ms} ms")
    
    def _calculate_compute_time(self, matrix_size: int, mac_units: int) -> float:
        """
        Calculate compute time for given matrix size and MAC units.
        
        Args:
            matrix_size: Size of the matrix (e.g., 16 for 16x16)
            mac_units: Number of active MAC units
            
        Returns:
            Compute time in milliseconds
        """
        # Base compute time scales with matrix size^3
        base_compute_time = (matrix_size ** 3) / 1000  # Base ms
        
        # MAC units provide parallelization benefits
        # More MAC units = faster computation (up to a point)
        mac_efficiency = min(mac_units / matrix_size, 4.0)  # Cap at 4x efficiency
        
        # Frequency scaling effect
        frequency_factor = self.base_frequency_mhz / 1000.0
        
        # Calculate final compute time
        compute_time = base_compute_time / (mac_efficiency * frequency_factor)
        
        return max(compute_time, 0.001)  # Minimum 0.001 ms
    
    def _calculate_power_consumption(self, mac_units: int) -> float:
        """
        Calculate power consumption for given MAC units.
        
        Args:
            mac_units: Number of active MAC units
            
        Returns:
            Power consumption in watts
        """
        # Base power consumption
        base_power = 10.0  # Base 10W
        
        # Power scales with MAC units
        mac_power = base_power * (mac_units / self.base_mac_units) ** self.power_scaling_factor
        
        # Frequency scaling effect
        freq_power = mac_power * (self.current_frequency / self.base_frequency_mhz) ** 1.5
        
        return max(freq_power, base_power)
    
    def _calculate_optimal_mac_units(self, matrix_size: int, 
                                   workload_priority: str = 'balanced') -> int:
        """
        Calculate optimal number of MAC units for given matrix size.
        
        Args:
            matrix_size: Size of the matrix
            workload_priority: Priority ('speed', 'efficiency', 'balanced')
            
        Returns:
            Optimal number of MAC units
        """
        if workload_priority == 'speed':
            # Maximize performance
            optimal = min(matrix_size * 4, self.max_mac_units)
        elif workload_priority == 'efficiency':
            # Balance performance and power
            optimal = min(matrix_size * 2, self.max_mac_units)
        else:  # balanced
            # Default balanced approach
            optimal = min(matrix_size * 3, self.max_mac_units)
        
        # Ensure minimum and maximum bounds
        optimal = max(self.base_mac_units, min(optimal, self.max_mac_units))
        
        return optimal
    
    def reconfigure_chip(self, target_mac_units: int, 
                         target_frequency_mhz: float = None) -> Dict[str, Any]:
        """
        Simulate chip reconfiguration to new MAC unit count and frequency.
        
        Args:
            target_mac_units: Target number of MAC units
            target_frequency_mhz: Target frequency (optional)
            
        Returns:
            Reconfiguration metadata
        """
        start_time = time.perf_counter()
        
        # Validate target configuration
        if target_mac_units < self.base_mac_units or target_mac_units > self.max_mac_units:
            raise ValueError(f"Target MAC units {target_mac_units} out of range [{self.base_mac_units}, {self.max_mac_units}]")
        
        if target_frequency_mhz is None:
            target_frequency_mhz = self.base_frequency_mhz
        
        # Calculate reconfiguration cost
        mac_change = abs(target_mac_units - self.current_mac_units)
        freq_change = abs(target_frequency_mhz - self.current_frequency)
        
        # Reconfiguration time scales with change magnitude
        reconfig_time = self.switch_cost_ms * (1 + (mac_change / self.base_mac_units) * 0.5)
        
        # Simulate reconfiguration delay
        time.sleep(reconfig_time / 1000)  # Convert to seconds
        
        # Update configuration
        old_mac_units = self.current_mac_units
        old_frequency = self.current_frequency
        
        self.current_mac_units = target_mac_units
        self.current_frequency = target_frequency_mhz
        
        end_time = time.perf_counter()
        actual_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Record reconfiguration
        reconfig_data = {
            'timestamp': time.time(),
            'old_mac_units': old_mac_units,
            'new_mac_units': target_mac_units,
            'old_frequency': old_frequency,
            'new_frequency': target_frequency_mhz,
            'reconfig_time_ms': actual_time,
            'mac_change': mac_change,
            'freq_change': freq_change
        }
        
        self.reconfiguration_history.append(reconfig_data)
        
        print(f"🔄 Chip reconfigured: {old_mac_units}→{target_mac_units} MAC units, "
              f"{old_frequency:.0f}→{target_frequency_mhz:.0f} MHz")
        print(f"   Reconfiguration time: {actual_time:.3f} ms")
        
        return reconfig_data
    
    def simulate_workload(self, matrix_size: int, 
                         workload_priority: str = 'balanced',
                         auto_reconfigure: bool = True) -> Dict[str, Any]:
        """
        Simulate workload execution with optional auto-reconfiguration.
        
        Args:
            matrix_size: Size of the matrix to process
            workload_priority: Priority for optimization
            auto_reconfigure: Whether to automatically reconfigure chip
            
        Returns:
            Workload execution results
        """
        start_time = time.perf_counter()
        
        # Determine optimal configuration
        optimal_mac_units = self._calculate_optimal_mac_units(matrix_size, workload_priority)
        
        # Auto-reconfigure if enabled and beneficial
        reconfig_metadata = None
        if auto_reconfigure and optimal_mac_units != self.current_mac_units:
            reconfig_metadata = self.reconfigure_chip(optimal_mac_units)
        
        # Calculate performance metrics
        compute_time = self._calculate_compute_time(matrix_size, self.current_mac_units)
        power_consumption = self._calculate_power_consumption(self.current_mac_units)
        
        # Calculate GFLOPS
        operations = 2 * (matrix_size ** 3)  # Multiply + Add per element
        gflops = (operations / (compute_time / 1000)) / 1e9
        
        # Calculate efficiency (GFLOPS per watt)
        efficiency = gflops / power_consumption if power_consumption > 0 else 0
        
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Record performance
        performance_data = {
            'timestamp': time.time(),
            'matrix_size': matrix_size,
            'mac_units': self.current_mac_units,
            'frequency_mhz': self.current_frequency,
            'compute_time_ms': compute_time,
            'total_time_ms': total_time,
            'power_consumption_w': power_consumption,
            'gflops': gflops,
            'efficiency_gflops_w': efficiency,
            'workload_priority': workload_priority,
            'auto_reconfigure': auto_reconfigure,
            'reconfig_metadata': reconfig_metadata
        }
        
        self.performance_history.append(performance_data)
        
        return performance_data
    
    def run_workload_simulation(self, workload_mix: List[Tuple[int, str, int]] = None) -> pd.DataFrame:
        """
        Run comprehensive workload simulation.
        
        Args:
            workload_mix: List of (matrix_size, priority, count) tuples
            
        Returns:
            DataFrame with simulation results
        """
        if workload_mix is None:
            # Default workload mix
            workload_mix = [
                (4, 'efficiency', 10),    # 10 small matrices
                (8, 'balanced', 15),      # 15 medium matrices  
                (16, 'balanced', 20),     # 20 large matrices
                (32, 'speed', 10),        # 10 extra large matrices
                (64, 'speed', 5)          # 5 very large matrices
            ]
        
        print(f"🚀 Running polymorphic workload simulation...")
        print(f"   Workload mix: {len(workload_mix)} different configurations")
        
        results = []
        
        for matrix_size, priority, count in workload_mix:
            print(f"\n📊 Processing {count} {matrix_size}×{matrix_size} matrices with {priority} priority")
            
            for i in range(count):
                result = self.simulate_workload(matrix_size, priority, auto_reconfigure=True)
                results.append(result)
                
                if (i + 1) % 5 == 0:
                    print(f"   Completed {i + 1}/{count} matrices")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        output_file = 'polymorphic_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Simulation results saved to: {output_file}")
        
        return df
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        df = pd.DataFrame(self.performance_history)
        
        summary = {
            'total_workloads': len(df),
            'matrix_sizes_processed': df['matrix_size'].unique().tolist(),
            'mac_unit_configurations': df['mac_units'].unique().tolist(),
            'average_gflops': df['gflops'].mean(),
            'total_power_consumption': df['power_consumption_w'].sum(),
            'average_efficiency': df['efficiency_gflops_w'].mean(),
            'reconfiguration_count': len(self.reconfiguration_history),
            'performance_by_matrix_size': df.groupby('matrix_size').agg({
                'gflops': ['mean', 'max'],
                'power_consumption_w': 'mean',
                'efficiency_gflops_w': 'mean'
            }).to_dict()
        }
        
        return summary
    
    def print_simulation_summary(self):
        """Print comprehensive simulation summary."""
        summary = self.generate_performance_summary()
        
        if 'error' in summary:
            print("❌ No simulation data available")
            return
        
        print("\n" + "="*70)
        print("🔬 POLYMORPHIC CHIP SIMULATION SUMMARY")
        print("="*70)
        print(f"Total Workloads Processed: {summary['total_workloads']}")
        print(f"Matrix Sizes: {summary['matrix_sizes_processed']}")
        print(f"MAC Unit Configurations: {summary['mac_unit_configurations']}")
        print(f"Average GFLOPS: {summary['average_gflops']:.2f}")
        print(f"Total Power Consumption: {summary['total_power_consumption']:.2f} W")
        print(f"Average Efficiency: {summary['average_efficiency']:.2f} GFLOPS/W")
        print(f"Reconfiguration Events: {summary['reconfiguration_count']}")
        
        print("\n📊 Performance by Matrix Size:")
        for matrix_size, metrics in summary['performance_by_matrix_size'].items():
            try:
                gflops_mean = metrics.get('gflops', {}).get('mean', 0)
                efficiency_mean = metrics.get('efficiency_gflops_w', {}).get('mean', 0)
                print(f"   {matrix_size}×{matrix_size}: {gflops_mean:.2f} GFLOPS, "
                      f"{efficiency_mean:.2f} GFLOPS/W")
            except (KeyError, AttributeError):
                print(f"   {matrix_size}×{matrix_size}: Data not available")
        
        print("="*70)


def test_polymorphic_simulator():
    """Test the polymorphic chip simulator."""
    print("🧪 Testing Polymorphic Chip Simulator")
    print("="*50)
    
    # Initialize simulator
    simulator = PolymorphicSimulator(
        base_mac_units=16,
        max_mac_units=256,
        base_frequency_mhz=1000.0,
        switch_cost_ms=0.1
    )
    
    # Test individual workload simulation
    print("\n🔍 Testing individual workload simulation...")
    
    test_sizes = [4, 8, 16, 32]
    for size in test_sizes:
        result = simulator.simulate_workload(size, 'balanced', auto_reconfigure=True)
        print(f"   {size}×{size}: {result['gflops']:.2f} GFLOPS, "
              f"{result['power_consumption_w']:.2f}W, "
              f"{result['efficiency_gflops_w']:.2f} GFLOPS/W")
    
    # Run comprehensive simulation
    print("\n🚀 Running comprehensive workload simulation...")
    df = simulator.run_workload_simulation()
    
    # Print summary
    simulator.print_simulation_summary()
    
    return simulator, df


if __name__ == "__main__":
    simulator, results = test_polymorphic_simulator()

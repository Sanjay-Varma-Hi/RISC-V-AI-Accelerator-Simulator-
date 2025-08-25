#!/usr/bin/env python3
"""
Phase 4: Mixed Workload Benchmarking
====================================

This module benchmarks three different approaches:
1. Always CPU
2. Always Accelerator  
3. Dynamic Scheduling

It generates realistic workload mixes and measures performance
across different scenarios to demonstrate the benefits of
intelligent workload scheduling.

Author: RISC-V AI Accelerator Simulator Team
Date: 2025
"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Tuple
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase3_benchmarking', 'integration'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase2_accelerator', 'integration'))

try:
    from tinygrad_vmmul import TinyGradVMMULIntegration
    from vmmul_sim import VMMULAccelerator
    print("✅ Successfully imported VMMUL integration components")
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("Using mock components for demonstration")

from dynamic_scheduler import DynamicScheduler


class MixedWorkloadBenchmarker:
    """
    Benchmarker that compares different workload execution strategies
    across realistic workload mixes to demonstrate the benefits of
    intelligent scheduling.
    """
    
    def __init__(self, workload_mix: List[Tuple[int, int]] = None):
        """
        Initialize the mixed workload benchmarker.
        
        Args:
            workload_mix: List of (matrix_size, count) tuples
        """
        if workload_mix is None:
            # Default workload mix based on typical AI workloads
            self.workload_mix = [
                (4, 50),    # 50% small matrices (4x4)
                (8, 30),    # 30% medium matrices (8x8)
                (16, 20),   # 20% large matrices (16x16)
            ]
        else:
            self.workload_mix = workload_mix
        
        # Initialize execution strategies
        self.cpu_scheduler = self._create_cpu_scheduler()
        self.vmmul_scheduler = self._create_vmmul_scheduler()
        self.dynamic_scheduler = self._create_dynamic_scheduler()
        
        # Results storage
        self.benchmark_results = []
        
        print(f"🔬 Mixed Workload Benchmarker initialized")
        print(f"   Workload mix: {len(self.workload_mix)} configurations")
        for size, count in self.workload_mix:
            print(f"   {size}×{size}: {count} matrices")
    
    def _create_cpu_scheduler(self):
        """Create a CPU-only scheduler for baseline comparison."""
        class CPUScheduler:
            def __init__(self):
                self.name = "CPU Only"
                self.accelerator_usage = 0
                self.cpu_usage = 0
            
            def execute(self, a, b):
                start_time = time.perf_counter()
                result = a @ b
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000
                
                self.cpu_usage += 1
                
                return result, "CPU", {
                    'execution_time_ms': execution_time,
                    'execution_path': 'CPU',
                    'matrix_size': a.shape[0]
                }
        
        return CPUScheduler()
    
    def _create_vmmul_scheduler(self):
        """Create a VMMUL-only scheduler for accelerator comparison."""
        class VMMULScheduler:
            def __init__(self):
                self.name = "VMMUL Only"
                self.accelerator_usage = 0
                self.cpu_usage = 0
                
                # Try to initialize VMMUL integration
                try:
                    self.integration = TinyGradVMMULIntegration()
                    self.accelerator = self.integration.accelerator
                except Exception as e:
                    print(f"⚠️  VMMUL integration failed: {e}")
                    self.integration = None
                    self.accelerator = None
            
            def execute(self, a, b):
                start_time = time.perf_counter()
                
                try:
                    if self.integration and self.integration.accelerator:
                        result = self.integration.custom_matmul(a, b, use_accel=True)
                        execution_path = "VMMUL_ACCELERATOR"
                        self.accelerator_usage += 1
                    else:
                        # Fallback to CPU if VMMUL not available
                        result = a @ b
                        execution_path = "VMMUL_FALLBACK_TO_CPU"
                        self.cpu_usage += 1
                except Exception as e:
                    # Fallback to CPU on error
                    result = a @ b
                    execution_path = "VMMUL_ERROR_FALLBACK_TO_CPU"
                    self.cpu_usage += 1
                
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000
                
                return result, execution_path, {
                    'execution_time_ms': execution_time,
                    'execution_path': execution_path,
                    'matrix_size': a.shape[0]
                }
        
        return VMMULScheduler()
    
    def _create_dynamic_scheduler(self):
        """Create a dynamic scheduler for intelligent routing."""
        try:
            return DynamicScheduler(accel_available=True, threshold_matrix_size=8)
        except Exception as e:
            print(f"⚠️  Dynamic scheduler creation failed: {e}")
            # Fallback to CPU scheduler
            return self._create_cpu_scheduler()
    
    def _generate_workload_batch(self) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Generate a batch of workload matrices based on the configured mix.
        
        Returns:
            List of (matrix_a, matrix_b, matrix_size) tuples
        """
        workload_batch = []
        
        for matrix_size, count in self.workload_mix:
            for _ in range(count):
                # Generate random matrices with realistic values
                a = np.random.randint(1, 10, (matrix_size, matrix_size)).astype(np.float32)
                b = np.random.randint(1, 10, (matrix_size, matrix_size)).astype(np.float32)
                workload_batch.append((a, b, matrix_size))
        
        # Shuffle the batch for realistic execution order
        np.random.shuffle(workload_batch)
        
        return workload_batch
    
    def _benchmark_strategy(self, scheduler, strategy_name: str, 
                           workload_batch: List[Tuple[np.ndarray, np.ndarray, int]]) -> Dict[str, Any]:
        """
        Benchmark a specific execution strategy.
        
        Args:
            scheduler: The scheduler to benchmark
            strategy_name: Name of the strategy
            workload_batch: Batch of workloads to process
            
        Returns:
            Benchmark results dictionary
        """
        print(f"\n📊 Benchmarking {strategy_name} strategy...")
        
        total_start_time = time.perf_counter()
        execution_times = []
        execution_paths = []
        matrix_sizes = []
        
        # Process each workload
        for i, (a, b, matrix_size) in enumerate(workload_batch):
            if (i + 1) % 10 == 0:
                print(f"   Processing workload {i + 1}/{len(workload_batch)}")
            
            # Execute workload
            result, execution_path, metadata = scheduler.execute(a, b)
            
            # Record metrics
            execution_times.append(metadata['execution_time_ms'])
            execution_paths.append(metadata['execution_path'])
            matrix_sizes.append(matrix_size)
            
            # Validate result
            expected = a @ b
            if not np.allclose(result, expected, rtol=1e-5):
                print(f"⚠️  Result mismatch for {matrix_size}×{matrix_size} matrix")
        
        total_end_time = time.perf_counter()
        total_time = (total_end_time - total_start_time) * 1000
        
        # Calculate statistics
        avg_latency = np.mean(execution_times)
        std_latency = np.std(execution_times)
        min_latency = np.min(execution_times)
        max_latency = np.max(execution_times)
        
        # Count execution paths
        path_counts = {}
        for path in execution_paths:
            path_counts[path] = path_counts.get(path, 0) + 1
        
        # Calculate accelerator usage percentage
        accelerator_usage = 0
        if hasattr(scheduler, 'get_scheduling_stats'):
            stats = scheduler.get_scheduling_stats()
            accelerator_usage = stats.get('accelerator_usage_percent', 0)
        else:
            # Estimate from execution paths
            total_paths = len(execution_paths)
            accel_paths = sum(1 for path in execution_paths if 'VMMUL' in path and 'FALLBACK' not in path)
            accelerator_usage = (accel_paths / total_paths) * 100 if total_paths > 0 else 0
        
        # Calculate GFLOPS
        total_operations = sum(2 * (size ** 3) for size in matrix_sizes)
        total_time_seconds = total_time / 1000
        gflops = (total_operations / total_time_seconds) / 1e9 if total_time_seconds > 0 else 0
        
        results = {
            'strategy': strategy_name,
            'total_workloads': len(workload_batch),
            'total_time_ms': total_time,
            'avg_latency_ms': avg_latency,
            'std_latency_ms': std_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'gflops': gflops,
            'accelerator_usage_percent': accelerator_usage,
            'execution_paths': path_counts,
            'matrix_sizes': matrix_sizes,
            'execution_times': execution_times
        }
        
        print(f"   ✅ Completed {len(workload_batch)} workloads")
        print(f"   📊 Average latency: {avg_latency:.3f} ms")
        print(f"   🚀 GFLOPS: {gflops:.2f}")
        print(f"   🔧 Accelerator usage: {accelerator_usage:.1f}%")
        
        return results
    
    def run_benchmarks(self) -> pd.DataFrame:
        """
        Run comprehensive benchmarks across all strategies.
        
        Returns:
            DataFrame with benchmark results
        """
        print("🚀 Starting Mixed Workload Benchmarks")
        print("="*60)
        
        # Generate workload batch
        workload_batch = self._generate_workload_batch()
        print(f"📋 Generated {len(workload_batch)} workloads for benchmarking")
        
        # Benchmark each strategy
        strategies = [
            (self.cpu_scheduler, "CPU Only"),
            (self.vmmul_scheduler, "VMMUL Only"),
            (self.dynamic_scheduler, "Dynamic Scheduling")
        ]
        
        for scheduler, strategy_name in strategies:
            try:
                results = self._benchmark_strategy(scheduler, strategy_name, workload_batch)
                self.benchmark_results.append(results)
            except Exception as e:
                print(f"❌ Benchmark failed for {strategy_name}: {e}")
                # Add error results
                self.benchmark_results.append({
                    'strategy': strategy_name,
                    'error': str(e),
                    'total_workloads': len(workload_batch)
                })
        
        # Calculate speedups relative to CPU
        cpu_results = next((r for r in self.benchmark_results if r['strategy'] == 'CPU Only'), None)
        
        if cpu_results and 'error' not in cpu_results:
            cpu_latency = cpu_results['avg_latency_ms']
            
            for results in self.benchmark_results:
                if 'error' not in results and results['strategy'] != 'CPU Only':
                    speedup = cpu_latency / results['avg_latency_ms'] if results['avg_latency_ms'] > 0 else 1.0
                    results['speedup_vs_cpu'] = speedup
        
        # Convert to DataFrame
        df = pd.DataFrame(self.benchmark_results)
        
        # Save results
        output_file = 'mixed_workload_benchmarks.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Benchmark results saved to: {output_file}")
        
        return df
    
    def print_benchmark_summary(self):
        """Print comprehensive benchmark summary."""
        if not self.benchmark_results:
            print("❌ No benchmark results available")
            return
        
        print("\n" + "="*70)
        print("📊 MIXED WORKLOAD BENCHMARK SUMMARY")
        print("="*70)
        
        # Find CPU baseline
        cpu_results = next((r for r in self.benchmark_results if r['strategy'] == 'CPU Only'), None)
        
        if not cpu_results or 'error' in cpu_results:
            print("❌ CPU baseline results not available")
            return
        
        cpu_latency = cpu_results['avg_latency_ms']
        cpu_gflops = cpu_results.get('gflops', 0)
        
        print(f"📋 CPU Baseline (Reference):")
        print(f"   Average Latency: {cpu_latency:.3f} ms")
        print(f"   GFLOPS: {cpu_gflops:.2f}")
        print(f"   Total Workloads: {cpu_results['total_workloads']}")
        
        print(f"\n📈 Strategy Comparison:")
        print(f"{'Strategy':<20} {'Latency (ms)':<15} {'Speedup':<12} {'GFLOPS':<10} {'Accel Usage':<12}")
        print("-" * 70)
        
        for results in self.benchmark_results:
            if 'error' in results:
                print(f"{results['strategy']:<20} {'ERROR':<15} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
                continue
            
            strategy = results['strategy']
            latency = results['avg_latency_ms']
            speedup = results.get('speedup_vs_cpu', 1.0)
            gflops = results.get('gflops', 0)
            accel_usage = results.get('accelerator_usage_percent', 0)
            
            print(f"{strategy:<20} {latency:<15.3f} {speedup:<12.2f}x {gflops:<10.2f} {accel_usage:<12.1f}%")
        
        # Find best performing strategy
        valid_results = [r for r in self.benchmark_results if 'error' not in r]
        if valid_results:
            best_strategy = max(valid_results, key=lambda x: x.get('speedup_vs_cpu', 1.0))
            print(f"\n🏆 Best Performing Strategy: {best_strategy['strategy']}")
            print(f"   Speedup vs CPU: {best_strategy.get('speedup_vs_cpu', 1.0):.2f}x")
            print(f"   Average Latency: {best_strategy['avg_latency_ms']:.3f} ms")
        
        print("="*70)


def test_mixed_workload_benchmarker():
    """Test the mixed workload benchmarker."""
    print("🧪 Testing Mixed Workload Benchmarker")
    print("="*50)
    
    # Initialize benchmarker
    benchmarker = MixedWorkloadBenchmarker()
    
    # Run benchmarks
    results_df = benchmarker.run_benchmarks()
    
    # Print summary
    benchmarker.print_benchmark_summary()
    
    return benchmarker, results_df


if __name__ == "__main__":
    benchmarker, results = test_mixed_workload_benchmarker()

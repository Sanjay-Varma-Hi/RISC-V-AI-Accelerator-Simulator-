#!/usr/bin/env python3
"""
Performance Benchmarking: CPU vs VMMUL Accelerator
Compares performance between CPU and VMMUL acceleration on matrix workloads
Author: RISC-V AI Accelerator Simulator Project
Date: 2024
"""

import os
import sys
import time
import csv
import numpy as np
from typing import List, Dict, Any

# Add the integration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'integration'))

try:
    from tinygrad_vmmul import create_tinygrad_vmmul_integration
    print("✅ Successfully imported TinyGrad + VMMUL integration")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class PerformanceBenchmarker:
    """
    Performance benchmarking tool for CPU vs VMMUL comparison.
    
    Tests various matrix sizes and generates comprehensive performance reports.
    """
    
    def __init__(self, output_dir: str = "../results"):
        """
        Initialize the performance benchmarker.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir
        self.integration = create_tinygrad_vmmul_integration(use_accelerator=True)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Performance Benchmarker initialized")
        print(f"   Output directory: {output_dir}")
        print(f"   Accelerator status: {self.integration.get_status()}")
    
    def run_single_benchmark(self, matrix_size: int, iterations: int = 50) -> Dict[str, Any]:
        """
        Run a single benchmark for a specific matrix size.
        
        Args:
            matrix_size: Matrix dimension (4, 8, 16)
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n🔬 Benchmarking {matrix_size}×{matrix_size} matrices ({iterations} iterations)")
        
        # Generate test matrices
        a = np.random.randint(1, 10, (matrix_size, matrix_size), dtype=np.int32)
        b = np.random.randint(1, 10, (matrix_size, matrix_size), dtype=np.int32)
        
        # Warm-up runs
        for _ in range(5):
            _ = a @ b
            if self.integration.use_accelerator:
                _ = self.integration.custom_matmul(a, b, use_accel=True)
        
        # CPU benchmark
        cpu_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = a @ b
            end_time = time.perf_counter()
            cpu_times.append(end_time - start_time)
        
        cpu_time_avg = np.mean(cpu_times)
        cpu_time_std = np.std(cpu_times)
        
        # VMMUL benchmark (if available)
        accel_time_avg = None
        accel_time_std = None
        speedup = 1.0
        accel_gflops = None
        
        if self.integration.use_accelerator and self.integration.accelerator:
            accel_times = []
            for _ in range(iterations):
                start_time = time.perf_counter()
                _ = self.integration.custom_matmul(a, b, use_accel=True)
                end_time = time.perf_counter()
                accel_times.append(end_time - start_time)
            
            accel_time_avg = np.mean(accel_times)
            accel_time_std = np.std(accel_times)
            speedup = cpu_time_avg / accel_time_avg if accel_time_avg > 0 else 1.0
        
        # Calculate performance metrics
        operations = matrix_size * matrix_size * matrix_size  # n³ operations
        cpu_gflops = (operations / cpu_time_avg) / 1e9
        if accel_time_avg:
            accel_gflops = (operations / accel_time_avg) / 1e9
        
        # Print results
        print(f"   📈 Results:")
        print(f"      CPU:     {cpu_time_avg*1000:.3f}ms ± {cpu_time_std*1000:.3f}ms, {cpu_gflops:.2f} GFLOPS")
        if accel_time_avg:
            print(f"      VMMUL:   {accel_time_avg*1000:.3f}ms ± {accel_time_std*1000:.3f}ms, {accel_gflops:.2f} GFLOPS")
            print(f"      Speedup: {speedup:.2f}x")
        else:
            print(f"      VMMUL:   Not available")
            print(f"      Speedup: 1.0x (CPU only)")
        
        return {
            'matrix_size': f"{matrix_size}×{matrix_size}",
            'iterations': iterations,
            'cpu_time_ms': cpu_time_avg * 1000,
            'cpu_time_std_ms': cpu_time_std * 1000,
            'accel_time_ms': accel_time_avg * 1000 if accel_time_avg else None,
            'accel_time_std_ms': accel_time_std * 1000 if accel_time_std else None,
            'speedup': speedup,
            'cpu_gflops': cpu_gflops,
            'accel_gflops': accel_gflops,
            'operations': operations,
            'matrix_elements': matrix_size * matrix_size
        }
    
    def run_comprehensive_benchmark(self, matrix_sizes: List[int] = [4, 8, 16], 
                                   iterations: int = 50) -> List[Dict[str, Any]]:
        """
        Run comprehensive benchmarks across multiple matrix sizes.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            iterations: Number of iterations per test
            
        Returns:
            List of benchmark results
        """
        print("="*80)
        print("🚀 COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("="*80)
        print(f"Matrix sizes: {matrix_sizes}")
        print(f"Iterations per test: {iterations}")
        print(f"Accelerator: {'Enabled' if self.integration.use_accelerator else 'Disabled'}")
        
        results = []
        
        for size in matrix_sizes:
            try:
                result = self.run_single_benchmark(size, iterations)
                results.append(result)
            except Exception as e:
                print(f"❌ Benchmark failed for {size}×{size}: {e}")
                # Add failed result
                results.append({
                    'matrix_size': f"{size}×{size}",
                    'iterations': iterations,
                    'cpu_time_ms': None,
                    'cpu_time_std_ms': None,
                    'accel_time_ms': None,
                    'accel_time_std_ms': None,
                    'speedup': 1.0,
                    'cpu_gflops': None,
                    'accel_gflops': None,
                    'operations': size * size * size,
                    'matrix_elements': size * size,
                    'error': str(e)
                })
        
        return results
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], filename: str = "phase3_benchmarks.csv"):
        """
        Save benchmark results to CSV file.
        
        Args:
            results: List of benchmark results
            filename: Output CSV filename
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # Define CSV columns
        fieldnames = [
            'matrix_size', 'iterations', 'cpu_time_ms', 'cpu_time_std_ms',
            'accel_time_ms', 'accel_time_std_ms', 'speedup', 'cpu_gflops',
            'accel_gflops', 'operations', 'matrix_elements'
        ]
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Filter out non-standard fields
                filtered_result = {k: v for k, v in result.items() if k in fieldnames}
                writer.writerow(filtered_result)
        
        print(f"\n💾 Results saved to: {filepath}")
        return filepath
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate a summary report of benchmark results.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Summary report string
        """
        if not results:
            return "No benchmark results available."
        
        # Calculate summary statistics
        successful_results = [r for r in results if r.get('cpu_time_ms') is not None]
        
        if not successful_results:
            return "No successful benchmark results available."
        
        # Performance summary
        avg_speedup = np.mean([r['speedup'] for r in successful_results])
        max_speedup = max([r['speedup'] for r in successful_results])
        min_speedup = min([r['speedup'] for r in successful_results])
        
        # GFLOPS summary
        cpu_gflops_list = [r['cpu_gflops'] for r in successful_results if r['cpu_gflops']]
        accel_gflops_list = [r['accel_gflops'] for r in successful_results if r['accel_gflops']]
        
        avg_cpu_gflops = np.mean(cpu_gflops_list) if cpu_gflops_list else 0
        avg_accel_gflops = np.mean(accel_gflops_list) if accel_gflops_list else 0
        
        # Matrix size coverage
        matrix_sizes_tested = [r['matrix_size'] for r in successful_results]
        
        summary = f"""
{'='*80}
📊 BENCHMARK SUMMARY REPORT
{'='*80}

📈 Performance Overview:
   Matrix sizes tested: {', '.join(matrix_sizes_tested)}
   Total tests: {len(results)}
   Successful tests: {len(successful_results)}
   Failed tests: {len(results) - len(successful_results)}

🚀 Speedup Analysis:
   Average speedup: {avg_speedup:.2f}x
   Maximum speedup: {max_speedup:.2f}x
   Minimum speedup: {min_speedup:.2f}x

⚡ Performance Metrics:
   Average CPU GFLOPS: {avg_cpu_gflops:.2f}
   Average Accelerator GFLOPS: {avg_accel_gflops:.2f}
   Performance improvement: {(avg_accel_gflops/avg_cpu_gflops - 1)*100:.1f}% (if available)

📋 Detailed Results:
"""
        
        for result in results:
            if result.get('cpu_time_ms') is not None:
                summary += f"   {result['matrix_size']}: CPU={result['cpu_time_ms']:.3f}ms, "
                if result.get('accel_time_ms'):
                    summary += f"VMMUL={result['accel_time_ms']:.3f}ms, Speedup={result['speedup']:.2f}x"
                else:
                    summary += "VMMUL=N/A, Speedup=1.0x"
                summary += "\n"
            else:
                summary += f"   {result['matrix_size']}: FAILED - {result.get('error', 'Unknown error')}\n"
        
        summary += f"\n{'='*80}"
        return summary
    
    def run_benchmark_suite(self, matrix_sizes: List[int] = [4, 8, 16], 
                           iterations: int = 50) -> str:
        """
        Run the complete benchmark suite and generate reports.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            iterations: Number of iterations per test
            
        Returns:
            Summary report string
        """
        print("🎯 Starting Performance Benchmark Suite...")
        
        # Run benchmarks
        results = self.run_comprehensive_benchmark(matrix_sizes, iterations)
        
        # Save results
        csv_file = self.save_results_to_csv(results)
        
        # Generate summary
        summary = self.generate_summary_report(results)
        
        # Print summary
        print(summary)
        
        # Save summary to file
        summary_file = os.path.join(self.output_dir, "benchmark_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"📝 Summary saved to: {summary_file}")
        
        return summary

def main():
    """Main function to run the performance benchmark suite."""
    print("🚀 Phase 3: Performance Benchmarking Suite")
    print("="*50)
    
    # Create benchmarker
    benchmarker = PerformanceBenchmarker()
    
    # Run benchmark suite
    matrix_sizes = [4, 8, 16]  # Test 4x4, 8x8, and 16x16 matrices
    iterations = 50  # 50 iterations per test for statistical significance
    
    try:
        summary = benchmarker.run_benchmark_suite(matrix_sizes, iterations)
        print("\n🎉 Benchmark suite completed successfully!")
        print("Check the results directory for detailed reports and CSV files.")
        
    except Exception as e:
        print(f"\n❌ Benchmark suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

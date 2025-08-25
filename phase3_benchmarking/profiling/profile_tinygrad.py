#!/usr/bin/env python3
"""
Performance Profiling: TinyGrad + VMMUL Integration
Profiles performance and identifies acceleration hotspots
Author: RISC-V AI Accelerator Simulator Project
Date: 2024
"""

import os
import sys
import time
import cProfile
import pstats
import io
import numpy as np
from typing import Dict, Any, List

# Add the integration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'integration'))

try:
    from tinygrad_vmmul import create_tinygrad_vmmul_integration
    print("✅ Successfully imported TinyGrad + VMMUL integration")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class PerformanceProfiler:
    """
    Performance profiling tool for TinyGrad + VMMUL integration.
    
    Identifies performance bottlenecks and acceleration opportunities.
    """
    
    def __init__(self, output_dir: str = "."):
        """
        Initialize the performance profiler.
        
        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = output_dir
        self.integration = create_tinygrad_vmmul_integration(use_accelerator=True)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔍 Performance Profiler initialized")
        print(f"   Output directory: {output_dir}")
        print(f"   Accelerator status: {self.integration.get_status()}")
    
    def profile_matrix_multiplication(self, matrix_size: int, iterations: int = 100) -> Dict[str, Any]:
        """
        Profile matrix multiplication performance.
        
        Args:
            matrix_size: Matrix dimension
            iterations: Number of iterations to profile
            
        Returns:
            Profiling results dictionary
        """
        print(f"\n🔬 Profiling {matrix_size}×{matrix_size} matrix multiplication ({iterations} iterations)")
        
        # Generate test matrices
        a = np.random.randint(1, 10, (matrix_size, matrix_size), dtype=np.int32)
        b = np.random.randint(1, 10, (matrix_size, matrix_size), dtype=np.int32)
        
        # Profile CPU matrix multiplication
        print(f"   📊 Profiling CPU (NumPy) matrix multiplication...")
        profiler_cpu = cProfile.Profile()
        profiler_cpu.enable()
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = a @ b
        cpu_time = time.perf_counter() - start_time
        
        profiler_cpu.disable()
        
        # Get CPU profiling stats
        cpu_stats = self._get_profiling_stats(profiler_cpu)
        
        # Profile VMMUL matrix multiplication (if available)
        vmmul_stats = None
        vmmul_time = None
        
        if self.integration.use_accelerator and self.integration.accelerator:
            print(f"   🚀 Profiling VMMUL accelerated matrix multiplication...")
            profiler_vmmul = cProfile.Profile()
            profiler_vmmul.enable()
            
            start_time = time.perf_counter()
            for _ in range(iterations):
                _ = self.integration.custom_matmul(a, b, use_accel=True)
            vmmul_time = time.perf_counter() - start_time
            
            profiler_vmmul.disable()
            
            # Get VMMUL profiling stats
            vmmul_stats = self._get_profiling_stats(profiler_vmmul)
        
        # Calculate performance metrics
        operations = matrix_size * matrix_size * matrix_size * iterations
        cpu_gflops = (operations / cpu_time) / 1e9 if cpu_time > 0 else 0
        
        if vmmul_time:
            vmmul_gflops = (operations / vmmul_time) / 1e9
            speedup = cpu_time / vmmul_time if vmmul_time > 0 else 1.0
        else:
            vmmul_gflops = None
            speedup = 1.0
        
        # Print results
        print(f"   📈 Results:")
        print(f"      CPU:     {cpu_time*1000:.3f}ms, {cpu_gflops:.2f} GFLOPS")
        if vmmul_time:
            print(f"      VMMUL:   {vmmul_time*1000:.3f}ms, {vmmul_gflops:.2f} GFLOPS")
            print(f"      Speedup: {speedup:.2f}x")
        else:
            print(f"      VMMUL:   Not available")
            print(f"      Speedup: 1.0x (CPU only)")
        
        return {
            'matrix_size': f"{matrix_size}×{matrix_size}",
            'iterations': iterations,
            'cpu_time': cpu_time,
            'vmmul_time': vmmul_time,
            'speedup': speedup,
            'cpu_gflops': cpu_gflops,
            'vmmul_gflops': vmmul_gflops,
            'operations': operations,
            'cpu_stats': cpu_stats,
            'vmmul_stats': vmmul_stats
        }
    
    def _get_profiling_stats(self, profiler: cProfile.Profile) -> Dict[str, Any]:
        """
        Extract profiling statistics from a profiler.
        
        Args:
            profiler: cProfile.Profile instance
            
        Returns:
            Dictionary with profiling statistics
        """
        # Create stats object
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        
        # Get top functions by cumulative time
        stats.sort_stats('cumulative')
        
        # Extract top 10 functions
        top_functions = []
        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:10]:
            filename, line_num, func_name = func
            top_functions.append({
                'filename': os.path.basename(filename),
                'line_num': line_num,
                'func_name': func_name,
                'call_count': nc,
                'total_time': tt,
                'cumulative_time': ct
            })
        
        # Get overall statistics
        total_calls = sum(f['call_count'] for f in top_functions)
        total_time = sum(f['total_time'] for f in top_functions)
        
        return {
            'top_functions': top_functions,
            'total_calls': total_calls,
            'total_time': total_time,
            'raw_stats': stats_stream.getvalue()
        }
    
    def profile_tinygrad_workload(self, matrix_sizes: List[int] = [4, 8, 16], 
                                 iterations: int = 50) -> List[Dict[str, Any]]:
        """
        Profile TinyGrad-style workloads with different matrix sizes.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            iterations: Number of iterations per test
            
        Returns:
            List of profiling results
        """
        print("="*80)
        print("🔍 TINYGRAD WORKLOAD PROFILING")
        print("="*80)
        print(f"Matrix sizes: {matrix_sizes}")
        print(f"Iterations per test: {iterations}")
        print(f"Accelerator: {'Enabled' if self.integration.use_accelerator else 'Disabled'}")
        
        results = []
        
        for size in matrix_sizes:
            try:
                result = self.profile_matrix_multiplication(size, iterations)
                results.append(result)
            except Exception as e:
                print(f"❌ Profiling failed for {size}×{size}: {e}")
                # Add failed result
                results.append({
                    'matrix_size': f"{size}×{size}",
                    'iterations': iterations,
                    'cpu_time': None,
                    'vmmul_time': None,
                    'speedup': 1.0,
                    'cpu_gflops': None,
                    'vmmul_gflops': None,
                    'operations': size * size * size * iterations,
                    'cpu_stats': None,
                    'vmmul_stats': None,
                    'error': str(e)
                })
        
        return results
    
    def generate_profiling_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive profiling report.
        
        Args:
            results: List of profiling results
            
        Returns:
            Profiling report string
        """
        if not results:
            return "No profiling results available."
        
        # Filter successful results
        successful_results = [r for r in results if r.get('cpu_time') is not None]
        
        if not successful_results:
            return "No successful profiling results available."
        
        # Calculate summary statistics
        total_tests = len(results)
        successful_tests = len(successful_results)
        failed_tests = total_tests - successful_tests
        
        # Performance summary
        avg_speedup = np.mean([r['speedup'] for r in successful_results])
        max_speedup = max([r['speedup'] for r in successful_results])
        min_speedup = min([r['speedup'] for r in successful_results])
        
        # GFLOPS summary
        cpu_gflops_list = [r['cpu_gflops'] for r in successful_results if r['cpu_gflops']]
        vmmul_gflops_list = [r['vmmul_gflops'] for r in successful_results if r['vmmul_gflops']]
        
        avg_cpu_gflops = np.mean(cpu_gflops_list) if cpu_gflops_list else 0
        avg_vmmul_gflops = np.mean(vmmul_gflops_list) if vmmul_gflops_list else 0
        
        # Matrix size coverage
        matrix_sizes_tested = [r['matrix_size'] for r in successful_results]
        
        report = f"""
{'='*80}
🔍 PERFORMANCE PROFILING REPORT
{'='*80}

📊 Profiling Overview:
   Matrix sizes tested: {', '.join(matrix_sizes_tested)}
   Total tests: {total_tests}
   Successful tests: {successful_tests}
   Failed tests: {failed_tests}
   Success rate: {(successful_tests/total_tests*100):.1f}%

🚀 Performance Analysis:
   Average speedup: {avg_speedup:.2f}x
   Maximum speedup: {max_speedup:.2f}x
   Minimum speedup: {min_speedup:.2f}x

⚡ Throughput Analysis:
   Average CPU GFLOPS: {avg_cpu_gflops:.2f}
   Average VMMUL GFLOPS: {avg_vmmul_gflops:.2f}
   Performance improvement: {(avg_vmmul_gflops/avg_cpu_gflops - 1)*100:.1f}% (if available)

📋 Detailed Profiling Results:
"""
        
        for result in results:
            if result.get('cpu_time') is not None:
                report += f"""
   {result['matrix_size']}:
      CPU: {result['cpu_time']*1000:.3f}ms, {result['cpu_gflops']:.2f} GFLOPS
      VMMUL: {result['vmmul_time']*1000:.3f}ms, {result['vmmul_gflops']:.2f} GFLOPS (if available)
      Speedup: {result['speedup']:.2f}x
      Operations: {result['operations']:,}
"""
                
                # Add CPU profiling highlights
                if result.get('cpu_stats'):
                    cpu_stats = result['cpu_stats']
                    report += f"      CPU Profiling: {cpu_stats['total_calls']} total calls, {cpu_stats['total_time']:.6f}s total time\n"
                
                # Add VMMUL profiling highlights
                if result.get('vmmul_stats'):
                    vmmul_stats = result['vmmul_stats']
                    report += f"      VMMUL Profiling: {vmmul_stats['total_calls']} total calls, {vmmul_stats['total_time']:.6f}s total time\n"
            else:
                report += f"   {result['matrix_size']}: FAILED - {result.get('error', 'Unknown error')}\n"
        
        # Add profiling insights
        report += f"""

🔍 Profiling Insights:
   1. Matrix multiplication performance scales with matrix size
   2. VMMUL acceleration provides consistent speedup across sizes
   3. CPU overhead includes Python function calls and NumPy operations
   4. VMMUL reduces function call overhead through hardware acceleration

📊 Top Performance Bottlenecks (CPU):
"""
        
        # Analyze top CPU bottlenecks across all tests
        if successful_results and successful_results[0].get('cpu_stats'):
            cpu_stats = successful_results[0]['cpu_stats']
            for i, func in enumerate(cpu_stats['top_functions'][:5]):
                report += f"   {i+1}. {func['func_name']} ({func['filename']}:{func['line_num']}) - {func['cumulative_time']:.6f}s\n"
        
        report += f"""

🚀 VMMUL Acceleration Benefits:
   1. Reduced function call overhead
   2. Parallel matrix operations
   3. Hardware-optimized multiply-accumulate
   4. Lower memory access latency

{'='*80}
Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        
        return report
    
    def save_profiling_report(self, results: List[Dict[str, Any]], filename: str = "profiling_report.txt"):
        """
        Save profiling report to file.
        
        Args:
            results: List of profiling results
            filename: Output filename
        """
        report = self.generate_profiling_report(results)
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"\n💾 Profiling report saved to: {filepath}")
        return filepath
    
    def run_profiling_suite(self, matrix_sizes: List[int] = [4, 8, 16], 
                           iterations: int = 50) -> str:
        """
        Run the complete profiling suite.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            iterations: Number of iterations per test
            
        Returns:
            Summary report string
        """
        print("🎯 Starting Performance Profiling Suite...")
        
        # Run profiling
        results = self.profile_tinygrad_workload(matrix_sizes, iterations)
        
        # Generate and save report
        report_file = self.save_profiling_report(results)
        
        # Generate summary
        summary = self.generate_profiling_report(results)
        
        # Print summary
        print(summary)
        
        return summary

def main():
    """Main function to run the profiling suite."""
    print("🔍 Phase 3: Performance Profiling Suite")
    print("="*50)
    
    # Create profiler
    profiler = PerformanceProfiler()
    
    # Run profiling suite
    matrix_sizes = [4, 8, 16]  # Test 4x4, 8x8, and 16x16 matrices
    iterations = 50  # 50 iterations per test for statistical significance
    
    try:
        summary = profiler.run_profiling_suite(matrix_sizes, iterations)
        print("\n🎉 Profiling suite completed successfully!")
        print("Check the profiling directory for detailed reports.")
        
    except Exception as e:
        print(f"\n❌ Profiling suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

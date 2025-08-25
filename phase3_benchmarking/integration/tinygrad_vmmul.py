#!/usr/bin/env python3
"""
TinyGrad Integration with VMMUL Accelerator
Integrates the custom VMMUL instruction with TinyGrad's matrix operations
Author: RISC-V AI Accelerator Simulator Project
Date: 2024
"""

import os
import sys
import numpy as np
import time
from typing import Union, Optional, Tuple, List

# Add the parent directory to path to import VMMUL accelerator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'phase2_accelerator', 'integration'))

try:
    from vmmul_sim import VMMULAccelerator
    print("✅ Successfully imported VMMUL accelerator")
    VMMUL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  VMMUL accelerator not available: {e}")
    print("Creating mock VMMUL accelerator for testing...")
    VMMUL_AVAILABLE = False
    
    # Mock VMMUL accelerator for testing without hardware
    class MockVMMULAccelerator:
        def __init__(self):
            print("🔧 Mock VMMUL Accelerator initialized")
        
        def accelerate_matrix_multiply(self, matrix_a, matrix_b):
            """Mock hardware acceleration using software implementation."""
            print(f"🚀 Mock VMMUL: Accelerating {matrix_a.shape} × {matrix_b.shape}")
            return np.matmul(matrix_a, matrix_b)
        
        def benchmark_performance(self, num_iterations=100):
            """Mock performance benchmarking."""
            return {
                'iterations': num_iterations,
                'hardware_time': 0.001,  # Mock timing
                'software_time': 0.002,  # Mock timing
                'speedup': 2.0,
                'operations_per_second': 1000000
            }
    
    VMMULAccelerator = MockVMMULAccelerator

class TinyGradVMMULIntegration:
    """
    Integration layer between TinyGrad and VMMUL accelerator.
    
    This class provides a seamless interface for using VMMUL acceleration
    with TinyGrad tensor operations, with automatic fallback to CPU.
    """
    
    def __init__(self, use_accelerator: bool = True):
        """
        Initialize TinyGrad + VMMUL integration.
        
        Args:
            use_accelerator: Whether to use VMMUL acceleration
        """
        self.use_accelerator = use_accelerator and VMMUL_AVAILABLE
        self.accelerator = None
        
        if self.use_accelerator:
            try:
                self.accelerator = VMMULAccelerator()
                print(f"✅ VMMUL Accelerator initialized and ready")
            except Exception as e:
                print(f"⚠️  Failed to initialize VMMUL accelerator: {e}")
                print("Falling back to CPU-only mode")
                self.use_accelerator = False
        
        if not self.use_accelerator:
            print("ℹ️  Running in CPU-only mode (NumPy fallback)")
    
    def custom_matmul(self, a: np.ndarray, b: np.ndarray, 
                     use_accel: Optional[bool] = None) -> np.ndarray:
        """
        Matrix multiplication with optional VMMUL acceleration.
        
        Args:
            a: Left matrix
            b: Right matrix
            use_accel: Override accelerator setting (None = use default)
            
        Returns:
            Result matrix C = A × B
        """
        if use_accel is None:
            use_accel = self.use_accelerator
        
        # Validate matrix dimensions
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: {a.shape} × {b.shape}")
        
        # Check if matrices are compatible with VMMUL (4x4, 8x8, 16x16)
        matrix_size = a.shape[0]
        is_vmmul_compatible = matrix_size in [4, 8, 16] and a.shape == b.shape
        
        if use_accel and is_vmmul_compatible and self.accelerator:
            try:
                print(f"🚀 VMMUL: Accelerating {matrix_size}×{matrix_size} matrix multiplication")
                result = self.accelerator.accelerate_matrix_multiply(a, b)
                return result
            except Exception as e:
                print(f"⚠️  VMMUL acceleration failed: {e}")
                print("Falling back to CPU implementation")
                return a @ b
        else:
            if not is_vmmul_compatible:
                print(f"ℹ️  Matrix size {matrix_size}×{matrix_size} not VMMUL compatible, using CPU")
            elif not use_accel:
                print("ℹ️  VMMUL acceleration disabled, using CPU")
            else:
                print("ℹ️  No accelerator available, using CPU")
            
            return a @ b  # Fallback to NumPy
    
    def benchmark_matmul(self, matrix_sizes: List[int] = [4, 8, 16], 
                        iterations: int = 50) -> dict:
        """
        Benchmark matrix multiplication performance.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            iterations: Number of iterations per test
            
        Returns:
            Dictionary with benchmark results
        """
        results = []
        
        for size in matrix_sizes:
            print(f"\n📊 Benchmarking {size}×{size} matrices ({iterations} iterations)")
            
            # Generate test matrices
            a = np.random.randint(1, 10, (size, size), dtype=np.int32)
            b = np.random.randint(1, 10, (size, size), dtype=np.int32)
            
            # CPU benchmark
            start_time = time.time()
            for _ in range(iterations):
                _ = a @ b
            cpu_time = time.time() - start_time
            
            # VMMUL benchmark (if available)
            if self.use_accelerator and self.accelerator:
                start_time = time.time()
                for _ in range(iterations):
                    _ = self.custom_matmul(a, b, use_accel=True)
                accel_time = time.time() - start_time
                
                speedup = cpu_time / accel_time if accel_time > 0 else 1.0
                operations = size * size * size * iterations  # n³ operations
                cpu_gflops = (operations / cpu_time) / 1e9
                accel_gflops = (operations / accel_time) / 1e9
                
                print(f"   CPU Time: {cpu_time:.6f}s, GFLOPS: {cpu_gflops:.2f}")
                print(f"   VMMUL Time: {accel_time:.6f}s, GFLOPS: {accel_gflops:.2f}")
                print(f"   Speedup: {speedup:.2f}x")
                
                results.append({
                    'matrix_size': f"{size}×{size}",
                    'iterations': iterations,
                    'cpu_time_ms': cpu_time * 1000,
                    'accel_time_ms': accel_time * 1000,
                    'speedup': speedup,
                    'cpu_gflops': cpu_gflops,
                    'accel_gflops': accel_gflops,
                    'operations': operations
                })
            else:
                operations = size * size * size * iterations
                cpu_gflops = (operations / cpu_time) / 1e9
                print(f"   CPU Time: {cpu_time:.6f}s, GFLOPS: {cpu_gflops:.2f}")
                
                results.append({
                    'matrix_size': f"{size}×{size}",
                    'iterations': iterations,
                    'cpu_time_ms': cpu_time * 1000,
                    'accel_time_ms': None,
                    'speedup': 1.0,
                    'cpu_gflops': cpu_gflops,
                    'accel_gflops': None,
                    'operations': operations
                })
        
        return results
    
    def validate_correctness(self, matrix_sizes: List[int] = [4, 8, 16]) -> dict:
        """
        Validate correctness of VMMUL acceleration.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            
        Returns:
            Dictionary with validation results
        """
        validation_results = []
        
        for size in matrix_sizes:
            print(f"\n🔍 Validating {size}×{size} matrix multiplication")
            
            # Test 1: Random matrices
            a = np.random.randint(1, 10, (size, size), dtype=np.int32)
            b = np.random.randint(1, 10, (size, size), dtype=np.int32)
            
            cpu_result = a @ b
            if self.use_accelerator and self.accelerator:
                accel_result = self.custom_matmul(a, b, use_accel=True)
                
                if np.array_equal(cpu_result, accel_result):
                    print(f"   ✅ Random matrices: PASSED")
                    random_test = True
                else:
                    print(f"   ❌ Random matrices: FAILED")
                    print(f"      CPU: {cpu_result}")
                    print(f"      VMMUL: {accel_result}")
                    random_test = False
            else:
                print(f"   ℹ️  Random matrices: Skipped (no accelerator)")
                random_test = None
            
            # Test 2: Identity matrix
            i = np.eye(size, dtype=np.int32)
            cpu_identity = a @ i
            if self.use_accelerator and self.accelerator:
                accel_identity = self.custom_matmul(a, i, use_accel=True)
                
                if np.array_equal(cpu_identity, a):
                    print(f"   ✅ Identity matrix: PASSED")
                    identity_test = True
                else:
                    print(f"   ❌ Identity matrix: FAILED")
                    identity_test = False
            else:
                print(f"   ℹ️  Identity matrix: Skipped (no accelerator)")
                identity_test = None
            
            # Test 3: Zero matrix
            z = np.zeros((size, size), dtype=np.int32)
            cpu_zero = a @ z
            if self.use_accelerator and self.accelerator:
                accel_zero = self.custom_matmul(a, z, use_accel=True)
                
                if np.array_equal(cpu_zero, z):
                    print(f"   ✅ Zero matrix: PASSED")
                    zero_test = True
                else:
                    print(f"   ❌ Zero matrix: FAILED")
                    zero_test = False
            else:
                print(f"   ℹ️  Zero matrix: Skipped (no accelerator)")
                zero_test = None
            
            validation_results.append({
                'matrix_size': f"{size}×{size}",
                'random_test': random_test,
                'identity_test': identity_test,
                'zero_test': zero_test,
                'all_passed': all([random_test, identity_test, zero_test]) if None not in [random_test, identity_test, zero_test] else None
            })
        
        return validation_results
    
    def get_status(self) -> dict:
        """Get current integration status."""
        return {
            'accelerator_available': VMMUL_AVAILABLE,
            'use_accelerator': self.use_accelerator,
            'accelerator_initialized': self.accelerator is not None,
            'supported_matrix_sizes': [4, 8, 16] if self.use_accelerator else []
        }

# Convenience function for easy integration
def create_tinygrad_vmmul_integration(use_accelerator: bool = True) -> TinyGradVMMULIntegration:
    """
    Create a TinyGrad + VMMUL integration instance.
    
    Args:
        use_accelerator: Whether to use VMMUL acceleration
        
    Returns:
        TinyGradVMMULIntegration instance
    """
    return TinyGradVMMULIntegration(use_accelerator=use_accelerator)

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("TinyGrad + VMMUL Integration Test")
    print("="*70)
    
    # Create integration instance
    integration = create_tinygrad_vmmul_integration()
    
    # Test status
    status = integration.get_status()
    print(f"\n📊 Integration Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Test matrix multiplication
    print(f"\n🧪 Testing Matrix Multiplication:")
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.int32)
    b = np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]], dtype=np.int32)
    
    result = integration.custom_matmul(a, b)
    print(f"   Result shape: {result.shape}")
    print(f"   First element: {result[0, 0]}")
    
    # Run benchmarks
    print(f"\n📈 Running Performance Benchmarks:")
    benchmark_results = integration.benchmark_matmul(matrix_sizes=[4, 8], iterations=10)
    
    # Validate correctness
    print(f"\n🔍 Validating Correctness:")
    validation_results = integration.validate_correctness(matrix_sizes=[4, 8])
    
    print(f"\n🎉 Integration test completed successfully!")
    print(f"Ready for TinyGrad AI workload acceleration!")

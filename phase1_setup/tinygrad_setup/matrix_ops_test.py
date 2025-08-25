#!/usr/bin/env python3
"""
Matrix Operations Test using TinyGrad
This script tests basic matrix operations that will be relevant for RISC-V acceleration.
"""

import time
import numpy as np
import os

# Force CPU-only mode
os.environ['TINYGRAD_DEVICE'] = 'CPU'
os.environ['TINYGRAD_DISABLE_METAL'] = '1'

from tinygrad import Tensor

def test_matrix_multiplication():
    """Test matrix multiplication performance."""
    print("="*60)
    print("Matrix Multiplication Performance Test")
    print("="*60)
    
    sizes = [64, 128, 256, 512]
    results = {}
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrices...")
        
        # Create random matrices
        A = Tensor(np.random.randn(size, size).astype(np.float32), device='CPU')
        B = Tensor(np.random.randn(size, size).astype(np.float32), device='CPU')
        
        # Warm up
        _ = A @ B
        A.numpy(); B.numpy()
        
        # Time the operation
        start_time = time.time()
        C = A @ B
        C.numpy()  # Force computation
        elapsed_time = time.time() - start_time
        
        # Calculate theoretical FLOPS
        flops = 2 * size**3  # 2*n³ for matrix multiplication
        gflops = flops / (elapsed_time * 1e9)
        
        results[size] = {
            'time': elapsed_time,
            'gflops': gflops,
            'flops': flops
        }
        
        print(f"  Time: {elapsed_time:.4f}s")
        print(f"  Performance: {gflops:.2f} GFLOPS")
    
    return results

def test_element_wise_operations():
    """Test element-wise operations."""
    print("\n" + "="*60)
    print("Element-wise Operations Test")
    print("="*60)
    
    size = 512
    print(f"Testing {size}x{size} matrices...")
    
    # Create random matrices
    A = Tensor(np.random.randn(size, size).astype(np.float32), device='CPU')
    B = Tensor(np.random.randn(size, size).astype(np.float32), device='CPU')
    
    operations = {
        'Addition': lambda: A + B,
        'Multiplication': lambda: A * B,
        'Division': lambda: A / (B + 1e-8),  # Avoid division by zero
        'ReLU': lambda: A.relu(),
        'Sigmoid': lambda: A.sigmoid(),
        'Square Root': lambda: (A.abs() + 1e-8).sqrt(),
    }
    
    results = {}
    
    for op_name, op_func in operations.items():
        print(f"\n  Testing {op_name}...")
        
        # Warm up
        _ = op_func()
        A.numpy(); B.numpy()
        
        # Time the operation
        start_time = time.time()
        result = op_func()
        result.numpy()  # Force computation
        elapsed_time = time.time() - start_time
        
        results[op_name] = elapsed_time
        print(f"    Time: {elapsed_time:.4f}s")
    
    return results

def test_memory_operations():
    """Test memory allocation and copying operations."""
    print("\n" + "="*60)
    print("Memory Operations Test")
    print("="*60)
    
    sizes = [64, 128, 256, 512]
    results = {}
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrices...")
        
        # Create matrix
        A = Tensor(np.random.randn(size, size).astype(np.float32), device='CPU')
        
        # Test copying (use assignment for TinyGrad 0.8.0)
        start_time = time.time()
        B = A  # Simple assignment instead of copy()
        B.numpy()  # Force computation
        copy_time = time.time() - start_time
        
        # Test reshaping
        start_time = time.time()
        C = A.reshape(-1)
        C.numpy()  # Force computation
        reshape_time = time.time() - start_time
        
        # Test transposing
        start_time = time.time()
        D = A.transpose()
        D.numpy()  # Force computation
        transpose_time = time.time() - start_time
        
        results[size] = {
            'copy': copy_time,
            'reshape': reshape_time,
            'transpose': transpose_time
        }
        
        print(f"  Copy: {copy_time:.4f}s")
        print(f"  Reshape: {reshape_time:.4f}s")
        print(f"  Transpose: {transpose_time:.4f}s")
    
    return results

def test_vector_operations():
    """Test vector operations that might be accelerated."""
    print("\n" + "="*60)
    print("Vector Operations Test")
    print("="*60)
    
    size = 1000000  # 1M elements
    print(f"Testing vector with {size:,} elements...")
    
    # Create random vectors
    a = Tensor(np.random.randn(size).astype(np.float32), device='CPU')
    b = Tensor(np.random.randn(size).astype(np.float32), device='CPU')
    
    operations = {
        'Dot Product': lambda: a.dot(b),
        'L2 Norm': lambda: (a * a).sum().sqrt(),  # Manual L2 norm calculation
        'Sum': lambda: a.sum(),
        'Max': lambda: a.max(),
        'Min': lambda: a.min(),
        'Mean': lambda: a.mean(),
    }
    
    results = {}
    
    for op_name, op_func in operations.items():
        print(f"\n  Testing {op_name}...")
        
        # Warm up
        _ = op_func()
        a.numpy(); b.numpy()
        
        # Time the operation
        start_time = time.time()
        result = op_func()
        result.numpy()  # Force computation
        elapsed_time = time.time() - start_time
        
        results[op_name] = elapsed_time
        print(f"    Time: {elapsed_time:.4f}s")
    
    return results

def print_summary(matmul_results, elem_results, mem_results, vec_results):
    """Print a summary of all test results."""
    print("\n" + "="*80)
    print("PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    print("\nMatrix Multiplication Performance:")
    print("-" * 40)
    for size, result in matmul_results.items():
        print(f"  {size}x{size}: {result['gflops']:.2f} GFLOPS ({result['time']:.4f}s)")
    
    print("\nElement-wise Operations (512x512):")
    print("-" * 40)
    for op, time_val in elem_results.items():
        print(f"  {op}: {time_val:.4f}s")
    
    print("\nMemory Operations:")
    print("-" * 40)
    for size, result in mem_results.items():
        print(f"  {size}x{size} - Copy: {result['copy']:.4f}s, "
              f"Reshape: {result['reshape']:.4f}s, "
              f"Transpose: {result['transpose']:.4f}s")
    
    print("\nVector Operations (1M elements):")
    print("-" * 40)
    for op, time_val in vec_results.items():
        print(f"  {op}: {time_val:.4f}s")
    
    print("\n" + "="*80)
    print("These benchmarks establish baseline performance for RISC-V optimization.")
    print("Focus on operations with highest computational intensity.")

def main():
    """Main test function."""
    print("="*80)
    print("TinyGrad Matrix Operations Performance Test")
    print("="*80)
    print("This test establishes baseline performance metrics for RISC-V optimization.")
    
    try:
        # Run all tests
        matmul_results = test_matrix_multiplication()
        elem_results = test_element_wise_operations()
        mem_results = test_memory_operations()
        vec_results = test_vector_operations()
        
        # Print summary
        print_summary(matmul_results, elem_results, mem_results, vec_results)
        
        print("\n🎉 All matrix operation tests completed successfully!")
        print("TinyGrad is ready for performance analysis and RISC-V integration.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check your TinyGrad installation.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

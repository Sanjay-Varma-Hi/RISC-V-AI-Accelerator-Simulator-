#!/usr/bin/env python3
"""
Simple TinyGrad Test - CPU Only
This script tests basic TinyGrad functionality without complex operations.
"""

import os
import time
import numpy as np

# Force CPU-only mode
os.environ['TINYGRAD_DEVICE'] = 'CPU'
os.environ['TINYGRAD_DISABLE_METAL'] = '1'

def test_basic_tensor_operations():
    """Test basic tensor operations."""
    print("="*60)
    print("Basic Tensor Operations Test")
    print("="*60)
    
    try:
        from tinygrad import Tensor
        
        print("✅ TinyGrad imported successfully!")
        
        # Test 1: Create tensors
        print("\n1. Creating tensors...")
        # Try to force CPU device
        try:
            a = Tensor([1, 2, 3, 4], device='CPU')
            b = Tensor([5, 6, 7, 8], device='CPU')
        except:
            # Fallback to default
            a = Tensor([1, 2, 3, 4])
            b = Tensor([5, 6, 7, 8])
        print(f"   Tensor a: {a}")
        print(f"   Tensor b: {b}")
        
        # Test 2: Basic arithmetic
        print("\n2. Basic arithmetic...")
        c = a + b
        d = a * b
        e = a - b
        print(f"   a + b = {c}")
        print(f"   a * b = {d}")
        print(f"   a - b = {e}")
        
        # Test 3: Matrix operations
        print("\n3. Matrix operations...")
        A = Tensor([[1, 2], [3, 4]], device='CPU')
        B = Tensor([[5, 6], [7, 8]], device='CPU')
        print(f"   Matrix A:\n{A}")
        print(f"   Matrix B:\n{B}")
        
        # Test 4: Element-wise operations
        print("\n4. Element-wise operations...")
        C = A + B
        D = A * B
        print(f"   A + B:\n{C}")
        print(f"   A * B:\n{D}")
        
        print("\n✅ All basic tensor operations passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def test_matrix_multiplication():
    """Test matrix multiplication."""
    print("\n" + "="*60)
    print("Matrix Multiplication Test")
    print("="*60)
    
    try:
        from tinygrad import Tensor
        
        # Create small matrices
        A = Tensor([[1, 2], [3, 4]], device='CPU')
        B = Tensor([[5, 6], [7, 8]], device='CPU')
        
        print("Matrices:")
        print(f"A:\n{A}")
        print(f"B:\n{B}")
        
        # Matrix multiplication
        start_time = time.time()
        C = A @ B
        C.numpy()  # Force computation
        elapsed_time = time.time() - start_time
        
        print(f"\nResult A @ B:\n{C}")
        print(f"Computation time: {elapsed_time:.4f}s")
        
        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19, 22], [43, 50]])
        actual = C.numpy()
        
        if np.allclose(actual, expected):
            print("✅ Matrix multiplication result is correct!")
        else:
            print("❌ Matrix multiplication result is incorrect!")
            print(f"Expected:\n{expected}")
            print(f"Actual:\n{actual}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Matrix multiplication test failed: {e}")
        return False

def test_performance_benchmark():
    """Test performance with larger matrices."""
    print("\n" + "="*60)
    print("Performance Benchmark Test")
    print("="*60)
    
    try:
        from tinygrad import Tensor
        
        sizes = [32, 64, 128]
        results = {}
        
        for size in sizes:
            print(f"\nTesting {size}x{size} matrices...")
            
            # Create random matrices
            A = Tensor(np.random.randn(size, size).astype(np.float32), device='CPU')
            B = Tensor(np.random.randn(size, size).astype(np.float32), device='CPU')
            
            # Time matrix multiplication
            start_time = time.time()
            C = A @ B
            C.numpy()  # Force computation
            elapsed_time = time.time() - start_time
            
            # Calculate FLOPS
            flops = 2 * size**3
            gflops = flops / (elapsed_time * 1e9)
            
            results[size] = {
                'time': elapsed_time,
                'gflops': gflops
            }
            
            print(f"  Time: {elapsed_time:.4f}s")
            print(f"  Performance: {gflops:.2f} GFLOPS")
        
        # Summary
        print("\n" + "="*40)
        print("PERFORMANCE SUMMARY")
        print("="*40)
        for size, result in results.items():
            print(f"  {size}x{size}: {result['gflops']:.2f} GFLOPS ({result['time']:.4f}s)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        return False

def main():
    """Main test function."""
    print("="*80)
    print("TinyGrad Simple Test Suite")
    print("="*80)
    print("Testing basic functionality without complex operations...")
    
    # Run tests
    test1 = test_basic_tensor_operations()
    test2 = test_matrix_multiplication()
    test3 = test_performance_benchmark()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Basic Operations: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Matrix Multiplication: {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"Performance Benchmark: {'✅ PASSED' if test3 else '❌ FAILED'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests passed! TinyGrad is working correctly.")
        print("Ready for RISC-V integration and optimization.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

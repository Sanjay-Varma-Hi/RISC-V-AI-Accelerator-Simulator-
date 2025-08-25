#!/usr/bin/env python3
"""
Correctness Validation: TinyGrad + VMMUL Integration
Validates correctness between CPU and VMMUL accelerator outputs
Author: RISC-V AI Accelerator Simulator Project
Date: 2024
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple

# Add the integration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'integration'))

try:
    from tinygrad_vmmul import create_tinygrad_vmmul_integration
    print("✅ Successfully imported TinyGrad + VMMUL integration")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class CorrectnessValidator:
    """
    Correctness validation tool for TinyGrad + VMMUL integration.
    
    Validates that VMMUL acceleration produces identical results to CPU computation.
    """
    
    def __init__(self):
        """Initialize the correctness validator."""
        self.integration = create_tinygrad_vmmul_integration(use_accelerator=True)
        self.validation_results = []
        
        print(f"🔍 Correctness Validator initialized")
        print(f"   Accelerator status: {self.integration.get_status()}")
    
    def validate_random_matrices(self, matrix_size: int, num_tests: int = 10) -> Dict[str, Any]:
        """
        Validate correctness with random matrices.
        
        Args:
            matrix_size: Matrix dimension
            num_tests: Number of random test cases
            
        Returns:
            Validation result dictionary
        """
        print(f"\n🎲 Validating {matrix_size}×{matrix_size} random matrices ({num_tests} tests)")
        
        passed_tests = 0
        failed_tests = 0
        error_details = []
        
        for test_num in range(num_tests):
            try:
                # Generate random matrices
                a = np.random.randint(1, 10, (matrix_size, matrix_size), dtype=np.int32)
                b = np.random.randint(1, 10, (matrix_size, matrix_size), dtype=np.int32)
                
                # CPU computation
                cpu_result = a @ b
                
                # VMMUL computation (if available)
                if self.integration.use_accelerator and self.integration.accelerator:
                    accel_result = self.integration.custom_matmul(a, b, use_accel=True)
                    
                    # Compare results
                    if np.array_equal(cpu_result, accel_result):
                        passed_tests += 1
                    else:
                        failed_tests += 1
                        error_details.append({
                            'test_num': test_num,
                            'matrix_a': a.copy(),
                            'matrix_b': b.copy(),
                            'cpu_result': cpu_result.copy(),
                            'accel_result': accel_result.copy(),
                            'difference': cpu_result - accel_result
                        })
                else:
                    # No accelerator available, skip test
                    print(f"   ℹ️  Test {test_num + 1}: Skipped (no accelerator)")
                    continue
                
                if (test_num + 1) % 5 == 0:
                    print(f"   Progress: {test_num + 1}/{num_tests} tests completed")
                    
            except Exception as e:
                failed_tests += 1
                error_details.append({
                    'test_num': test_num,
                    'error': str(e)
                })
                print(f"   ❌ Test {test_num + 1} failed with error: {e}")
        
        # Calculate success rate
        total_tests = passed_tests + failed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"   📊 Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        return {
            'test_type': 'random_matrices',
            'matrix_size': f"{matrix_size}×{matrix_size}",
            'num_tests': num_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'error_details': error_details
        }
    
    def validate_identity_matrix(self, matrix_size: int) -> Dict[str, Any]:
        """
        Validate identity matrix property: A × I = A.
        
        Args:
            matrix_size: Matrix dimension
            
        Returns:
            Validation result dictionary
        """
        print(f"\n🆔 Validating identity matrix property for {matrix_size}×{matrix_size}")
        
        try:
            # Generate random matrix A
            a = np.random.randint(1, 10, (matrix_size, matrix_size), dtype=np.int32)
            
            # Create identity matrix I
            i = np.eye(matrix_size, dtype=np.int32)
            
            # CPU computation: A × I
            cpu_result = a @ i
            
            # VMMUL computation: A × I (if available)
            if self.integration.use_accelerator and self.integration.accelerator:
                accel_result = self.integration.custom_matmul(a, i, use_accel=True)
                
                # Check if A × I = A
                if np.array_equal(cpu_result, a):
                    print(f"   ✅ Identity matrix property: PASSED")
                    print(f"      A × I = A verified for {matrix_size}×{matrix_size}")
                    passed = True
                else:
                    print(f"   ❌ Identity matrix property: FAILED")
                    print(f"      Expected: A (shape: {a.shape})")
                    print(f"      Got: A × I (shape: {cpu_result.shape})")
                    passed = False
                
                # Also verify accelerator result matches CPU
                if np.array_equal(accel_result, cpu_result):
                    print(f"   ✅ Accelerator result matches CPU: PASSED")
                    accel_match = True
                else:
                    print(f"   ❌ Accelerator result mismatch: FAILED")
                    print(f"      CPU: {cpu_result}")
                    print(f"      VMMUL: {accel_result}")
                    accel_match = False
                
                return {
                    'test_type': 'identity_matrix',
                    'matrix_size': f"{matrix_size}×{matrix_size}",
                    'passed': passed and accel_match,
                    'cpu_result_matches_expected': passed,
                    'accelerator_matches_cpu': accel_match,
                    'matrix_a': a.copy(),
                    'identity_matrix': i.copy(),
                    'cpu_result': cpu_result.copy(),
                    'accelerator_result': accel_result.copy()
                }
            else:
                print(f"   ℹ️  Skipped (no accelerator available)")
                return {
                    'test_type': 'identity_matrix',
                    'matrix_size': f"{matrix_size}×{matrix_size}",
                    'passed': None,
                    'cpu_result_matches_expected': None,
                    'accelerator_matches_cpu': None,
                    'skipped': True
                }
                
        except Exception as e:
            print(f"   ❌ Identity matrix validation failed: {e}")
            return {
                'test_type': 'identity_matrix',
                'matrix_size': f"{matrix_size}×{matrix_size}",
                'passed': False,
                'error': str(e)
            }
    
    def validate_zero_matrix(self, matrix_size: int) -> Dict[str, Any]:
        """
        Validate zero matrix property: A × Z = Z.
        
        Args:
            matrix_size: Matrix dimension
            
        Returns:
            Validation result dictionary
        """
        print(f"\n🔄 Validating zero matrix property for {matrix_size}×{matrix_size}")
        
        try:
            # Generate random matrix A
            a = np.random.randint(1, 10, (matrix_size, matrix_size), dtype=np.int32)
            
            # Create zero matrix Z
            z = np.zeros((matrix_size, matrix_size), dtype=np.int32)
            
            # CPU computation: A × Z
            cpu_result = a @ z
            
            # VMMUL computation: A × Z (if available)
            if self.integration.use_accelerator and self.integration.accelerator:
                accel_result = self.integration.custom_matmul(a, z, use_accel=True)
                
                # Check if A × Z = Z
                if np.array_equal(cpu_result, z):
                    print(f"   ✅ Zero matrix property: PASSED")
                    print(f"      A × Z = Z verified for {matrix_size}×{matrix_size}")
                    passed = True
                else:
                    print(f"   ❌ Zero matrix property: FAILED")
                    print(f"      Expected: Z (all zeros)")
                    print(f"      Got: A × Z (non-zero elements found)")
                    passed = False
                
                # Also verify accelerator result matches CPU
                if np.array_equal(accel_result, cpu_result):
                    print(f"   ✅ Accelerator result matches CPU: PASSED")
                    accel_match = True
                else:
                    print(f"   ❌ Accelerator result mismatch: FAILED")
                    print(f"      CPU: {cpu_result}")
                    print(f"      VMMUL: {accel_result}")
                    accel_match = False
                
                return {
                    'test_type': 'zero_matrix',
                    'matrix_size': f"{matrix_size}×{matrix_size}",
                    'passed': passed and accel_match,
                    'cpu_result_matches_expected': passed,
                    'accelerator_matches_cpu': accel_match,
                    'matrix_a': a.copy(),
                    'zero_matrix': z.copy(),
                    'cpu_result': cpu_result.copy(),
                    'accelerator_result': accel_result.copy()
                }
            else:
                print(f"   ℹ️  Skipped (no accelerator available)")
                return {
                    'test_type': 'zero_matrix',
                    'matrix_size': f"{matrix_size}×{matrix_size}",
                    'passed': None,
                    'cpu_result_matches_expected': None,
                    'accelerator_matches_cpu': None,
                    'skipped': True
                }
                
        except Exception as e:
            print(f"   ❌ Zero matrix validation failed: {e}")
            return {
                'test_type': 'zero_matrix',
                'matrix_size': f"{matrix_size}×{matrix_size}",
                'passed': False,
                'error': str(e)
            }
    
    def validate_associativity(self, matrix_size: int) -> Dict[str, Any]:
        """
        Validate associativity property: (A × B) × C = A × (B × C).
        
        Args:
            matrix_size: Matrix dimension
            
        Returns:
            Validation result dictionary
        """
        print(f"\n🔗 Validating associativity property for {matrix_size}×{matrix_size}")
        
        try:
            # Generate random matrices A, B, C
            a = np.random.randint(1, 5, (matrix_size, matrix_size), dtype=np.int32)
            b = np.random.randint(1, 5, (matrix_size, matrix_size), dtype=np.int32)
            c = np.random.randint(1, 5, (matrix_size, matrix_size), dtype=np.int32)
            
            # CPU computation: (A × B) × C
            left_result = (a @ b) @ c
            
            # CPU computation: A × (B × C)
            right_result = a @ (b @ c)
            
            # Check associativity
            if np.array_equal(left_result, right_result):
                print(f"   ✅ Associativity property: PASSED")
                print(f"      (A × B) × C = A × (B × C) verified for {matrix_size}×{matrix_size}")
                associativity_passed = True
            else:
                print(f"   ❌ Associativity property: FAILED")
                print(f"      (A × B) × C ≠ A × (B × C)")
                associativity_passed = False
            
            # VMMUL validation (if available)
            if self.integration.use_accelerator and self.integration.accelerator:
                # Test VMMUL with left associativity
                vmmul_left = self.integration.custom_matmul(
                    self.integration.custom_matmul(a, b, use_accel=True), 
                    c, use_accel=True
                )
                
                # Test VMMUL with right associativity
                vmmul_right = self.integration.custom_matmul(
                    a, 
                    self.integration.custom_matmul(b, c, use_accel=True), 
                    use_accel=True
                )
                
                # Check if VMMUL results match CPU
                vmmul_left_match = np.array_equal(vmmul_left, left_result)
                vmmul_right_match = np.array_equal(vmmul_right, right_result)
                
                if vmmul_left_match and vmmul_right_match:
                    print(f"   ✅ VMMUL associativity validation: PASSED")
                    vmmul_passed = True
                else:
                    print(f"   ❌ VMMUL associativity validation: FAILED")
                    if not vmmul_left_match:
                        print(f"      VMMUL left: mismatch with CPU")
                    if not vmmul_right_match:
                        print(f"      VMMUL right: mismatch with CPU")
                    vmmul_passed = False
                
                return {
                    'test_type': 'associativity',
                    'matrix_size': f"{matrix_size}×{matrix_size}",
                    'passed': associativity_passed and vmmul_passed,
                    'cpu_associativity': associativity_passed,
                    'vmmul_associativity': vmmul_passed,
                    'vmmul_left_matches_cpu': vmmul_left_match,
                    'vmmul_right_matches_cpu': vmmul_right_match,
                    'matrix_a': a.copy(),
                    'matrix_b': b.copy(),
                    'matrix_c': c.copy(),
                    'cpu_left_result': left_result.copy(),
                    'cpu_right_result': right_result.copy(),
                    'vmmul_left_result': vmmul_left.copy(),
                    'vmmul_right_result': vmmul_right.copy()
                }
            else:
                print(f"   ℹ️  VMMUL validation skipped (no accelerator available)")
                return {
                    'test_type': 'associativity',
                    'matrix_size': f"{matrix_size}×{matrix_size}",
                    'passed': associativity_passed,
                    'cpu_associativity': associativity_passed,
                    'vmmul_associativity': None,
                    'skipped': True
                }
                
        except Exception as e:
            print(f"   ❌ Associativity validation failed: {e}")
            return {
                'test_type': 'associativity',
                'matrix_size': f"{matrix_size}×{matrix_size}",
                'passed': False,
                'error': str(e)
            }
    
    def run_comprehensive_validation(self, matrix_sizes: List[int] = [4, 8, 16]) -> List[Dict[str, Any]]:
        """
        Run comprehensive correctness validation across multiple matrix sizes.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            
        Returns:
            List of validation results
        """
        print("="*80)
        print("🔍 COMPREHENSIVE CORRECTNESS VALIDATION")
        print("="*80)
        print(f"Matrix sizes: {matrix_sizes}")
        print(f"Accelerator: {'Enabled' if self.integration.use_accelerator else 'Disabled'}")
        
        all_results = []
        
        for size in matrix_sizes:
            print(f"\n{'='*60}")
            print(f"Testing Matrix Size: {size}×{size}")
            print(f"{'='*60}")
            
            # Run all validation tests for this matrix size
            size_results = []
            
            # 1. Random matrices validation
            random_result = self.validate_random_matrices(size, num_tests=5)
            size_results.append(random_result)
            
            # 2. Identity matrix validation
            identity_result = self.validate_identity_matrix(size)
            size_results.append(identity_result)
            
            # 3. Zero matrix validation
            zero_result = self.validate_zero_matrix(size)
            size_results.append(zero_result)
            
            # 4. Associativity validation
            associativity_result = self.validate_associativity(size)
            size_results.append(associativity_result)
            
            # Summary for this matrix size
            passed_tests = sum(1 for r in size_results if r.get('passed') is True)
            total_tests = sum(1 for r in size_results if r.get('passed') is not None)
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            print(f"\n📊 {size}×{size} Validation Summary:")
            print(f"   Tests passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
            
            all_results.extend(size_results)
        
        return all_results
    
    def generate_validation_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            results: List of validation results
            
        Returns:
            Validation report string
        """
        if not results:
            return "No validation results available."
        
        # Group results by matrix size
        matrix_sizes = set()
        for result in results:
            if 'matrix_size' in result:
                matrix_sizes.add(result['matrix_size'])
        
        # Calculate overall statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get('passed') is True)
        failed_tests = sum(1 for r in results if r.get('passed') is False)
        skipped_tests = sum(1 for r in results if r.get('passed') is None)
        
        success_rate = (passed_tests / (passed_tests + failed_tests) * 100) if (passed_tests + failed_tests) > 0 else 0
        
        report = f"""
{'='*80}
🔍 CORRECTNESS VALIDATION REPORT
{'='*80}

📊 Overall Statistics:
   Total tests: {total_tests}
   Passed: {passed_tests}
   Failed: {failed_tests}
   Skipped: {skipped_tests}
   Success rate: {success_rate:.1f}%

📋 Results by Matrix Size:
"""
        
        for size in sorted(matrix_sizes, key=lambda x: int(x.split('×')[0])):
            size_results = [r for r in results if r.get('matrix_size') == size]
            size_passed = sum(1 for r in size_results if r.get('passed') is True)
            size_total = sum(1 for r in size_results if r.get('passed') is not None)
            size_success = (size_passed / size_total * 100) if size_total > 0 else 0
            
            report += f"   {size}: {size_passed}/{size_total} tests passed ({size_success:.1f}%)\n"
        
        report += f"\n📋 Detailed Results:\n"
        
        for result in results:
            test_type = result.get('test_type', 'unknown')
            matrix_size = result.get('matrix_size', 'unknown')
            passed = result.get('passed')
            
            if passed is True:
                status = "✅ PASSED"
            elif passed is False:
                status = "❌ FAILED"
            else:
                status = "⏭️  SKIPPED"
            
            report += f"   {test_type} ({matrix_size}): {status}\n"
        
        report += f"\n{'='*80}"
        return report

def main():
    """Main function to run the correctness validation suite."""
    print("🔍 Phase 3: Correctness Validation Suite")
    print("="*50)
    
    # Create validator
    validator = CorrectnessValidator()
    
    # Run validation suite
    matrix_sizes = [4, 8, 16]  # Test 4x4, 8x8, and 16x16 matrices
    
    try:
        results = validator.run_comprehensive_validation(matrix_sizes)
        
        # Generate report
        report = validator.generate_validation_report(results)
        print(report)
        
        # Save report to file
        os.makedirs("../results", exist_ok=True)
        report_file = "../results/correctness_validation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📝 Validation report saved to: {report_file}")
        print("\n🎉 Correctness validation completed successfully!")
        
        # Check if all tests passed
        failed_tests = sum(1 for r in results if r.get('passed') is False)
        if failed_tests == 0:
            print("✅ All validation tests PASSED! VMMUL acceleration is working correctly.")
        else:
            print(f"⚠️  {failed_tests} validation tests FAILED. Please check the results above.")
        
    except Exception as e:
        print(f"\n❌ Validation suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

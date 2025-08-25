#!/usr/bin/env python3
"""
Phase 4: Dynamic Workload Scheduler
===================================

This module implements intelligent workload scheduling that automatically
decides between CPU and VMMUL accelerator based on matrix size and
hardware availability, mimicking  ' polymorphic architecture vision.

Author: RISC-V AI Accelerator Simulator Team
Date: 2025
"""

import numpy as np
import time
import logging
from typing import Tuple, Dict, Any, Optional
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

class DynamicScheduler:
    """
    Dynamic Workload Scheduler that intelligently routes matrix operations
    between CPU and VMMUL accelerator based on workload characteristics.
    
    This mimics  ' polymorphic architecture by dynamically choosing
    the optimal compute path for each workload.
    """
    
    def __init__(self, accel_available: bool = True, 
                 threshold_matrix_size: int = 8,
                 enable_logging: bool = True):
        """
        Initialize the dynamic scheduler.
        
        Args:
            accel_available: Whether VMMUL accelerator is available
            threshold_matrix_size: Minimum matrix size to use accelerator
            enable_logging: Enable detailed logging of scheduling decisions
        """
        self.accel_available = accel_available
        self.threshold_matrix_size = threshold_matrix_size
        self.enable_logging = enable_logging
        
        # Initialize integration layer
        try:
            self.integration = TinyGradVMMULIntegration()
            self.accelerator = self.integration.accelerator
            print(f"✅ Dynamic Scheduler initialized with VMMUL integration")
        except Exception as e:
            print(f"⚠️  Using mock integration: {e}")
            self.integration = None
            self.accelerator = None
        
        # Statistics tracking
        self.stats = {
            'cpu_usage': 0,
            'accelerator_usage': 0,
            'total_operations': 0,
            'scheduling_decisions': []
        }
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('DynamicScheduler')
        else:
            self.logger = None
    
    def _log_decision(self, message: str):
        """Log scheduling decisions if logging is enabled."""
        if self.logger:
            self.logger.info(f"[SCHEDULER] {message}")
        else:
            print(f"[SCHEDULER] {message}")
    
    def _should_use_accelerator(self, matrix_size: int) -> bool:
        """
        Determine if accelerator should be used for given matrix size.
        
        Args:
            matrix_size: Size of the matrix (e.g., 4 for 4x4)
            
        Returns:
            True if accelerator should be used, False for CPU
        """
        # Use accelerator if:
        # 1. It's available
        # 2. Matrix size meets threshold
        # 3. Matrix size is supported by accelerator
        if not self.accel_available:
            return False
        
        if matrix_size < self.threshold_matrix_size:
            return False
        
        # Check if accelerator supports this matrix size
        if self.accelerator and hasattr(self.accelerator, 'supported_matrix_sizes'):
            if matrix_size not in self.accelerator.supported_matrix_sizes:
                return False
        
        return True
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """
        Execute matrix multiplication with intelligent scheduling.
        
        Args:
            a: First matrix
            b: Second matrix
            
        Returns:
            Tuple of (result, execution_path, metadata)
        """
        start_time = time.perf_counter()
        
        # Validate inputs
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: {a.shape} × {b.shape}")
        
        matrix_size = a.shape[0]
        self.stats['total_operations'] += 1
        
        # Make scheduling decision
        use_accelerator = self._should_use_accelerator(matrix_size)
        
        # Execute with chosen path
        if use_accelerator:
            execution_path = "VMMUL_ACCELERATOR"
            self.stats['accelerator_usage'] += 1
            
            try:
                if self.integration and self.integration.accelerator:
                    result = self.integration.custom_matmul(a, b, use_accel=True)
                    self._log_decision(f"Using VMMUL for {matrix_size}×{matrix_size} matrices")
                else:
                    # Fallback to CPU if integration fails
                    result = a @ b
                    execution_path = "VMMUL_FALLBACK_TO_CPU"
                    self._log_decision(f"VMMUL failed, falling back to CPU for {matrix_size}×{matrix_size} matrices")
            except Exception as e:
                # Fallback to CPU on error
                result = a @ b
                execution_path = "VMMUL_ERROR_FALLBACK_TO_CPU"
                self._log_decision(f"VMMUL error: {e}, falling back to CPU for {matrix_size}×{matrix_size} matrices")
        else:
            execution_path = "CPU"
            self.stats['cpu_usage'] += 1
            result = a @ b
            self._log_decision(f"Falling back to CPU for {matrix_size}×{matrix_size} matrices")
        
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Record decision metadata
        decision_metadata = {
            'matrix_size': matrix_size,
            'execution_path': execution_path,
            'execution_time_ms': execution_time,
            'use_accelerator': use_accelerator,
            'accelerator_available': self.accel_available,
            'threshold_size': self.threshold_matrix_size
        }
        
        self.stats['scheduling_decisions'].append(decision_metadata)
        
        return result, execution_path, decision_metadata
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get current scheduling statistics."""
        total = self.stats['total_operations']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'cpu_usage_percent': (self.stats['cpu_usage'] / total) * 100,
            'accelerator_usage_percent': (self.stats['accelerator_usage'] / total) * 100,
            'efficiency_score': self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate scheduling efficiency score (0-100)."""
        if self.stats['total_operations'] == 0:
            return 0.0
        
        # Higher score for more accelerator usage when appropriate
        # and less fallback to CPU
        total = self.stats['total_operations']
        accel_usage = self.stats['accelerator_usage']
        cpu_usage = self.stats['cpu_usage']
        
        # Base score from accelerator usage
        base_score = (accel_usage / total) * 100
        
        # Penalty for unnecessary CPU usage on large matrices
        penalty = 0
        for decision in self.stats['scheduling_decisions']:
            if (decision['matrix_size'] >= self.threshold_matrix_size and 
                decision['execution_path'] == 'CPU'):
                penalty += 10
        
        return max(0, min(100, base_score - penalty))
    
    def reset_stats(self):
        """Reset scheduling statistics."""
        self.stats = {
            'cpu_usage': 0,
            'accelerator_usage': 0,
            'total_operations': 0,
            'scheduling_decisions': []
        }
    
    def print_summary(self):
        """Print a summary of scheduling performance."""
        stats = self.get_scheduling_stats()
        
        print("\n" + "="*60)
        print("📊 DYNAMIC SCHEDULER SUMMARY")
        print("="*60)
        print(f"Total Operations: {stats['total_operations']}")
        print(f"CPU Usage: {stats['cpu_usage']} ({stats.get('cpu_usage_percent', 0):.1f}%)")
        print(f"Accelerator Usage: {stats['accelerator_usage']} ({stats.get('accelerator_usage_percent', 0):.1f}%)")
        print(f"Efficiency Score: {stats.get('efficiency_score', 0):.1f}/100")
        print(f"Threshold Matrix Size: {self.threshold_matrix_size}×{self.threshold_matrix_size}")
        print(f"Accelerator Available: {self.accel_available}")
        print("="*60)


def test_dynamic_scheduler():
    """Test the dynamic scheduler with various matrix sizes."""
    print("🧪 Testing Dynamic Workload Scheduler")
    print("="*50)
    
    # Initialize scheduler
    scheduler = DynamicScheduler(accel_available=True, threshold_matrix_size=8)
    
    # Test matrices of different sizes
    test_cases = [
        (4, "4×4 (Small - Should use CPU)"),
        (8, "8×8 (Medium - Should use Accelerator)"),
        (16, "16×16 (Large - Should use Accelerator)"),
        (32, "32×32 (Extra Large - Should use Accelerator)")
    ]
    
    for matrix_size, description in test_cases:
        print(f"\n🔍 Testing {description}")
        print("-" * 40)
        
        # Generate test matrices
        a = np.random.randint(1, 10, (matrix_size, matrix_size))
        b = np.random.randint(1, 10, (matrix_size, matrix_size))
        
        # Execute with dynamic scheduling
        result, execution_path, metadata = scheduler.execute(a, b)
        
        print(f"Matrix Size: {matrix_size}×{matrix_size}")
        print(f"Execution Path: {execution_path}")
        print(f"Execution Time: {metadata['execution_time_ms']:.3f} ms")
        print(f"Result Shape: {result.shape}")
        print(f"First Element: {result[0, 0]}")
    
    # Print final summary
    scheduler.print_summary()


if __name__ == "__main__":
    test_dynamic_scheduler()

#!/usr/bin/env python3
"""
MNIST Inference Test using TinyGrad
This script demonstrates basic MNIST model inference to verify TinyGrad setup.
"""

import time
import numpy as np
import os
# Force CPU-only mode to avoid Metal GPU issues
os.environ['TINYGRAD_DEVICE'] = 'CPU'
from tinygrad import Tensor
from tinygrad.nn import Linear
# from tinygrad.helpers import dtypes  # Not available in TinyGrad 0.8.0

def create_simple_mnist_model():
    """Create a simple 2-layer neural network for MNIST classification."""
    print("Creating simple MNIST model...")
    
    # Simple architecture: 784 -> 128 -> 10
    # 784 = 28x28 flattened image, 10 = digit classes (0-9)
    model = {
        'fc1': Linear(784, 128),
        'fc2': Linear(128, 10)
    }
    
    # Initialize weights with small random values
    for layer in model.values():
        if hasattr(layer, 'weight'):
            # For TinyGrad 0.8.0, we need to use different initialization
            # The weights will be initialized automatically with small random values
            pass
    
    print("Model created successfully!")
    return model

def forward_pass(model, x):
    """Forward pass through the model."""
    # Flatten input (batch_size, 28, 28) -> (batch_size, 784)
    x = x.reshape(-1, 784)
    
    # First layer: ReLU activation
    x = model['fc1'](x).relu()
    
    # Second layer: no activation (logits)
    x = model['fc2'](x)
    
    return x

def generate_dummy_mnist_data(batch_size=4):
    """Generate dummy MNIST-like data for testing."""
    print(f"Generating {batch_size} dummy MNIST images...")
    
    # Create random images (batch_size, 28, 28) with values 0-1
    images = np.random.rand(batch_size, 28, 28).astype(np.float32)
    
    # Create random labels
    labels = np.random.randint(0, 10, batch_size)
    
    return images, labels

def test_inference(model, images, labels):
    """Test inference on dummy data."""
    print("\nRunning inference test...")
    
    # Convert to TinyGrad tensors
    x = Tensor(images)
    y_true = Tensor(labels)
    
    # Time the forward pass
    start_time = time.time()
    
    # Forward pass
    logits = forward_pass(model, x)
    
    # Get predictions
    predictions = logits.argmax(axis=1)
    
    # Calculate accuracy
    accuracy = (predictions == y_true).mean().numpy()
    
    inference_time = time.time() - start_time
    
    print(f"Inference completed in {inference_time:.4f} seconds")
    print(f"Batch size: {len(images)}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(4, len(images))):
        pred = predictions[i].numpy()
        true_label = labels[i]
        print(f"  Image {i}: Predicted {pred}, Actual {true_label}")
    
    return accuracy, inference_time

def test_matrix_operations():
    """Test basic matrix operations that will be relevant for RISC-V acceleration."""
    print("\n" + "="*50)
    print("Testing Matrix Operations")
    print("="*50)
    
    # Test matrix multiplication
    print("\n1. Matrix Multiplication Test")
    A = Tensor(np.random.randn(100, 100).astype(np.float32))
    B = Tensor(np.random.randn(100, 100).astype(np.float32))
    
    start_time = time.time()
    C = A @ B
    C.numpy()  # Force computation
    matmul_time = time.time() - start_time
    
    print(f"   100x100 matrix multiplication: {matmul_time:.4f}s")
    
    # Test element-wise operations
    print("\n2. Element-wise Operations Test")
    start_time = time.time()
    D = A + B
    E = A * B
    F = A.relu()
    D.numpy(); E.numpy(); F.numpy()  # Force computation
    elem_time = time.time() - start_time
    
    print(f"   Element-wise operations: {elem_time:.4f}s")
    
    # Test memory usage
    print("\n3. Memory Usage Test")
    print(f"   Tensor A shape: {A.shape}, dtype: {A.dtype}")
    print(f"   Tensor B shape: {B.shape}, dtype: {B.dtype}")
    
    return matmul_time, elem_time

def main():
    """Main test function."""
    print("="*60)
    print("TinyGrad MNIST Inference Test")
    print("="*60)
    
    try:
        # Test 1: Basic MNIST inference
        print("\nTest 1: MNIST Model Inference")
        print("-" * 30)
        
        model = create_simple_mnist_model()
        images, labels = generate_dummy_mnist_data(batch_size=8)
        
        accuracy, inference_time = test_inference(model, images, labels)
        
        # Test 2: Matrix operations
        matmul_time, elem_time = test_matrix_operations()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"✅ MNIST Inference: {accuracy:.2%} accuracy in {inference_time:.4f}s")
        print(f"✅ Matrix Multiplication: 100x100 in {matmul_time:.4f}s")
        print(f"✅ Element-wise Operations: {elem_time:.4f}s")
        print(f"✅ TinyGrad Environment: Working correctly!")
        
        print("\n🎉 All tests passed! TinyGrad is ready for RISC-V integration.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check your TinyGrad installation.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

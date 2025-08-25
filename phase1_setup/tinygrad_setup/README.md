# TinyGrad Setup and Testing

This directory contains TinyGrad installation instructions and test scripts for the RISC-V AI Accelerator Simulator project.

## TinyGrad Overview

TinyGrad is a lightweight deep learning framework written in Python. It's perfect for:
- Understanding neural network internals
- Custom hardware acceleration research
- Educational purposes
- Prototyping AI workloads

## Installation on macOS

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone TinyGrad Repository
```bash
cd riscv-ai-accelerator/phase1_setup/tinygrad_setup
git clone https://github.com/geohot/tinygrad.git
cd tinygrad
```

### Step 2: Install Dependencies
```bash
# Install TinyGrad and its dependencies
pip install -e .

# Alternative: Install from requirements
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
# Test basic import
python -c "import tinygrad; print('TinyGrad imported successfully!')"

# Check version
python -c "import tinygrad; print(f'TinyGrad version: {tinygrad.__version__}')"
```

## Test Scripts

### 1. Basic MNIST Inference (`mnist_test.py`)
A simple script that loads a pre-trained MNIST model and runs inference on test images.

### 2. Matrix Operations Test (`matrix_ops_test.py`)
Tests basic matrix operations that will be relevant for RISC-V acceleration.

### 3. Performance Benchmark (`benchmark.py`)
Basic performance testing to establish baseline metrics.

## Running the Tests

### MNIST Test
```bash
cd tinygrad
python ../mnist_test.py
```

Expected output:
- Model loading confirmation
- Inference results on test images
- Accuracy metrics

### Matrix Operations Test
```bash
python ../matrix_ops_test.py
```

Expected output:
- Matrix multiplication results
- Timing information
- Memory usage

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the tinygrad directory when running tests
2. **Memory Issues**: MNIST model requires ~50MB RAM
3. **Python Version**: Ensure Python 3.8+ is installed

### Performance Notes

- First run may be slower due to JIT compilation
- Apple Silicon M1/M2 should provide good performance
- GPU acceleration not required for basic testing

## Next Steps

After successful testing:
1. Analyze TinyGrad's matrix operations
2. Identify bottlenecks for RISC-V optimization
3. Design custom instructions for common operations
4. Plan integration with RISC-V simulator

## Resources

- [TinyGrad GitHub](https://github.com/geohot/tinygrad)
- [TinyGrad Documentation](https://tinygrad.org/)
- [RISC-V Specification](https://riscv.org/technical/specifications/)

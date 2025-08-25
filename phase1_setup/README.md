# Phase 1: Foundation Setup

This directory contains everything you need to set up the RISC-V AI Accelerator Simulator foundation.

## 🚀 Quick Start Guide

### 1. Install RISC-V Simulator (Ripes)

**Note:** Ripes is not available in Homebrew. Use one of these methods:

**Method A: Download from GitHub (Recommended)**
1. Visit: https://github.com/mortbopet/Ripes/releases
2. Download the latest `.dmg` file for macOS
3. Open the `.dmg` file and drag Ripes to Applications folder
4. Launch Ripes from Applications

**Method B: Build from Source**
```bash
git clone https://github.com/mortbopet/Ripes.git
cd Ripes
mkdir build && cd build
cmake ..
make
```

### 2. Set Up TinyGrad Environment

```bash
# Navigate to the tinygrad setup directory
cd riscv-ai-accelerator/phase1_setup/tinygrad_setup

# Clone TinyGrad repository
git clone https://github.com/geohot/tinygrad.git
cd tinygrad

# Install TinyGrad and dependencies
pip install -e .

# Verify installation
python -c "import tinygrad; print('✅ TinyGrad installed successfully!')"
```

### 3. Test Your Setup

```bash
# Test RISC-V programs in Ripes
# Open Ripes and load: riscv_basics/matrix_mult.s

# Test TinyGrad
cd tinygrad
python ../mnist_test.py
python ../matrix_ops_test.py
```

## 📁 Directory Structure

```
phase1_setup/
├── riscv_basics/           # RISC-V assembly programs
│   ├── basic_arithmetic.s  # Basic arithmetic operations
│   ├── matrix_mult.s       # 2x2 matrix multiplication
│   ├── memory_ops.s        # Memory operations with arrays
│   └── README.md           # RISC-V setup instructions
├── tinygrad_setup/         # TinyGrad environment
│   ├── mnist_test.py       # MNIST inference test
│   ├── matrix_ops_test.py  # Matrix operations benchmark
│   └── README.md           # TinyGrad setup instructions
└── README.md               # This file
```

## 🔧 Detailed Setup Instructions

### RISC-V Environment Setup

#### Option A: Ripes (GUI Simulator) - **Recommended for Learning**

**Why Ripes?**
- Visual interface showing registers, memory, and pipeline
- Step-by-step execution for debugging
- Real-time updates as you step through code
- Cross-platform compatibility

**Installation:**
1. Install via Homebrew: `brew install --cask ripes`
2. Launch Ripes from Applications folder
3. Create new project
4. Load assembly file: File → Load Assembly
5. Set architecture to RV32I (32-bit, integer only)
6. Click "Assemble" button
7. Use "Step" button to execute instruction by instruction

#### Option B: Spike (Command Line) - **For Advanced Users**

```bash
# Install Spike
brew install riscv-isa-sim

# Test a program
spike pk riscv_basics/basic_arithmetic.s
```

### TinyGrad Environment Setup

#### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

#### Step-by-Step Installation

1. **Clone Repository**
   ```bash
   cd riscv-ai-accelerator/phase1_setup/tinygrad_setup
   git clone https://github.com/geohot/tinygrad.git
   cd tinygrad
   ```

2. **Install Dependencies**
   ```bash
   # Install TinyGrad in development mode
   pip install -e .
   
   # Alternative: Install from requirements
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python -c "import tinygrad; print('TinyGrad imported successfully!')"
   python -c "import tinygrad; print(f'Version: {tinygrad.__version__}')"
   ```

## 🧪 Testing Your Setup

### RISC-V Testing

1. **Open Ripes** and create a new project
2. **Load Assembly**: File → Load Assembly → Select `matrix_mult.s`
3. **Configure**: Set architecture to RV32I
4. **Assemble**: Click "Assemble" button
5. **Run**: Use "Step" to execute instruction by instruction
6. **Monitor**: Watch registers and memory in real-time

**Expected Results for `matrix_mult.s`:**
- Matrix A: [1 2; 3 4]
- Matrix B: [5 6; 7 8]
- Result: [19 22; 43 50]

### TinyGrad Testing

1. **Basic MNIST Test**
   ```bash
   cd tinygrad
   python ../mnist_test.py
   ```

   **Expected Output:**
   - Model creation confirmation
   - Inference results on dummy data
   - Accuracy metrics
   - Matrix operation performance

2. **Matrix Operations Benchmark**
   ```bash
   python ../matrix_ops_test.py
   ```

   **Expected Output:**
   - Matrix multiplication performance (GFLOPS)
   - Element-wise operation timing
   - Memory operation benchmarks
   - Vector operation performance

## 📊 Expected Performance

### RISC-V Simulation
- **Ripes**: Real-time simulation, suitable for learning
- **Spike**: Faster execution, suitable for batch testing

### TinyGrad on Apple Silicon
- **Matrix Multiplication**: 1-10 GFLOPS (depending on matrix size)
- **MNIST Inference**: <100ms for batch of 8 images
- **Memory Operations**: <1ms for 512x512 matrices

## 🚨 Troubleshooting

### Common Issues

1. **Ripes Installation**
   - If Homebrew fails, download .dmg from GitHub releases
   - Ensure macOS version compatibility

2. **TinyGrad Import Errors**
   - Make sure you're in the tinygrad directory
   - Check Python version (3.8+ required)
   - Try `pip install --upgrade pip` first

3. **Performance Issues**
   - First run may be slower due to JIT compilation
   - Ensure sufficient RAM (>4GB recommended)
   - Close other applications to free memory

### Getting Help

- **RISC-V Issues**: Check Ripes GitHub issues
- **TinyGrad Issues**: Check TinyGrad GitHub issues
- **General Setup**: Review this README and subdirectory READMEs

## 🎯 Next Steps After Phase 1

Once you've successfully completed all tests:

1. **Analyze Performance**: Use benchmark results to identify bottlenecks
2. **Study RISC-V**: Understand the assembly programs and instruction set
3. **Explore TinyGrad**: Dive deeper into the framework's internals
4. **Plan Phase 2**: Design custom AI instructions for RISC-V

## 📚 Additional Resources

- [RISC-V Specification](https://riscv.org/technical/specifications/)
- [TinyGrad Documentation](https://tinygrad.org/)
- [Ripes GitHub](https://github.com/mortbopet/Ripes)
- [Spike Simulator](https://github.com/riscv-riscv-isa-sim)

---

**🎉 Congratulations!** You've completed Phase 1 setup. Your RISC-V AI Accelerator Simulator foundation is ready for the next phase of development.

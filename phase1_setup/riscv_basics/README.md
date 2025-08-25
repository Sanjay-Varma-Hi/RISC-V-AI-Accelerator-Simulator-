# RISC-V Basics Setup

This directory contains RISC-V assembly programs and setup instructions for the RISC-V AI Accelerator Simulator project.

## RISC-V Simulator Setup

### Option 1: Ripes (Recommended for Learning)
Ripes is a user-friendly GUI simulator perfect for learning RISC-V assembly.

#### Installation on macOS:
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
# The executable will be in the build directory
```

#### Why Ripes?
- **Visual Interface**: See registers, memory, and pipeline stages
- **Step-by-step Execution**: Debug and understand each instruction
- **Real-time Updates**: Watch values change as you step through code
- **Cross-platform**: Works on macOS, Windows, and Linux

### Option 2: Spike (Command Line)
For more advanced users who prefer command-line tools.

```bash
# Install Spike via Homebrew
brew install riscv-isa-sim

# Or build from source
git clone https://github.com/riscv-riscv-isa-sim.git
cd riscv-isa-sim
mkdir build && cd build
../configure --prefix=/usr/local
make
sudo make install
```

### Option 3: QEMU (Alternative Simulator)
```bash
# Install QEMU
brew install qemu

# Run RISC-V programs
qemu-riscv32 -L /usr/local/riscv64-unknown-elf -E LD_LIBRARY_PATH=/usr/local/riscv64-unknown-elf/lib program
```

## Sample Programs

### 1. Basic Arithmetic (`basic_arithmetic.s`)
Simple addition and multiplication operations.

### 2. Matrix Multiplication (`matrix_mult.s`)
Loads two 2x2 matrices and multiplies them using basic RISC-V instructions.

### 3. Memory Operations (`memory_ops.s`)
Demonstrates load/store operations with arrays.

## Running Programs in Ripes

1. **Open Ripes** and create a new project
2. **Load Assembly**: File → Load Assembly → Select your .s file
3. **Configure**: Set architecture to RV32I (32-bit, integer only)
4. **Assemble**: Click "Assemble" button
5. **Run**: Use "Step" to execute instruction by instruction
6. **Monitor**: Watch registers and memory in real-time

## Running Programs with Spike

```bash
# Assemble the program first (you'll need a RISC-V assembler)
riscv64-unknown-elf-gcc -march=rv32i -mabi=ilp32 -nostdlib -o program program.s

# Run with Spike
spike pk program
```

## Expected Outputs

Each program includes expected register values and memory contents after execution. Use these to verify your understanding and debug any issues.

## Next Steps

After mastering these basics, you'll be ready to:
- Design custom AI instructions
- Implement vector operations
- Optimize matrix multiplication algorithms
- Integrate with TinyGrad workloads

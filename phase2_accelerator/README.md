# Phase 2: Custom RISC-V AI Accelerator

This phase implements a custom SIMD instruction `VMMUL` (Vectorized Matrix Multiply) to accelerate matrix multiplication operations.

## 🎯 Phase 2 Goals

- Design and implement custom `VMMUL` instruction
- Create Verilog implementation with testbench
- Integrate with RISC-V simulation environment
- Prepare Python bridge for TinyGrad integration

## 🚀 Custom Instruction: VMMUL

### Instruction Specification

**Mnemonic:** `VMMUL`
**Full Name:** Vectorized Matrix Multiply
**Type:** R-type instruction
**Purpose:** Accelerate 4x4 matrix multiplication using SIMD operations

### Instruction Format

```
31    25 24    20 19    15 14    12 11    7  6     0
| funct7 |   rs2   |   rs1   | funct3 |   rd   | opcode |
```

**Fields:**
- `funct7`: 7-bit function code (custom encoding)
- `rs2`: Source register 2 (second matrix base address)
- `rs1`: Source register 1 (first matrix base address)  
- `funct3`: Function code (000 for VMMUL)
- `rd`: Destination register (result matrix base address)
- `opcode`: 7-bit opcode (custom opcode)

### Binary Encoding

```
VMMUL rd, rs1, rs2
```

**Custom Opcode:** `1111011` (0x7B)
**Function Code:** `0000000` (0x00)
**Function3:** `000` (0x0)

**Complete Encoding:**
```
0000000_rs2_rs1_000_rd_1111011
```

### Example Assembly Usage

```assembly
# Load matrix addresses into registers
la x1, matrix_a      # x1 = address of matrix A
la x2, matrix_b      # x2 = address of matrix B
la x3, result        # x3 = address of result matrix

# Execute VMMUL instruction
vmmul x3, x1, x2     # result = A × B
```

### Pseudocode Implementation

```python
def VMMUL(rd, rs1, rs2):
    # Load 4x4 matrices from memory
    matrix_a = load_matrix_from_memory(rs1)  # 4x4 matrix A
    matrix_b = load_matrix_from_memory(rs2)  # 4x4 matrix B
    
    # Initialize result matrix
    result = [[0 for _ in range(4)] for _ in range(4)]
    
    # Parallel matrix multiplication using SIMD
    for i in range(4):
        for j in range(4):
            # Parallel multiply-accumulate for row i × column j
            acc = 0
            for k in range(4):
                acc += matrix_a[i][k] * matrix_b[k][j]
            result[i][j] = acc
    
    # Store result matrix to memory
    store_matrix_to_memory(rd, result)
```

### Memory Layout

**Input Matrix A (4x4):**
```
Memory Address: rs1
[rs1+0]  [rs1+4]  [rs1+8]  [rs1+12]   # Row 0: A[0][0] A[0][1] A[0][2] A[0][3]
[rs1+16] [rs1+20] [rs1+24] [rs1+28]   # Row 1: A[1][0] A[1][1] A[1][2] A[1][3]
[rs1+32] [rs1+36] [rs1+40] [rs1+44]   # Row 2: A[2][0] A[2][1] A[2][2] A[2][3]
[rs1+48] [rs1+52] [rs1+56] [rs1+60]   # Row 3: A[3][0] A[3][1] A[3][2] A[3][3]
```

**Input Matrix B (4x4):**
```
Memory Address: rs2
[rs2+0]  [rs2+4]  [rs2+8]  [rs2+12]   # Row 0: B[0][0] B[0][1] B[0][2] B[0][3]
[rs2+16] [rs2+20] [rs2+24] [rs2+28]   # Row 1: B[1][0] B[1][1] B[1][2] B[1][3]
[rs2+32] [rs2+36] [rs2+40] [rs2+44]   # Row 2: B[2][0] B[2][1] B[2][2] B[2][3]
[rs2+48] [rs2+52] [rs2+56] [rs2+60]   # Row 3: B[3][0] B[3][1] B[3][2] B[3][3]
```

**Result Matrix (4x4):**
```
Memory Address: rd
[rd+0]   [rd+4]   [rd+8]   [rd+12]    # Row 0: R[0][0] R[0][1] R[0][2] R[0][3]
[rd+16]  [rd+20]  [rd+24]  [rd+28]    # Row 1: R[1][0] R[1][1] R[1][2] R[1][3]
[rd+32]  [rd+36]  [rd+40]  [rd+44]    # Row 2: R[2][0] R[2][1] R[2][2] R[2][3]
[rd+48]  [rd+52]  [rd+56]  [rd+60]    # Row 3: R[3][0] R[3][1] R[3][2] R[3][3]
```

## 🔧 Implementation Components

### 1. Verilog RTL (`rtl/vmmul.v`)
- Main VMMUL module implementation
- 4x4 matrix multiplication logic
- SIMD parallel processing units

### 2. Testbench (`rtl/vmmul_tb.v`)
- Comprehensive testing of VMMUL functionality
- Sample matrix inputs and expected outputs
- Waveform generation for verification

### 3. RISC-V Integration (`riscv_tests/test_vmmul.S`)
- Assembly program using VMMUL instruction
- Matrix data setup and result verification
- Integration with RISC-V simulator

### 4. Python Bridge (`integration/vmmul_sim.py`)
- Python interface to VMMUL accelerator
- NumPy matrix input/output handling
- TinyGrad integration hooks

## 📊 Performance Benefits

**Standard RISC-V Matrix Multiplication:**
- 64 multiply operations
- 48 add operations  
- Sequential execution
- Estimated: 112 cycles

**VMMUL Instruction:**
- 64 multiply operations (parallel)
- 48 add operations (parallel)
- SIMD execution
- Estimated: 16-32 cycles

**Speedup:** 3.5x to 7x faster matrix multiplication

## 🚀 Next Steps

1. **Implement Verilog RTL** - Core VMMUL functionality
2. **Create Testbench** - Verification and testing
3. **Simulate on EDA Playground** - Online verification
4. **Integrate with RISC-V** - Instruction set extension
5. **Build Python Bridge** - TinyGrad integration

## 📚 Resources

- [RISC-V Specification](https://riscv.org/technical/specifications/)
- [Verilog Tutorial](https://www.asic-world.com/verilog/veritut.html)
- [EDA Playground](https://www.edaplayground.com/)
- [TinyGrad Documentation](https://tinygrad.org/)

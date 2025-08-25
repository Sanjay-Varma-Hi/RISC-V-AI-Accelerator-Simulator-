# Basic Arithmetic Operations in RISC-V
# This program demonstrates basic arithmetic operations
# Architecture: RV32I (32-bit, integer only)

.text
.global _start

_start:
    # Load immediate values into registers
    li x1, 10          # x1 = 10
    li x2, 5           # x2 = 5
    li x3, 3           # x3 = 3
    
    # Addition: x4 = x1 + x2
    add x4, x1, x2     # x4 = 10 + 5 = 15
    
    # Subtraction: x5 = x1 - x2
    sub x5, x1, x2     # x5 = 10 - 5 = 5
    
    # Multiplication: x6 = x2 * x3
    mul x6, x2, x3     # x6 = 5 * 3 = 15
    
    # Division: x7 = x1 / x2
    div x7, x1, x2     # x7 = 10 / 5 = 2
    
    # Remainder: x8 = x1 % x2
    rem x8, x1, x2     # x8 = 10 % 5 = 0
    
    # Logical AND: x9 = x1 & x2
    and x9, x1, x2     # x9 = 10 & 5 = 0 (1010 & 0101 = 0000)
    
    # Logical OR: x10 = x1 | x2
    or x10, x1, x2     # x10 = 10 | 5 = 15 (1010 | 0101 = 1111)
    
    # Logical XOR: x11 = x1 ^ x2
    xor x11, x1, x2    # x11 = 10 ^ 5 = 15 (1010 ^ 0101 = 1111)
    
    # Shift left: x12 = x2 << 2
    slli x12, x2, 2    # x12 = 5 << 2 = 20 (101 << 2 = 10100)
    
    # Shift right: x13 = x1 >> 1
    srli x13, x1, 1    # x13 = 10 >> 1 = 5 (1010 >> 1 = 101)
    
    # End program
    j _start            # Loop back to start (infinite loop for simulation)

# Expected Register Values After Execution:
# x1 = 10, x2 = 5, x3 = 3
# x4 = 15 (addition)
# x5 = 5  (subtraction)
# x6 = 15 (multiplication)
# x7 = 2  (division)
# x8 = 0  (remainder)
# x9 = 0  (AND)
# x10 = 15 (OR)
# x11 = 15 (XOR)
# x12 = 20 (shift left)
# x13 = 5  (shift right)

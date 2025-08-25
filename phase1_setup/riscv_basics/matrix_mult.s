# Matrix Multiplication in RISC-V
# This program multiplies two 2x2 matrices using basic RISC-V instructions
# Matrix A: [1 2]  Matrix B: [5 6]  Result: [19 22]
#           [3 4]            [7 8]           [43 50]
# Architecture: RV32I (32-bit, integer only)

.data
    # Matrix A: 2x2 matrix stored row by row
    matrix_a: .word 1, 2, 3, 4
    
    # Matrix B: 2x2 matrix stored row by row  
    matrix_b: .word 5, 6, 7, 8
    
    # Result matrix: 2x2 matrix to store A * B
    result: .space 16  # 4 words * 4 bytes = 16 bytes

.text
.global _start

_start:
    # Load base addresses
    la x1, matrix_a      # x1 = address of matrix A
    la x2, matrix_b      # x2 = address of matrix B
    la x3, result        # x3 = address of result matrix
    
    # Calculate result[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    # Load A[0][0] and B[0][0]
    lw x4, 0(x1)         # x4 = A[0][0] = 1
    lw x5, 0(x2)         # x5 = B[0][0] = 5
    mul x6, x4, x5       # x6 = 1 * 5 = 5
    
    # Load A[0][1] and B[1][0]
    lw x7, 4(x1)         # x7 = A[0][1] = 2
    lw x8, 8(x2)         # x8 = B[1][0] = 7
    mul x9, x7, x8       # x9 = 2 * 7 = 14
    
    # Add the two products
    add x10, x6, x9      # x10 = 5 + 14 = 19
    sw x10, 0(x3)        # Store result[0][0] = 19
    
    # Calculate result[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1]
    lw x4, 0(x1)         # x4 = A[0][0] = 1
    lw x5, 4(x2)         # x5 = B[0][1] = 6
    mul x6, x4, x5       # x6 = 1 * 6 = 6
    
    lw x7, 4(x1)         # x7 = A[0][1] = 2
    lw x8, 12(x2)        # x8 = B[1][1] = 8
    mul x9, x7, x8       # x9 = 2 * 8 = 16
    
    add x10, x6, x9      # x10 = 6 + 16 = 22
    sw x10, 4(x3)        # Store result[0][1] = 22
    
    # Calculate result[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0]
    lw x4, 8(x1)         # x4 = A[1][0] = 3
    lw x5, 0(x2)         # x5 = B[0][0] = 5
    mul x6, x4, x5       # x6 = 3 * 5 = 15
    
    lw x7, 12(x1)        # x7 = A[1][1] = 4
    lw x8, 8(x2)         # x8 = B[1][0] = 7
    mul x9, x7, x8       # x9 = 4 * 7 = 28
    
    add x10, x6, x9      # x10 = 15 + 28 = 43
    sw x10, 8(x3)        # Store result[1][0] = 43
    
    # Calculate result[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1]
    lw x4, 8(x1)         # x4 = A[1][0] = 3
    lw x5, 4(x2)         # x5 = B[0][1] = 6
    mul x6, x4, x5       # x6 = 3 * 6 = 18
    
    lw x7, 12(x1)        # x7 = A[1][1] = 4
    lw x8, 12(x2)        # x8 = B[1][1] = 8
    mul x9, x7, x8       # x9 = 4 * 8 = 32
    
    add x10, x6, x9      # x10 = 18 + 32 = 50
    sw x10, 12(x3)       # Store result[1][1] = 50
    
    # End program
    j _start              # Loop back to start (infinite loop for simulation)

# Expected Results:
# Matrix A: [1 2]  Matrix B: [5 6]  Result: [19 22]
#           [3 4]            [7 8]           [43 50]
#
# Memory Layout (word-aligned):
# matrix_a: [1, 2, 3, 4] at addresses [0x1000, 0x1004, 0x1008, 0x100C]
# matrix_b: [5, 6, 7, 8] at addresses [0x1010, 0x1014, 0x1018, 0x101C]
# result:   [19, 22, 43, 50] at addresses [0x1020, 0x1024, 0x1028, 0x102C]
#
# Register Summary:
# x1 = base address of matrix A
# x2 = base address of matrix B  
# x3 = base address of result matrix
# x4-x10 = temporary calculation registers

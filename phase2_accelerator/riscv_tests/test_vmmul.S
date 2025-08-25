# Test VMMUL Instruction - RISC-V Assembly
# This program demonstrates the custom VMMUL instruction for 4x4 matrix multiplication
# Architecture: RV32I with custom VMMUL extension

.data
    # Matrix A: 4x4 matrix stored row by row
    matrix_a: .word 1, 2, 3, 4,    # Row 0: [1 2 3 4]
              .word 5, 6, 7, 8,    # Row 1: [5 6 7 8]
              .word 9, 10, 11, 12, # Row 2: [9 10 11 12]
              .word 13, 14, 15, 16 # Row 3: [13 14 15 16]
    
    # Matrix B: 4x4 matrix stored row by row
    matrix_b: .word 5, 6, 7, 8,    # Row 0: [5 6 7 8]
              .word 9, 10, 11, 12, # Row 1: [9 10 11 12]
              .word 13, 14, 15, 16,# Row 2: [13 14 15 16]
              .word 17, 18, 19, 20 # Row 3: [17 18 19 20]
    
    # Result matrix: 4x4 matrix to store A × B
    result: .space 64  # 16 words * 4 bytes = 64 bytes
    
    # Expected result for verification
    expected: .word 90, 100, 110, 120,   # Row 0: [90 100 110 120]
              .word 202, 228, 254, 280,  # Row 1: [202 228 254 280]
              .word 314, 356, 398, 440,  # Row 2: [314 356 398 440]
              .word 426, 484, 542, 600   # Row 3: [426 484 542 600]

.text
.global _start

_start:
    # Load base addresses into registers
    la x1, matrix_a      # x1 = address of matrix A
    la x2, matrix_b      # x2 = address of matrix B
    la x3, result        # x3 = address of result matrix
    la x4, expected      # x4 = address of expected results
    
    # Execute VMMUL instruction (custom instruction)
    # This would be the custom VMMUL instruction in a real implementation
    # For now, we'll simulate it with a call to our custom function
    # In the actual RISC-V implementation, this would be:
    # vmmul x3, x1, x2
    
    # Call our matrix multiplication function
    jal ra, matrix_multiply
    
    # Verify results
    jal ra, verify_results
    
    # End program
    j _start

# Matrix multiplication function (simulates VMMUL behavior)
matrix_multiply:
    # Save return address
    addi sp, sp, -4
    sw ra, 0(sp)
    
    # Initialize loop counters
    li x5, 0           # i = 0 (row index)
    li x6, 0           # j = 0 (column index)
    li x7, 0           # k = 0 (inner loop index)
    
    # Outer loop: for i = 0 to 3
    outer_loop:
        li x6, 0       # j = 0 for each row
        
        # Middle loop: for j = 0 to 3
        middle_loop:
            # Calculate result[i][j] = sum(A[i][k] * B[k][j])
            li x7, 0   # k = 0
            li x8, 0   # accumulator = 0
            
            # Inner loop: for k = 0 to 3
            inner_loop:
                # Load A[i][k]
                slli x9, x5, 2      # x9 = i * 4 (row offset)
                add x9, x9, x7      # x9 = i * 4 + k
                slli x9, x9, 2      # x9 = (i * 4 + k) * 4 (byte offset)
                add x9, x1, x9      # x9 = base_addr_A + offset
                lw x10, 0(x9)       # x10 = A[i][k]
                
                # Load B[k][j]
                slli x11, x7, 2     # x11 = k * 4 (row offset)
                add x11, x11, x6    # x11 = k * 4 + j
                slli x11, x11, 2    # x11 = (k * 4 + j) * 4 (byte offset)
                add x11, x2, x11    # x11 = base_addr_B + offset
                lw x12, 0(x11)      # x12 = B[k][j]
                
                # Multiply and accumulate
                mul x13, x10, x12   # x13 = A[i][k] * B[k][j]
                add x8, x8, x13     # accumulator += A[i][k] * B[k][j]
                
                # Increment k
                addi x7, x7, 1
                li x14, 4
                blt x7, x14, inner_loop
            
            # Store result[i][j]
            slli x9, x5, 2          # x9 = i * 4 (row offset)
            add x9, x9, x6          # x9 = i * 4 + j
            slli x9, x9, 2          # x9 = (i * 4 + j) * 4 (byte offset)
            add x9, x3, x9          # x9 = base_addr_result + offset
            sw x8, 0(x9)            # result[i][j] = accumulator
            
            # Increment j
            addi x6, x6, 1
            li x14, 4
            blt x6, x14, middle_loop
        
        # Increment i
        addi x5, x5, 1
        li x14, 4
        blt x5, x14, outer_loop
    
    # Restore return address and return
    lw ra, 0(sp)
    addi sp, sp, 4
    ret

# Result verification function
verify_results:
    # Save return address
    addi sp, sp, -4
    sw ra, 0(sp)
    
    # Initialize verification counter
    li x5, 0           # element counter
    li x6, 0           # error counter
    
    # Loop through all 16 elements
    verify_loop:
        # Load result element
        slli x7, x5, 2      # x7 = counter * 4 (byte offset)
        add x8, x3, x7      # x8 = result + offset
        lw x9, 0(x8)        # x9 = result[element]
        
        # Load expected element
        add x10, x4, x7     # x10 = expected + offset
        lw x11, 0(x10)      # x11 = expected[element]
        
        # Compare
        beq x9, x11, verify_ok
        
        # Mismatch found
        addi x6, x6, 1      # increment error counter
        
        verify_ok:
        # Increment counter
        addi x5, x5, 1
        li x7, 16
        blt x5, x7, verify_loop
    
    # Print verification result
    beq x6, x0, verification_passed
    
    # Verification failed
    li x10, 1               # exit code 1 (error)
    j verification_end
    
    verification_passed:
    li x10, 0               # exit code 0 (success)
    
    verification_end:
    # Restore return address and return
    lw ra, 0(sp)
    addi sp, sp, 4
    ret

# Data section for debugging
.data
    debug_msg: .string "VMMUL Test Program - Matrix Multiplication\n"
    success_msg: .string "✅ All results verified successfully!\n"
    error_msg: .string "❌ Verification failed!\n"

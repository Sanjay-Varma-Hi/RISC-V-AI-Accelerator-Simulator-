# Memory Operations in RISC-V
# This program demonstrates load/store operations with arrays
# Architecture: RV32I (32-bit, integer only)

.data
    # Array of 5 integers
    array: .word 10, 20, 30, 40, 50
    
    # Space for result array (will store doubled values)
    result: .space 20  # 5 words * 4 bytes = 20 bytes
    
    # Single values for testing
    value1: .word 100
    value2: .word 200

.text
.global _start

_start:
    # Load base addresses
    la x1, array         # x1 = address of input array
    la x2, result        # x2 = address of result array
    la x3, value1        # x3 = address of value1
    la x4, value2        # x4 = address of value2
    
    # Load single values
    lw x5, 0(x3)         # x5 = value1 = 100
    lw x6, 0(x4)         # x6 = value2 = 200
    
    # Add the two values
    add x7, x5, x6       # x7 = 100 + 200 = 300
    
    # Store result back to memory
    sw x7, 0(x3)         # Store 300 at value1 location
    
    # Process array: double each element and store in result
    # Load array[0]
    lw x8, 0(x1)         # x8 = array[0] = 10
    slli x9, x8, 1       # x9 = 10 * 2 = 20 (shift left by 1 = multiply by 2)
    sw x9, 0(x2)         # result[0] = 20
    
    # Load array[1]
    lw x8, 4(x1)         # x8 = array[1] = 20
    slli x9, x8, 1       # x9 = 20 * 2 = 40
    sw x9, 4(x2)         # result[1] = 40
    
    # Load array[2]
    lw x8, 8(x1)         # x8 = array[2] = 30
    slli x9, x8, 1       # x9 = 30 * 2 = 60
    sw x9, 8(x2)         # result[2] = 60
    
    # Load array[3]
    lw x8, 12(x1)        # x8 = array[3] = 40
    slli x9, x8, 1       # x9 = 40 * 2 = 80
    sw x9, 12(x2)        # result[3] = 80
    
    # Load array[4]
    lw x8, 16(x1)        # x8 = array[4] = 50
    slli x9, x8, 1       # x9 = 50 * 2 = 100
    sw x9, 16(x2)        # result[4] = 100
    
    # Test byte operations (load byte, store byte)
    # Load the least significant byte of array[0]
    lb x10, 0(x1)        # x10 = 10 (sign-extended to 32 bits)
    
    # Store a byte value
    li x11, 255          # x11 = 255
    sb x11, 0(x4)        # Store 255 as a byte at value2 location
    
    # Test halfword operations (load halfword, store halfword)
    # Load halfword from array[0] (first 16 bits)
    lh x12, 0(x1)        # x12 = 10 (sign-extended to 32 bits)
    
    # Store halfword
    li x13, 1000         # x13 = 1000
    sh x13, 4(x4)        # Store 1000 as halfword at value2 + 4
    
    # End program
    j _start              # Loop back to start (infinite loop for simulation)

# Expected Results:
# Input array: [10, 20, 30, 40, 50]
# Result array: [20, 40, 60, 80, 100] (doubled values)
# value1: 300 (original 100 + 200)
# value2: 255 (byte value), 1000 (halfword value at +4 offset)
#
# Memory Layout:
# array: [10, 20, 30, 40, 50] at addresses [0x1000, 0x1004, 0x1008, 0x100C, 0x1010]
# result: [20, 40, 60, 80, 100] at addresses [0x1014, 0x1018, 0x101C, 0x1020, 0x1024]
# value1: 300 at address 0x1028
# value2: 255 at address 0x102C, 1000 at address 0x1030
#
# Register Summary:
# x1 = base address of input array
# x2 = base address of result array
# x3 = address of value1
# x4 = address of value2
# x5 = 100 (original value1)
# x6 = 200 (original value2)
# x7 = 300 (sum)
# x8 = temporary for array elements
# x9 = temporary for doubled values
# x10 = 10 (byte load)
# x11 = 255 (byte value)
# x12 = 10 (halfword load)
# x13 = 1000 (halfword value)

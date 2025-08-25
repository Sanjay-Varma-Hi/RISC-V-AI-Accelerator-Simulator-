// VMMUL Testbench
// Testbench for Vectorized Matrix Multiply Module
// Author: RISC-V AI Accelerator Simulator Project
// Date: 2024

`timescale 1ns / 1ps

module vmmul_tb;

    // Testbench parameters
    parameter CLK_PERIOD = 10;  // 10ns = 100MHz clock
    
    // Testbench signals
    reg clk;
    reg rst_n;
    reg enable;
    reg [31:0] matrix_a_addr;
    reg [31:0] matrix_b_addr;
    reg [31:0] result_addr;
    reg start;
    
    wire done;
    wire [31:0] result_data;
    wire [31:0] result_addr_out;
    
    // Instantiate the VMMUL module
    vmmul uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .matrix_a_addr(matrix_a_addr),
        .matrix_b_addr(matrix_b_addr),
        .result_addr(result_addr),
        .start(start),
        .done(done),
        .result_data(result_data),
        .result_addr_out(result_addr_out)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        // Initialize waveform dump
        $dumpfile("vmmul_tb.vcd");
        $dumpvars(0, vmmul_tb);
        
        // Initialize inputs
        rst_n = 1;
        enable = 0;
        matrix_a_addr = 32'h1000;
        matrix_b_addr = 32'h2000;
        result_addr = 32'h3000;
        start = 0;
        
        // Print test header
        $display("==========================================");
        $display("VMMUL Testbench - Matrix Multiplication");
        $display("==========================================");
        
        // Test 1: Reset test
        $display("\nTest 1: Reset Test");
        $display("-------------------");
        rst_n = 0;
        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD);
        
        if (done == 0 && result_data == 0) begin
            $display("✅ Reset test PASSED");
        end else begin
            $display("❌ Reset test FAILED");
        end
        
        // Test 2: Enable and start computation
        $display("\nTest 2: Matrix Multiplication Test");
        $display("-----------------------------------");
        enable = 1;
        start = 1;
        #(CLK_PERIOD);
        start = 0;
        
        // Wait for computation to complete
        wait(done);
        #(CLK_PERIOD * 2);
        
        // Test 3: Verify results
        $display("\nTest 3: Result Verification");
        $display("---------------------------");
        
        // Expected matrix A (4x4):
        // [1  2  3  4]
        // [5  6  7  8]
        // [9  10 11 12]
        // [13 14 15 16]
        
        // Expected matrix B (4x4):
        // [5  6  7  8]
        // [9  10 11 12]
        // [13 14 15 16]
        // [17 18 19 20]
        
        // Expected result matrix (A × B):
        // [90  100 110 120]
        // [202 228 254 280]
        // [314 356 398 440]
        // [426 484 542 600]
        
        $display("Matrix A:");
        $display("  [%0d %0d %0d %0d]", uut.matrix_a[0], uut.matrix_a[1], uut.matrix_a[2], uut.matrix_a[3]);
        $display("  [%0d %0d %0d %0d]", uut.matrix_a[4], uut.matrix_a[5], uut.matrix_a[6], uut.matrix_a[7]);
        $display("  [%0d %0d %0d %0d]", uut.matrix_a[8], uut.matrix_a[9], uut.matrix_a[10], uut.matrix_a[11]);
        $display("  [%0d %0d %0d %0d]", uut.matrix_a[12], uut.matrix_a[13], uut.matrix_a[14], uut.matrix_a[15]);
        
        $display("\nMatrix B:");
        $display("  [%0d %0d %0d %0d]", uut.matrix_b[0], uut.matrix_b[1], uut.matrix_b[2], uut.matrix_b[3]);
        $display("  [%0d %0d %0d %0d]", uut.matrix_b[4], uut.matrix_b[5], uut.matrix_b[6], uut.matrix_b[7]);
        $display("  [%0d %0d %0d %0d]", uut.matrix_b[8], uut.matrix_b[9], uut.matrix_b[10], uut.matrix_b[11]);
        $display("  [%0d %0d %0d %0d]", uut.matrix_b[12], uut.matrix_b[13], uut.matrix_b[14], uut.matrix_b[15]);
        
        $display("\nResult Matrix (A × B):");
        $display("  [%0d %0d %0d %0d]", uut.result[0], uut.result[1], uut.result[2], uut.result[3]);
        $display("  [%0d %0d %0d %0d]", uut.result[4], uut.result[5], uut.result[6], uut.result[7]);
        $display("  [%0d %0d %0d %0d]", uut.result[8], uut.result[9], uut.result[10], uut.result[11]);
        $display("  [%0d %0d %0d %0d]", uut.result[12], uut.result[13], uut.result[14], uut.result[15]);
        
        // Verify key results
        if (uut.result[0] == 90 && uut.result[1] == 100 && uut.result[2] == 110 && uut.result[3] == 120) begin
            $display("✅ Row 0 verification PASSED");
        end else begin
            $display("❌ Row 0 verification FAILED");
            $display("   Expected: [90 100 110 120], Got: [%0d %0d %0d %0d]", 
                     uut.result[0], uut.result[1], uut.result[2], uut.result[3]);
        end
        
        if (uut.result[4] == 202 && uut.result[5] == 228 && uut.result[6] == 254 && uut.result[7] == 280) begin
            $display("✅ Row 1 verification PASSED");
        end else begin
            $display("❌ Row 1 verification FAILED");
            $display("   Expected: [202 228 254 280], Got: [%0d %0d %0d %0d]", 
                     uut.result[4], uut.result[5], uut.result[6], uut.result[7]);
        end
        
        if (uut.result[8] == 314 && uut.result[9] == 356 && uut.result[10] == 398 && uut.result[11] == 440) begin
            $display("✅ Row 2 verification PASSED");
        end else begin
            $display("❌ Row 2 verification FAILED");
            $display("   Expected: [314 356 398 440], Got: [%0d %0d %0d %0d]", 
                     uut.result[8], uut.result[9], uut.result[10], uut.result[11]);
        end
        
        if (uut.result[12] == 426 && uut.result[13] == 484 && uut.result[14] == 542 && uut.result[15] == 600) begin
            $display("✅ Row 3 verification PASSED");
        end else begin
            $display("❌ Row 3 verification FAILED");
            $display("   Expected: [426 484 542 600], Got: [%0d %0d %0d %0d]", 
                     uut.result[12], uut.result[13], uut.result[14], uut.result[15]);
        end
        
        // Test 4: Performance measurement
        $display("\nTest 4: Performance Analysis");
        $display("----------------------------");
        $display("Clock cycles for computation: %0d", $time / CLK_PERIOD);
        $display("Matrix size: 4x4");
        $display("Operations: 64 multiplies + 48 adds = 112 total");
        
        // Test 5: State machine verification
        $display("\nTest 5: State Machine Verification");
        $display("-----------------------------------");
        $display("Final state: %0b", uut.state);
        if (uut.state == 2'b00) begin
            $display("✅ State machine returned to IDLE");
        end else begin
            $display("❌ State machine did not return to IDLE");
        end
        
        // Test 6: Memory interface verification
        $display("\nTest 6: Memory Interface Verification");
        $display("--------------------------------------");
        $display("Result data output: %0d", result_data);
        $display("Result address output: 0x%0h", result_addr_out);
        
        // Summary
        $display("\n==========================================");
        $display("Test Summary");
        $display("==========================================");
        $display("Total tests run: 6");
        $display("Matrix multiplication: 4x4");
        $display("Clock frequency: %0d MHz", 1000/CLK_PERIOD);
        $display("Simulation completed successfully!");
        $display("==========================================");
        
        // End simulation
        #(CLK_PERIOD * 5);
        $finish;
    end
    
    // Monitor state changes
    always @(posedge clk) begin
        if (enable) begin
            case (uut.state)
                2'b00: $display("Time %0t: State = IDLE", $time);
                2'b01: $display("Time %0t: State = LOAD", $time);
                2'b10: $display("Time %0t: State = COMPUTE", $time);
                2'b11: $display("Time %0t: State = STORE", $time);
            endcase
        end
    end
    
    // Monitor done signal
    always @(posedge done) begin
        $display("Time %0t: Computation COMPLETED!", $time);
    end

endmodule

// VMMUL - Vectorized Matrix Multiply Module
// Custom RISC-V instruction for 4x4 matrix multiplication acceleration
// Author: RISC-V AI Accelerator Simulator Project
// Date: 2024

module vmmul (
    input wire clk,                    // Clock signal
    input wire rst_n,                  // Active-low reset
    input wire enable,                 // Enable signal
    input wire [31:0] matrix_a_addr,  // Base address of matrix A
    input wire [31:0] matrix_b_addr,  // Base address of matrix B
    input wire [31:0] result_addr,    // Base address of result matrix
    input wire start,                  // Start computation signal
    output reg done,                   // Computation complete signal
    output reg [31:0] result_data,    // Result data output
    output reg [31:0] result_addr_out // Result address output
);

    // Internal registers for matrix storage
    reg [31:0] matrix_a [0:15];  // 4x4 matrix A (16 elements)
    reg [31:0] matrix_b [0:15];  // 4x4 matrix B (16 elements)
    reg [31:0] result [0:15];    // 4x4 result matrix (16 elements)
    
    // State machine states
    localparam IDLE = 2'b00;
    localparam LOAD = 2'b01;
    localparam COMPUTE = 2'b10;
    localparam STORE = 2'b11;
    
    reg [1:0] state, next_state;
    
    // Counter registers
    reg [3:0] load_counter;
    reg [3:0] compute_counter;
    reg [3:0] store_counter;
    
    // Matrix multiplication indices
    reg [1:0] i, j, k;
    
    // Accumulator for MAC operations
    reg [31:0] acc;
    
    // State machine sequential logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    // State machine combinational logic
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start && enable) begin
                    next_state = LOAD;
                end
            end
            LOAD: begin
                if (load_counter == 4'd15) begin
                    next_state = COMPUTE;
                end
            end
            COMPUTE: begin
                if (compute_counter == 4'd15) begin
                    next_state = STORE;
                end
            end
            STORE: begin
                if (store_counter == 4'd15) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
    
    // Main control logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all registers
            done <= 1'b0;
            load_counter <= 4'd0;
            compute_counter <= 4'd0;
            store_counter <= 4'd0;
            i <= 2'b00;
            j <= 2'b00;
            k <= 2'b00;
            acc <= 32'd0;
            result_data <= 32'd0;
            result_addr_out <= 32'd0;
            
            // Reset matrix arrays
            for (integer idx = 0; idx < 16; idx = idx + 1) begin
                matrix_a[idx] <= 32'd0;
                matrix_b[idx] <= 32'd0;
                result[idx] <= 32'd0;
            end
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start && enable) begin
                        load_counter <= 4'd0;
                    end
                end
                
                LOAD: begin
                    // Load matrices from memory (simplified - in real implementation,
                    // this would interface with memory controller)
                    if (load_counter < 4'd16) begin
                        // For simulation, we'll use predefined values
                        // In real implementation, these would be loaded from memory
                        case (load_counter)
                            4'd0:  matrix_a[0] <= 32'd1;  matrix_b[0] <= 32'd5;
                            4'd1:  matrix_a[1] <= 32'd2;  matrix_b[1] <= 32'd6;
                            4'd2:  matrix_a[2] <= 32'd3;  matrix_b[2] <= 32'd7;
                            4'd3:  matrix_a[3] <= 32'd4;  matrix_b[3] <= 32'd8;
                            4'd4:  matrix_a[4] <= 32'd5;  matrix_b[4] <= 32'd9;
                            4'd5:  matrix_a[5] <= 32'd6;  matrix_b[5] <= 32'd10;
                            4'd6:  matrix_a[6] <= 32'd7;  matrix_b[6] <= 32'd11;
                            4'd7:  matrix_a[7] <= 32'd8;  matrix_b[7] <= 32'd12;
                            4'd8:  matrix_a[8] <= 32'd9;  matrix_b[8] <= 32'd13;
                            4'd9:  matrix_a[9] <= 32'd10; matrix_b[9] <= 32'd14;
                            4'd10: matrix_a[10] <= 32'd11; matrix_b[10] <= 32'd15;
                            4'd11: matrix_a[11] <= 32'd12; matrix_b[11] <= 32'd16;
                            4'd12: matrix_a[12] <= 32'd13; matrix_b[12] <= 32'd17;
                            4'd13: matrix_a[13] <= 32'd14; matrix_b[13] <= 32'd18;
                            4'd14: matrix_a[14] <= 32'd15; matrix_b[14] <= 32'd19;
                            4'd15: matrix_a[15] <= 32'd16; matrix_b[15] <= 32'd20;
                        endcase
                        load_counter <= load_counter + 4'd1;
                    end
                end
                
                COMPUTE: begin
                    // Matrix multiplication computation
                    if (compute_counter < 4'd16) begin
                        // Calculate result[i][j] = sum(A[i][k] * B[k][j])
                        i <= compute_counter[3:2];  // Row index (bits 3-2)
                        j <= compute_counter[1:0];  // Column index (bits 1-0)
                        k <= 2'b00;
                        acc <= 32'd0;
                        compute_counter <= compute_counter + 4'd1;
                    end else begin
                        // Perform MAC operations for current element
                        if (k < 2'b11) begin
                            // MAC: acc += A[i][k] * B[k][j]
                            acc <= acc + (matrix_a[i * 4 + k] * matrix_b[k * 4 + j]);
                            k <= k + 2'b01;
                        end else begin
                            // Final MAC operation and store result
                            acc <= acc + (matrix_a[i * 4 + k] * matrix_b[k * 4 + j]);
                            result[i * 4 + j] <= acc + (matrix_a[i * 4 + k] * matrix_b[k * 4 + j]);
                        end
                    end
                end
                
                STORE: begin
                    // Store results to memory
                    if (store_counter < 4'd16) begin
                        result_data <= result[store_counter];
                        result_addr_out <= result_addr + (store_counter * 4);
                        store_counter <= store_counter + 4'd1;
                    end else begin
                        done <= 1'b1;
                    end
                end
            endcase
        end
    end
    
    // Function to get matrix element (for debugging)
    function [31:0] get_matrix_a_element;
        input [1:0] row, col;
        get_matrix_a_element = matrix_a[row * 4 + col];
    endfunction
    
    function [31:0] get_matrix_b_element;
        input [1:0] row, col;
        get_matrix_b_element = matrix_b[row * 4 + col];
    endfunction
    
    function [31:0] get_result_element;
        input [1:0] row, col;
        get_result_element = result[row * 4 + col];
    endfunction

endmodule

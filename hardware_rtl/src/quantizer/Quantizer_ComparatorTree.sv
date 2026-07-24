module Quantizer_ComparatorTree #(
    parameter NUM_ELEMENTS = 32
    , parameter EXP_BITS = 8
    , parameter MAN_CMP_BITS = 4    // mantissa bits used for comparison
    , parameter CMP_BITS = EXP_BITS + MAN_CMP_BITS  // 12-bit magnitude comparison
    , parameter LOG2_NUM = $clog2(NUM_ELEMENTS)
) (
    // Input: sign, exponent, and 4 MSB of mantissa for each element
    input logic [NUM_ELEMENTS-1:0] sign_in
    , input logic [NUM_ELEMENTS-1:0][EXP_BITS-1:0] exp_in
    , input logic [NUM_ELEMENTS-1:0][MAN_CMP_BITS-1:0] mant4_in

    // Output: the element with maximum absolute value
    , output logic sign_max
    , output logic [EXP_BITS-1:0] exp_max
    , output logic [MAN_CMP_BITS-1:0] mant4_max
);

    // Binary comparator tree: compare by magnitude {exp, mant_4b}
    // Larger {exp, mant_4b} = larger absolute value
    genvar s, j;
    generate
        for (s = 0; s <= LOG2_NUM; s++) begin : tree
            localparam int N = NUM_ELEMENTS >> s;
            logic [N-1:0] t_sign;
            logic [N-1:0][EXP_BITS-1:0] t_exp;
            logic [N-1:0][MAN_CMP_BITS-1:0] t_mant4;

            if (s == 0) begin : init
                for (j = 0; j < N; j++) begin : elem
                    assign t_sign[j]  = sign_in[j];
                    assign t_exp[j]   = exp_in[j];
                    assign t_mant4[j] = mant4_in[j];
                end
            end else begin : reduce
                for (j = 0; j < N; j++) begin : elem
                    // Compare magnitudes: {exp, mant4}
                    logic a_geq_b;
                    assign a_geq_b = {tree[s-1].t_exp[2*j], tree[s-1].t_mant4[2*j]}
                                  >= {tree[s-1].t_exp[2*j+1], tree[s-1].t_mant4[2*j+1]};

                    assign t_sign[j]  = a_geq_b ? tree[s-1].t_sign[2*j]  : tree[s-1].t_sign[2*j+1];
                    assign t_exp[j]   = a_geq_b ? tree[s-1].t_exp[2*j]   : tree[s-1].t_exp[2*j+1];
                    assign t_mant4[j] = a_geq_b ? tree[s-1].t_mant4[2*j] : tree[s-1].t_mant4[2*j+1];
                end
            end
        end
    endgenerate

    assign sign_max  = tree[LOG2_NUM].t_sign[0];
    assign exp_max   = tree[LOG2_NUM].t_exp[0];
    assign mant4_max = tree[LOG2_NUM].t_mant4[0];

endmodule

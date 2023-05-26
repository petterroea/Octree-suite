#include "quantization.h"
#include <stdio.h>
#include <iostream>

int quantization_lookup_table[DCT_SIZE * DCT_SIZE * DCT_SIZE];

void build_quantization_table() {
    for(int x = 0; x < DCT_SIZE; x++) {
        for(int y = 0; y < DCT_SIZE; y++) {
            for(int z = 0; z < DCT_SIZE; z++) { // 4 to 21
                quantization_lookup_table[DCT_ADDRESS(x, y, z)] = 4 + static_cast<int>((static_cast<float>(x + y + z) / 3.0f) * 2.0f);
            }
        }
    }
}

void do_quantization(float* dct_in, int* quantized_out) {
    for(int z = 0; z < DCT_SIZE; z++) {
        for(int y = 0; y < DCT_SIZE; y++) {
            for(int x = 0; x < DCT_SIZE; x++) {
                float in = dct_in[DCT_ADDRESS(x, y, z)];
                float quant = static_cast<float>(quantization_lookup_table[DCT_ADDRESS(x, y, z)]);
                //std::cout << "In " << in << " quant " << quant << std::endl;
                quantized_out[DCT_ADDRESS(x, y, z)] = static_cast<int>(in / quant);
            }
        }
    }
    /*
    for(int x = 0; x < DCT_SIZE; x++) {
        for(int y = 0; y < DCT_SIZE; y++) {
            for(int z = 0; z < DCT_SIZE; z++) {
                printf("%d\t", quantized_out[DCT_ADDRESS(x, y, z)]);
            }
            printf("\n");
        }
        printf("\n\n");
    }*/
}

void do_de_quantization(int* quantized_in, float* dct_out) {
    for(int z = 0; z < DCT_SIZE; z++) {
        for(int y = 0; y < DCT_SIZE; y++) {
            for(int x = 0; x < DCT_SIZE; x++) {
                dct_out[DCT_ADDRESS(x, y, z)] = static_cast<float>(quantized_in[DCT_ADDRESS(x, y, z)]) * static_cast<float>(quantization_lookup_table[DCT_ADDRESS(x, y, z)]);
            }
        }
    }
    for(int x = 0; x < DCT_SIZE; x++) {
        for(int y = 0; y < DCT_SIZE; y++) {
            for(int z = 0; z < DCT_SIZE; z++) {
                printf("%f\t", dct_out[DCT_ADDRESS(x, y, z)]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}
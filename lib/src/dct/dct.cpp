#include "dct.h"

#include <cmath>
#include <math.h>
#include <stdio.h>
#include <numbers>

#define pi M_PI

float dct_lookup_buffer[DCT_SIZE * DCT_SIZE * DCT_SIZE * DCT_SIGNAL_SIZE * DCT_SIGNAL_SIZE * DCT_SIGNAL_SIZE];

void build_lookup_table() {
    // Per signal
    for(int i = 0; i < DCT_SIGNAL_SIZE; i++) {
        for(int j = 0; j < DCT_SIGNAL_SIZE; j++) {
            for(int k = 0; k < DCT_SIGNAL_SIZE; k++) {
                // Per pixel
                for(int x = 0; x < DCT_SIZE; x++) {
                    for(int y = 0; y < DCT_SIZE; y++) {
                        for(int z = 0; z < DCT_SIZE; z++) {
                        int table_i = i*DCT_SIGNAL_SIZE;
                        int table_j = j*DCT_SIGNAL_SIZE;
                        int table_k = k*DCT_SIGNAL_SIZE;

                        dct_lookup_buffer[DCT_LOOKUP_TABLE_ADDRESS(table_i + x, table_j+y, table_k + z)] = 
                            cos((2 * x + 1) * i * pi / (2 * DCT_SIZE)) *
                            cos((2 * y + 1) * j * pi / (2 * DCT_SIZE)) *
                            cos((2 * z + 1) * k * pi / (2 * DCT_SIZE));
                        }
                    }
                }
            }
        }
    }
}

// https://www.geeksforgeeks.org/discrete-cosine-transform-algorithm-program/
void do_dct(float* square, float* dct) {
    int i, j, k;
    int x, y, z;
 
    float ci, cj, ck;
    float dct1, sum;
    /*
    printf("Pre dct\n");
    for (i = 0; i < DCT_SIZE; i++) {
        for (j = 0; j < DCT_SIZE; j++) {
            for (k = 0; k < DCT_SIZE; k++) {
                printf("%f\t", square[i + j * DCT_SIZE + k * DCT_SIZE * DCT_SIZE]);
            }
            printf("\n");
        }
        printf("\n\n");
    }*/
    
    // For each signal
    for (k = 0; k < DCT_SIZE; k++) {
        for (j = 0; j < DCT_SIZE; j++) {
            for (i = 0; i < DCT_SIZE; i++) {
    
                // ci and cj depends on frequency as well as
                // number of row and columns of specified matrix
                if (i == 0)
                    ci = 1 / sqrt(DCT_SIZE);
                else
                    ci = sqrt(2) / sqrt(DCT_SIZE);

                if (j == 0)
                    cj = 1 / sqrt(DCT_SIZE);
                else
                    cj = sqrt(2) / sqrt(DCT_SIZE);

                if (k == 0)
                    ck = 1 / sqrt(DCT_SIZE);
                else
                    ck = sqrt(2) / sqrt(DCT_SIZE);
    
                // sum will temporarily store the sum of
                // cosine signals
                // For each pixel
                sum = 0;
                for (z = 0; z < DCT_SIZE; z++) {
                    for (y = 0; y < DCT_SIZE; y++) {
                        for (x = 0; x < DCT_SIZE; x++) {
                            int signal_i= i*DCT_SIGNAL_SIZE;
                            int signal_j= j*DCT_SIGNAL_SIZE;
                            int signal_k= k*DCT_SIGNAL_SIZE;
                            
                            dct1 = square[x + y * DCT_SIZE + z * DCT_SIZE * DCT_SIZE] * dct_lookup_buffer[DCT_LOOKUP_TABLE_ADDRESS(signal_i + x, signal_j + y, signal_k + z)];
                                
                            sum = sum + dct1;
                        }
                    }
                }
                dct[i + j * DCT_SIZE + k * DCT_SIZE * DCT_SIZE] = ci * cj * ck * sum;
            }
        }
    }
    /*
    printf("Post dct:\n");
    for (i = 0; i < DCT_SIZE; i++) {
        for (j = 0; j < DCT_SIZE; j++) {
            for (k = 0; k < DCT_SIZE; k++) {
                printf("%f\t", dct[DCT_ADDRESS(i, j, k)]);
            }
            printf("\n");
        }
        printf("\n\n");
    }*/
}

void reverse_dct(float* dct_input, float* square_out) {
    int i, j, k;
    int x, y, z;
 
    float ci, cj, ck;
    float dct1, sum;
 
    // For each pixel
    for (z = 0; z < DCT_SIZE; z++) {
        for (y = 0; y < DCT_SIZE; y++) {
            for (x = 0; x < DCT_SIZE; x++) {
    
                // ci and cj depends on frequency as well as
                // number of row and columns of specified matrix
                // sum will temporarily store the sum of
                // cosine signals
                // For each signal
                sum = 0;
                for (k = 0; k < DCT_SIZE; k++) {
                    for (j = 0; j < DCT_SIZE; j++) {
                        for (i = 0; i < DCT_SIZE; i++) {
                            if (i == 0)
                                ci = 1 / sqrt(DCT_SIZE);
                            else
                                ci = sqrt(2) / sqrt(DCT_SIZE);

                            if (j == 0)
                                cj = 1 / sqrt(DCT_SIZE);
                            else
                                cj = sqrt(2) / sqrt(DCT_SIZE);

                            if (k == 0)
                                ck = 1 / sqrt(DCT_SIZE);
                            else
                                ck = sqrt(2) / sqrt(DCT_SIZE);
        
        
                            int signal_i = i*DCT_SIGNAL_SIZE;
                            int signal_j = j*DCT_SIGNAL_SIZE;
                            int signal_k = k*DCT_SIGNAL_SIZE;

                            float dct_input_nocicj = dct_input[DCT_ADDRESS(i, j, k)] * ci * cj * ck;
                            
                            dct1 = dct_input_nocicj * dct_lookup_buffer[DCT_LOOKUP_TABLE_ADDRESS(signal_i + x, signal_j + y, signal_k + z)];
                                
                            sum = sum + dct1;
                        }
                    }
                }
                square_out[DCT_ADDRESS(x, y, z)] = sum;
            }
        }
    }
 
    printf("Post DCT:\n");
    for (i = 0; i < DCT_SIZE; i++) {
        for (j = 0; j < DCT_SIZE; j++) {
            for (k = 0; k < DCT_SIZE; k++) {
                printf("%f\t", square_out[DCT_ADDRESS(i, j, k)]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}
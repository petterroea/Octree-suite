#pragma once
#define DCT_SIZE 8
#define DCT_SIGNAL_SIZE 8

#define DCT_LOOKUP_TABLE_ADDRESS(x, y, z) ((x) + (y)*DCT_SIZE*DCT_SIGNAL_SIZE + (z) * DCT_SIZE * DCT_SIGNAL_SIZE * DCT_SIZE * DCT_SIGNAL_SIZE)
#define DCT_ADDRESS(x, y, z) ((x) + (y)*DCT_SIZE + (z) * DCT_SIZE * DCT_SIZE)

// 3d lookup table
extern float dct_lookup_buffer[DCT_SIZE * DCT_SIZE * DCT_SIZE * DCT_SIGNAL_SIZE * DCT_SIGNAL_SIZE * DCT_SIGNAL_SIZE];

void build_lookup_table();

void do_dct(float* square, float* output);
void reverse_dct(float* dct_input, float* square_out);
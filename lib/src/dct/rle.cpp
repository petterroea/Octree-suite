#include "rle.h"

#include <iostream>
#include <exception>

#include "dct.h"
//#include "utils.h"
#include "quantization.h"

#define SAME_VALUE 0b10000000
#define ARBITRARY_VALUE 0b00000000

#define OPTYPE_MASK 0b10000000

#define BYTE_MASK 0b01000000
#define SHORT_MASK 0b00000000

#define DATATYPE_MASK 0b01000000

#define LEN_SAMEBYTE 0b00100000
#define LEN_SHORT 0b00000000
#define LEN_SIZE_MASK 0b00100000

#define LEN_MASK 0b00011111

void write_repeat(int current_value, int repeat_count, std::ofstream* out) {
    if (current_value > 127 || current_value < -126) {
        unsigned char opcode = SAME_VALUE | (static_cast<unsigned char>(repeat_count) & LEN_MASK) | SHORT_MASK;
        short value = static_cast<short>(current_value);

        out->write((const char*)&opcode, sizeof(opcode));
        out->write((const char*)&value, sizeof(value));
    } else {
        unsigned char opcode = SAME_VALUE | (static_cast<unsigned char>(repeat_count) & LEN_MASK) | BYTE_MASK;
        char value = static_cast<char>(current_value);

        out->write((const char*)&opcode, sizeof(opcode));
        out->write((const char*)&value, sizeof(value));
    }
}

void write_arbitrary(int* data, int repeat_count, std::ofstream* out) {
    bool is_short = false;
    for(int i = 0; i < repeat_count; i++) {
        if(data[i] > 127 || data[i] < -126) {
            is_short = true;
            break;
        }
    }

    if(is_short) {
        //std::cout << "Writing short" << std::endl;
        unsigned char opcode = ARBITRARY_VALUE | (static_cast<unsigned char>(repeat_count) & LEN_MASK) | SHORT_MASK;

        out->write((const char*)&opcode, sizeof(opcode));

        for(int i = 0; i < repeat_count; i++) {
            short value = static_cast<short>(data[i]);
            out->write((const char*)&value, sizeof(value));
        }
    } else {
        //std::cout << "Writing long" << std::endl;
        unsigned char opcode = ARBITRARY_VALUE | (static_cast<unsigned char>(repeat_count) & LEN_MASK) | BYTE_MASK;

        out->write((const char*)&opcode, sizeof(opcode));

        for(int i = 0; i < repeat_count; i++) {
            char value = static_cast<char>(data[i]);
            out->write((const char*)&value, sizeof(value));
        }

    }

}

void write_run_time_length(int* data, int length, std::ofstream* out) {
    /*
    std::cout << "Writing the following quant: " << std::endl;
    for(int i = 0; i < length; i++) {
        std::cout << std::to_string(data[i]) << " ";
    }
    std::cout << std::endl;
    */

    int bytes_accounted_for = 0;

    int cur_offset = 0;
    while(cur_offset < length - 3) {
        /*
        if(cur_offset > 64) {
            throw std::runtime_error("fuck");
        }*/
        int current_value = data[cur_offset];
        int next = data[cur_offset + 1];
        int next_next = data[cur_offset + 2];
        // If there is a lot of similarity we issue a "same value" opcode
        if( current_value == next && next == next_next) {
            int end = cur_offset;
            while(end != length && end-cur_offset != 0x) {
                if(data[end] != current_value) {
                    break;
                }
                end++;
            }
            end--;
            int repeat_count = end - cur_offset + 1;
            std::cout << "Repeat: " << std::to_string(repeat_count) << "x " << std::to_string(current_value) << std::endl;

            write_repeat(current_value, repeat_count, out);

            bytes_accounted_for += repeat_count;
            cur_offset = end + 1;
        } else {
            //std::cout << "TEST: " << cur_offset << std::endl;
            // Encode "arbitrary content"
            int end = cur_offset;
            while(end != length - 3) {
                int test_cur = data[end];
                int test_next = data[end + 1];
                int test_next_next = data[end + 2];
                if(test_cur == test_next && test_next_next == test_next) {
                    break;
                }
                end++;
            }
            int arbitrary_count = (end - cur_offset);
            std::cout << "Arbitrary: " << arbitrary_count << " values: ";
            for(int i = cur_offset; i < end; i++) {
                std::cout << std::to_string(data[i]) << " ";
            }
            std::cout << std::endl;

            write_arbitrary(&data[cur_offset], arbitrary_count, out);

            bytes_accounted_for += arbitrary_count;
            cur_offset = end;
        }
    }
    if(cur_offset < length) {
        int arbitrary_count= (length - cur_offset);
        std::cout << "END: " << arbitrary_count << "x arbitrary content:";
        for(int i = cur_offset; i < length; i++) {
            std::cout << std::to_string(data[i]) << " ";
        }
        std::cout << std::endl;

        write_arbitrary(&data[cur_offset], arbitrary_count, out);

        bytes_accounted_for += arbitrary_count;
    }
    if(bytes_accounted_for != length) {
        throw std::runtime_error("Invalid bytes_accounted_for - bug!");
    }
}

void read_run_time_length(int* data, int length, std::ifstream* in) {
    int at = 0;
    while(at < length) {
        unsigned char opcode;
        in->read((char*)&opcode, sizeof(opcode));
        std::cout << "Opcode: " << std::to_string(opcode) << std::endl;

        int amount = (opcode & LEN_MASK);
        std::cout << "Reading " << amount << std::endl;

        if((opcode & OPTYPE_MASK) == SAME_VALUE) {
            std::cout << "Same value" << std::endl;
            int value = 0;
            if((opcode & DATATYPE_MASK) == SHORT_MASK) {
                std::cout << "Read short" << std::endl;
                short val;
                in->read((char*)&val, sizeof(val));
                std::cout << "read " << val << std::endl;

                value = static_cast<int>(val);
            } else if((opcode & DATATYPE_MASK) == BYTE_MASK) {
                std::cout << "Read byte" << std::endl;
                char val;
                in->read((char*)&val, sizeof(val));
                std::cout << "read " << std::to_string(val) << std::endl;

                value = static_cast<int>(val);
            }
            for(int i = 0; i < amount; i++) {
                data[at++] = value;
            }
        } else if((opcode & OPTYPE_MASK) == ARBITRARY_VALUE) {
            std::cout << "arbitrary" << std::endl;
            if((opcode & DATATYPE_MASK) == SHORT_MASK) {
                for(int i = 0; i < amount; i++) {
                    short val;
                    in->read((char*)&val, sizeof(val));
                    data[at++] = static_cast<int>(val);
                }
            } else if((opcode & DATATYPE_MASK) == BYTE_MASK) {
                for(int i = 0; i < amount; i++) {
                    char val;
                    in->read((char*)&val, sizeof(val));
                    data[at++] = static_cast<int>(val);
                }
            }

        }
    }
    /*
    std::cout << "Output quant: " << std::endl;
    for(int i = 0; i < length; i++) {
        std::cout << std::to_string(data[i]) << " ";
    }
    std::cout << std::endl;
    */
}
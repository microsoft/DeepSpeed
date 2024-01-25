#pragma once
#include <cstddef>
#include <cstdint>

/*
 * Inputs:
 * (1) unsigned char Weight_6bit [M*K*6/8]
 * Outputs:
 * (1) unsigned char Weight_2bit [M*K*2/8]
 * (2) unsigned char Weight_4bit [M*K*4/8]
 *
 * Assumption: Weight_6bit, Weight_2bit, Weight_4bit all stored continuously in row-major.
 * 8 FP6 = 6 Bytes
 * 8 FP4 = 4 Bytes
 * 8 FP2 = 2 Bytes
 */
void GenMatrix_Weight_FP6(unsigned char* Weight_6bit,
                          unsigned char* Weight_2bit,
                          unsigned char* Weight_4bit,
                          size_t M,
                          size_t K);


/*
 * Inputs:
 * (1) uint16_t Weight_16bit[M*K]
 * Outputs:
 * (1) unsigned char Weight_6bit[M*K*6/8]
 */
void PackMatrix_Weight_FP6(uint16_t* Weight_16bit, unsigned char* Weight_6bit, size_t M, size_t K);

#include <cuda.h>

__device__ uint32_t Generate(volatile uint64_t &state, uint64_t inc){
    
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((state >> 18u)^state) >> 27u;
    uint32_t rot = state >> 59u;
    // Update state
    state = (state * 6364136223846793005ULL + 2*inc+1);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}



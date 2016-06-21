#include <iostream>
#include <arm_neon.h>
#include "rng.hpp"

inline uint16x8_t cv_vpacks_u32(uint32x4_t v0, uint32x4_t v1)
{
	return vcombine_u16(vqmovn_u32(v0), vqmovn_u32(v1));
}

inline uint8x16_t cv_vpacks_u16(uint16x8_t v0, uint16x8_t v1)
{
	return vcombine_u8(vqmovn_u16(v0), vqmovn_u16(v1));
}

inline int16x8_t cv_vpacks_s32(int32x4_t v0, int32x4_t v1)
{
	return vcombine_s16(vqmovn_s32(v0), vqmovn_s32(v1));
}

inline int8x16_t cv_vpacks_s16(int16x8_t v0, int16x8_t v1)
{
	return vcombine_s8(vqmovn_s16(v0), vqmovn_s16(v1));
}

inline uint16x8_t cv_vpack_u32(uint32x4_t v0, uint32x4_t v1)
{
	return vcombine_u16(vmovn_u32(v0), vmovn_u32(v1));
}

inline uint8x16_t cv_vpack_u16(uint16x8_t v0, uint16x8_t v1)
{
	return vcombine_u8(vmovn_u16(v0), vmovn_u16(v1));
}

inline int16x8_t cv_vpack_s32(int32x4_t v0, int32x4_t v1)
{
	return vcombine_s16(vmovn_s32(v0), vmovn_s32(v1));
}

inline int8x16_t cv_vpack_s16(int16x8_t v0, int16x8_t v1)
{
	return vcombine_s8(vmovn_s16(v0), vmovn_s16(v1));
}

int main(int argc, char** argv)
{
	RNG r(0x123);
	return 0;
}

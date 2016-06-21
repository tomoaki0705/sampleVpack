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

void fill(uint32_t* ptr, RNG& r)
{
	for(uint32_t i = 0;i < 4;i++)
		ptr[i] = (r.next() & 0x7fff);
}

void debugPack(const uint32_t* src0, const uint32_t* src1, uint16_t* dst)
{
	for(uint32_t i = 0;i < (16/sizeof(uint32_t));i++)
	{
		dst[i*2  ] = (uint16_t)(src0[i]);
		dst[i*2+1] = (uint16_t)(src1[i]);
	}
}

int main(int argc, char** argv)
{
	RNG r(0x123);
	uint32_t src0[4], src1[4];
	uint16_t dst[8];
	fill(src0, r);
	fill(src1, r);
	debugPack((const uint32_t*)src0, (const uint32_t*)src1, (uint16_t*)dst);
	std::cout << "src0:";
	for(uint32_t i = 0;i < 4;i++)
	{
		std::cout << '\t' << src0[i];
	}
	std::cout << std::endl;
	std::cout << "src1:";
	for(uint32_t i = 0;i < 4;i++)
	{
		std::cout << '\t' << src1[i];
	}
	std::cout << std::endl;
	std::cout << "dst :";
	for(uint32_t i = 0;i < 8;i++)
	{
		std::cout << '\t' << dst[i];
	}
	std::cout << std::endl;

	return 0;
}

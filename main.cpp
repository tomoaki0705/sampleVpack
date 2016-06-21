#include <iostream>
#include <arm_neon.h>
#include "rng.hpp"

template <typename ST, typename DT>
inline DT cv_vpacks_(ST v0, ST v1)
{
	DT a;
	return a;
}

template <typename ST, typename DT>
inline DT cv_vpack_(ST v0, ST v1)
{
	DT a;
	return a;
}

template <>
inline uint16x8_t cv_vpacks_(uint32x4_t v0, uint32x4_t v1)
{
	return vcombine_u16(vqmovn_u32(v0), vqmovn_u32(v1));
}

template <>
inline uint8x16_t cv_vpacks_(uint16x8_t v0, uint16x8_t v1)
{
	return vcombine_u8(vqmovn_u16(v0), vqmovn_u16(v1));
}

template <>
inline int16x8_t cv_vpacks_(int32x4_t v0, int32x4_t v1)
{
	return vcombine_s16(vqmovn_s32(v0), vqmovn_s32(v1));
}

template <>
inline int8x16_t cv_vpacks_(int16x8_t v0, int16x8_t v1)
{
	return vcombine_s8(vqmovn_s16(v0), vqmovn_s16(v1));
}

template <>
inline uint16x8_t cv_vpack_(uint32x4_t v0, uint32x4_t v1)
{
	return vcombine_u16(vmovn_u32(v0), vmovn_u32(v1));
}

template <>
inline uint8x16_t cv_vpack_(uint16x8_t v0, uint16x8_t v1)
{
	return vcombine_u8(vmovn_u16(v0), vmovn_u16(v1));
}

template <>
inline int16x8_t cv_vpack_(int32x4_t v0, int32x4_t v1)
{
	return vcombine_s16(vmovn_s32(v0), vmovn_s32(v1));
}

template <>
inline int8x16_t cv_vpack_(int16x8_t v0, int16x8_t v1)
{
	return vcombine_s8(vmovn_s16(v0), vmovn_s16(v1));
}


template <typename ST>
void fill(ST* ptr, RNG& r)
{
	unsigned int cLoop = 16/sizeof(ST);
	for(unsigned int i = 0;i < cLoop;i++)
		ptr[i] = (r.next() & 0x7f);
}

template <typename ST, typename DT>
void debugPack(const ST* src0, const ST* src1, DT* dst)
{
	for(uint32_t i = 0;i < (16/sizeof(ST));i++)
	{
		dst[i*2  ] = (DT)(src0[i]);
		dst[i*2+1] = (DT)(src1[i]);
	}
}

template <typename ST, typename DT>
void dumpArray(const ST* src0, const ST* src1, const DT* dst)
{
	using namespace std;
	unsigned int srcLoop = 16/sizeof(ST);
	unsigned int dstLoop = 16/sizeof(DT);
	cout << "src0:";
	for(unsigned int i = 0;i < srcLoop;i++)
	{
		cout << '\t' << src0[i];
	}
	cout << endl;
	cout << "src1:";
	for(unsigned int i = 0;i < srcLoop;i++)
	{
		cout << '\t' << src1[i];
	}
	cout << endl;
	cout << "dst :";
	for(unsigned int i = 0;i < dstLoop;i++)
	{
		cout << '\t' << dst[i];
	}
	cout << endl;
}

template <typename ST, typename DT>
void verifyArray(ST* src0, ST* src1, DT* dst, RNG& r)
{
	fill(src0, r);
	fill(src1, r);
	debugPack((const ST*)src0, (const ST*)src1, (DT*)dst);
	dumpArray(src0, src1, dst);
}

int main(int argc, char** argv)
{
	RNG r(0x123);
	uint32_t src0[4], src1[4];
	uint16_t dst[8];
	verifyArray(src0, src1, dst, r);

	return 0;
}

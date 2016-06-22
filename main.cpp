#include <iostream>
#include <iomanip>
#include <arm_neon.h>
#include "rng.hpp"
const unsigned int cVerifyLoop = 1000;
const unsigned int cProgress   = (cVerifyLoop/10);

inline uint32x4_t cv_vunpack_lo_u32(uint32x4_t v0, uint32x4_t v1)
{
	uint32x2x2_t result = vzip_u32(vget_low_u32(v0), vget_low_u32(v1));
	return vcombine_u32(result.val[0], result.val[1]);
}

inline uint32x4_t cv_vunpack_hi_u32(uint32x4_t v0, uint32x4_t v1)
{
	uint32x2x2_t result = vzip_u32(vget_high_u32(v0), vget_high_u32(v1));
	return vcombine_u32(result.val[0], result.val[1]);
}

inline uint16x8_t cv_vunpack_lo_u16(uint16x8_t v0, uint16x8_t v1)
{
	uint16x4x2_t result = vzip_u16(vget_low_u16(v0), vget_low_u16(v1));
	return vcombine_u16(result.val[0], result.val[1]);
}

inline uint16x8_t cv_vunpack_hi_u16(uint16x8_t v0, uint16x8_t v1)
{
	uint16x4x2_t result = vzip_u16(vget_high_u16(v0), vget_high_u16(v1));
	return vcombine_u16(result.val[0], result.val[1]);
}

inline uint8x16_t cv_vunpack_lo_u8(uint8x16_t v0, uint8x16_t v1)
{
	uint8x8x2_t result = vzip_u8(vget_low_u8(v0), vget_low_u8(v1));
	return vcombine_u8(result.val[0], result.val[1]);
}

inline uint8x16_t cv_vunpack_hi_u8(uint8x16_t v0, uint8x16_t v1)
{
	uint8x8x2_t result = vzip_u8(vget_high_u8(v0), vget_high_u8(v1));
	return vcombine_u8(result.val[0], result.val[1]);
}

inline int32x4_t cv_vunpack_lo_s32(int32x4_t v0, int32x4_t v1)
{
	int32x2x2_t result = vzip_s32(vget_low_s32(v0), vget_low_s32(v1));
	return vcombine_s32(result.val[0], result.val[1]);
}

inline int32x4_t cv_vunpack_hi_s32(int32x4_t v0, int32x4_t v1)
{
	int32x2x2_t result = vzip_s32(vget_high_s32(v0), vget_high_s32(v1));
	return vcombine_s32(result.val[0], result.val[1]);
}

inline int16x8_t cv_vunpack_lo_s16(int16x8_t v0, int16x8_t v1)
{
	int16x4x2_t result = vzip_s16(vget_low_s16(v0), vget_low_s16(v1));
	return vcombine_s16(result.val[0], result.val[1]);
}

inline int16x8_t cv_vunpack_hi_s16(int16x8_t v0, int16x8_t v1)
{
	int16x4x2_t result = vzip_s16(vget_high_s16(v0), vget_high_s16(v1));
	return vcombine_s16(result.val[0], result.val[1]);
}

inline int8x16_t cv_vunpack_lo_s8(int8x16_t v0, int8x16_t v1)
{
	int8x8x2_t result = vzip_s8(vget_low_s8(v0), vget_low_s8(v1));
	return vcombine_s8(result.val[0], result.val[1]);
}

inline int8x16_t cv_vunpack_hi_s8(int8x16_t v0, int8x16_t v1)
{
	int8x8x2_t result = vzip_s8(vget_high_s8(v0), vget_high_s8(v1));
	return vcombine_s8(result.val[0], result.val[1]);
}

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


template <typename ST>
void fill(ST* ptr, RNG& r)
{
	unsigned int cLoop = 16/sizeof(ST);
	for(unsigned int i = 0;i < cLoop;i++)
		ptr[i] = r.next();
}

template <typename T>
void debugUnpack(const T* src0, const T* src1, T* dst0, T* dst1)
{
	unsigned int cLoop = (16 / sizeof(T)) >> 1;
	for(unsigned int i = 0;i < cLoop;i++)
	{
		dst0[i*2  ] = src0[i];
		dst0[i*2+1] = src1[i];
		dst1[i*2  ] = src0[i+cLoop];
		dst1[i*2+1] = src1[i+cLoop];
	}
}

template <typename T>
void dumpArray(const char* header, const T* src0)
{
	using namespace std;
	unsigned int cLoop = 16/sizeof(T);
	cout << header;
	for(unsigned int i = 0;i < cLoop;i++)
	{
		cout << '\t' << src0[i];
	}
	cout << endl;
}

template <typename T>
void debugUnpackVector(const T* src0, const T* src1, T* dstLow, T* dstHigh)
{
	return;
}

template <typename ST, typename DT>
void debugVector(const ST* src0, const ST* src1, DT* dst)
{
	return;
}

template <>
void debugUnpackVector(const int32_t* src0, const int32_t* src1, int32_t* dstLow, int32_t* dstHigh)
{
	int32x4_t v0 = vld1q_s32(src0);
	int32x4_t v1 = vld1q_s32(src1);
	int32x4_t d0 = cv_vunpack_lo_s32(v0, v1);
	int32x4_t d1 = cv_vunpack_hi_s32(v0, v1);
	vst1q_s32(dstLow,  d0);
	vst1q_s32(dstHigh, d1);
	return;
}

template <>
void debugUnpackVector(const int16_t* src0, const int16_t* src1, int16_t* dstLow, int16_t* dstHigh)
{
	int16x8_t v0 = vld1q_s16(src0);
	int16x8_t v1 = vld1q_s16(src1);
	int16x8_t d0 = cv_vunpack_lo_s16(v0, v1);
	int16x8_t d1 = cv_vunpack_hi_s16(v0, v1);
	vst1q_s16(dstLow,  d0);
	vst1q_s16(dstHigh, d1);
	return;
}

template <>
void debugUnpackVector(const int8_t* src0, const int8_t* src1, int8_t* dstLow, int8_t* dstHigh)
{
	int8x16_t v0 = vld1q_s8(src0);
	int8x16_t v1 = vld1q_s8(src1);
	int8x16_t d0 = cv_vunpack_lo_s8(v0, v1);
	int8x16_t d1 = cv_vunpack_hi_s8(v0, v1);
	vst1q_s8(dstLow,  d0);
	vst1q_s8(dstHigh, d1);
	return;
}

template <>
void debugUnpackVector(const uint32_t* src0, const uint32_t* src1, uint32_t* dstLow, uint32_t* dstHigh)
{
	uint32x4_t v0 = vld1q_u32(src0);
	uint32x4_t v1 = vld1q_u32(src1);
	uint32x4_t d0 = cv_vunpack_lo_u32(v0, v1);
	uint32x4_t d1 = cv_vunpack_hi_u32(v0, v1);
	vst1q_u32(dstLow,  d0);
	vst1q_u32(dstHigh, d1);
	return;
}

template <>
void debugUnpackVector(const uint16_t* src0, const uint16_t* src1, uint16_t* dstLow, uint16_t* dstHigh)
{
	uint16x8_t v0 = vld1q_u16(src0);
	uint16x8_t v1 = vld1q_u16(src1);
	uint16x8_t d0 = cv_vunpack_lo_u16(v0, v1);
	uint16x8_t d1 = cv_vunpack_hi_u16(v0, v1);
	vst1q_u16(dstLow,  d0);
	vst1q_u16(dstHigh, d1);
	return;
}

template <>
void debugUnpackVector(const uint8_t* src0, const uint8_t* src1, uint8_t* dstLow, uint8_t* dstHigh)
{
	uint8x16_t v0 = vld1q_u8(src0);
	uint8x16_t v1 = vld1q_u8(src1);
	uint8x16_t d0 = cv_vunpack_lo_u8(v0, v1);
	uint8x16_t d1 = cv_vunpack_hi_u8(v0, v1);
	vst1q_u8(dstLow,  d0);
	vst1q_u8(dstHigh, d1);
	return;
}

template <>
void debugVector(const int32_t* src0, const int32_t* src1, int16_t* dst)
{
	int32x4_t v0 = vld1q_s32(src0);
	int32x4_t v1 = vld1q_s32(src1);
	int16x8_t d0 = cv_vpack_s32(v0, v1);
	vst1q_s16(dst, d0);
	return;
}

template <>
void debugVector(const int16_t* src0, const int16_t* src1, int8_t* dst)
{
	int16x8_t v0 = vld1q_s16(src0);
	int16x8_t v1 = vld1q_s16(src1);
	int8x16_t d0 = cv_vpack_s16(v0, v1);
	vst1q_s8(dst, d0);
	return;
}

template <>
void debugVector(const uint32_t* src0, const uint32_t* src1, uint16_t* dst)
{
	uint32x4_t v0 = vld1q_u32(src0);
	uint32x4_t v1 = vld1q_u32(src1);
	uint16x8_t d0 = cv_vpack_u32(v0, v1);
	vst1q_u16(dst, d0);
	return;
}

template <>
void debugVector(const uint16_t* src0, const uint16_t* src1, uint8_t* dst)
{
	uint16x8_t v0 = vld1q_u16(src0);
	uint16x8_t v1 = vld1q_u16(src1);
	uint8x16_t d0 = cv_vpack_u16(v0, v1);
	vst1q_u8(dst, d0);
	return;
}

template <typename T>
void verifyArray(T* src0, T* src1, T* dst_l, T* dst_h, T* dst_v0, T* dst_v1, RNG& r)
{
	fill(src0, r);
	fill(src1, r);
	debugUnpack((const T*)src0, (const T*)src1, (T*)dst_l,  (T*)dst_h);
	debugUnpackVector((const T*)src0, (const T*)src1, (T*)dst_v0, (T*)dst_v1);
	unsigned int cLoop = 16/sizeof(T);
	bool hasDifference = false;
	for(unsigned int i = 0;i < cLoop;i++)
	{
		if(dst_l[i] != dst_v0[i])
		{
			hasDifference = true;
			break;
		}
		if(dst_h[i] != dst_v1[i])
		{
			hasDifference = true;
			break;
		}
	}
	if(hasDifference)
	{
		dumpArray("src0:", src0);
		dumpArray("src1:", src1);
		dumpArray("dstL:", dst_l);
		dumpArray("vec0:", dst_v0);
		dumpArray("dstH:", dst_h);
		dumpArray("vec1:", dst_v1);
	}
}

void showProgress(unsigned int i)
{
	if(((i / cProgress) * cProgress) == i)
		std::cout << "Passed " << std::setw(4) << std::setfill(' ') << i << " / " << cVerifyLoop << std::endl;
	return ;
}

int main(int argc, char** argv)
{
	RNG r(0x123);
	uint32_t src0[4],   src1[4];
	uint32_t dst_l[4],  dst_h[4];
	uint32_t dst_v0[4], dst_v1[4];
	std::cout << "verify uint32" << std::endl;
	for(unsigned int i = 0;i < cVerifyLoop;i++)
	{
		showProgress(i);
		verifyArray(src0, src1, dst_l, dst_h, dst_v0, dst_v1, r);
	}
	std::cout << "verify uint16" << std::endl;
	for(unsigned int i = 0;i < cVerifyLoop;i++)
	{
		showProgress(i);
		verifyArray((uint16_t*)src0, (uint16_t*)src1, (uint16_t*)dst_l, (uint16_t*)dst_h, (uint16_t*)dst_v0, (uint16_t*)dst_v1, r);
	}
	std::cout << "verify uint8" << std::endl;
	for(unsigned int i = 0;i < cVerifyLoop;i++)
	{
		showProgress(i);
		verifyArray((uint8_t*)src0, (uint8_t*)src1, (uint8_t*)dst_l, (uint8_t*)dst_h, (uint8_t*)dst_v0, (uint8_t*)dst_v1, r);
	}

	return 0;
}

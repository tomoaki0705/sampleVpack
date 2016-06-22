#include <limits.h>

template<typename _Tp> static inline _Tp saturate_cast(uint8_t v)  { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(int8_t v)   { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(uint16_t v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(int16_t v)  { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(uint32_t v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(int32_t v)  { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(float v)    { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(double v)   { return _Tp(v); }

template<> inline uint8_t saturate_cast<uint8_t>(int8_t v)         { return (uint8_t)std::max((int32_t)v, 0); }
template<> inline uint8_t saturate_cast<uint8_t>(uint16_t v)       { return (uint8_t)std::min((uint32_t)v, (uint32_t)UCHAR_MAX); }
template<> inline uint8_t saturate_cast<uint8_t>(int32_t v)        { return (uint8_t)((uint32_t)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline uint8_t saturate_cast<uint8_t>(int16_t v)        { return saturate_cast<uint8_t>((int32_t)v); }
template<> inline uint8_t saturate_cast<uint8_t>(uint32_t v)       { return (uint8_t)std::min(v, (uint32_t)UCHAR_MAX); }

template<> inline int8_t saturate_cast<int8_t>(uint8_t v)          { return (int8_t)std::min((int32_t)v, SCHAR_MAX); }
template<> inline int8_t saturate_cast<int8_t>(uint16_t v)         { return (int8_t)std::min((uint32_t)v, (uint32_t)SCHAR_MAX); }
template<> inline int8_t saturate_cast<int8_t>(int32_t v)          { return (int8_t)((uint32_t)(v-SCHAR_MIN) <= (uint32_t)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline int8_t saturate_cast<int8_t>(int16_t v)          { return saturate_cast<int8_t>((int32_t)v); }
template<> inline int8_t saturate_cast<int8_t>(uint32_t v)         { return (int8_t)std::min(v, (uint32_t)SCHAR_MAX); }

template<> inline uint16_t saturate_cast<uint16_t>(int8_t v)       { return (uint16_t)std::max((int32_t)v, 0); }
template<> inline uint16_t saturate_cast<uint16_t>(int16_t v)      { return (uint16_t)std::max((int32_t)v, 0); }
template<> inline uint16_t saturate_cast<uint16_t>(int32_t v)      { return (uint16_t)((uint32_t)v <= (uint32_t)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline uint16_t saturate_cast<uint16_t>(uint32_t v)     { return (uint16_t)std::min(v, (uint32_t)USHRT_MAX); }

template<> inline int16_t saturate_cast<int16_t>(uint16_t v)       { return (int16_t)std::min((int32_t)v, SHRT_MAX); }
template<> inline int16_t saturate_cast<int16_t>(int32_t v)        { return (int16_t)((uint32_t)(v - SHRT_MIN) <= (uint32_t)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline int16_t saturate_cast<int16_t>(uint32_t v)       { return (int16_t)std::min(v, (uint32_t)SHRT_MAX); }



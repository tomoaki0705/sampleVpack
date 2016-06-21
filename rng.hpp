typedef uint64_t uint64;

class RNG
{
public:
	inline RNG()              { state = 0xffffffff; }
	inline RNG(uint64 _state) { state = _state ? _state : 0xffffffff; }
	uint64 state;
	inline unsigned next()
	{
		state = (uint64)(unsigned)state * 4164903690U + (unsigned)(state >> 32);
		return (unsigned)state;
	}
};


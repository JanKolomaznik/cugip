#pragma once

#define CUGIL_DECL_HOST __host
#define CUGIL_DECL_DEVICE __device
#define CUGIL_DECL_HYBRID CUGIL_DECL_HOST CUGIL_DECL_DEVICE

namespace cugip {

template<typename TType>
struct device_ptr
{
	CUGIL_DECL_HYBRID
	device_ptr(): p(0) 
	{ /*empty*/ }

	CUGIL_DECL_HYBRID
	device_ptr(const device_ptr &aArg): p(aArg.p) 
	{ /*empty*/ }

	CUGIL_DECL_HYBRID TType * 
	operator->()
	{ return p; }

	CUGIL_DECL_HYBRID device_ptr &
	operator=(const device_ptr &aArg)
	{ p = aArg.p; }

	CUGIL_DECL_HYBRID device_ptr &
	operator=(TType *aArg)
	{ p = aArg; }

	CUGIL_DECL_HYBRID
	operator bool() const
	{ return p != 0; }

	TType *p;
};

template<typename TType, int tChannelCount>
struct element
{
	TType data[tChannelCount];
};

typedef element<unsigned char, 3> element_rgb8_t;



}//namespace cugip

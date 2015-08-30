#pragma once

#include <memory>

class AAutomatonWrapper
{
public:
	virtual
	~AAutomatonWrapper() {}

	virtual void
	runIterations(int aIterationCount) = 0;

	virtual void
	setStartImage(const unsigned char *aBuffer, int aWidth, int aHeight, int aBytesPerLine) = 0;

	virtual void
	getCurrentImage(unsigned char *aBuffer, int aWidth, int aHeight, int aBytesPerLine) = 0;
};

std::unique_ptr<AAutomatonWrapper>
getConwaysAutomatonWrapper();

std::unique_ptr<AAutomatonWrapper>
getCCLAutomatonWrapper();

std::unique_ptr<AAutomatonWrapper>
getCCLAutomatonWrapper2();

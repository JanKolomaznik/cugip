#pragma once

#include <memory>

class AAutomatonWrapper
{
public:
	virtual
	~AAutomatonWrapper() {}

	virtual void
	runIterations(int aIterationCount) = 0;

	virtual bool
	preprocessingEnabled() const
	{
		return mPreprocessingEnabled;
	}

	virtual void
	enablePreprocessing(bool aEnable)
	{
		mPreprocessingEnabled = aEnable;
	}

	virtual void
	setStartImage(const unsigned char *aBuffer, int aWidth, int aHeight, int aBytesPerLine) = 0;

	virtual void
	getCurrentImage(unsigned char *aBuffer, int aWidth, int aHeight, int aBytesPerLine) = 0;

	bool mPreprocessingEnabled = true;
};

std::unique_ptr<AAutomatonWrapper>
getConwaysAutomatonWrapper();

std::unique_ptr<AAutomatonWrapper>
getCCLAutomatonWrapper();

std::unique_ptr<AAutomatonWrapper>
getCCLAutomatonWrapper2();

std::unique_ptr<AAutomatonWrapper>
getWShedAutomatonWrapper();

std::unique_ptr<AAutomatonWrapper>
getWShedAutomaton2Wrapper();


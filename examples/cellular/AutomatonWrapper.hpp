#pragma once

#include <memory>
#include <string>

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

	virtual std::string
	getInfoOnPosition(int aX, int aY) const
	{
		return std::to_string(aX) + std::string(" x ") + std::to_string(aY);
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

std::unique_ptr<AAutomatonWrapper>
getReactionDiffusionWrapper();

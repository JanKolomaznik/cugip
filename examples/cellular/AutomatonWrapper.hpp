#pragma once

#include <memory>
#include <string>
#include <map>
#include <vector>
#include <functional>

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

typedef std::vector<std::pair<std::string, std::function<std::unique_ptr<AAutomatonWrapper>()>>> AutomatonFactoryMap;
//typedef std::map<std::string, std::function<std::unique_ptr<AAutomatonWrapper>()>> AutomatonFactoryMap;

const AutomatonFactoryMap & automatonFactoryMap();

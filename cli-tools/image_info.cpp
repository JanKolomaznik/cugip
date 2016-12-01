// This is different from CropImageFilter only in the way
// that the region to crop is specified.
#include "itkImage.h"
#include <itkImageFileReader.h>
#include <itkExtractImageFilter.h>
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include <itkSize.h>
#include <itkIndex.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <map>
#include <iostream>
#include <iomanip>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>


#include <cugip/host_image_view.hpp>
#include <cugip/for_each.hpp>

namespace ba = boost::accumulators;
namespace po = boost::program_options;

typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

template<typename TType>
void imageStatisticsImplementation(std::string aPath)
{
	const unsigned int Dimension = 3;
	typedef TType PixelType;
	typedef itk::Image<PixelType, Dimension> ImageType;
	typedef itk::ImageFileReader<ImageType> ReaderType;
	typename ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(aPath);
	reader->Update();
	typename ImageType::Pointer image = reader->GetOutput();

	ba::accumulator_set< double, ba::features<ba::tag::min, ba::tag::max/*, ba::tag::mean*/>> acc;

	cugip::simple_vector<int, 3> size;
	for (int i = 0; i < 3; ++i) {
		size[i] = image->GetLargestPossibleRegion().GetSize()[i];
	}
	auto imageView = makeConstHostImageView(image->GetPixelContainer()->GetBufferPointer(), size);

	cugip::for_each(
		imageView,
		[&](const TType &aValue) {
			acc(aValue);
		});

	std::cout << "Min " << ba::extract_result<ba::tag::min>(acc) << '\n';
	std::cout << "Max " << ba::extract_result<ba::tag::max>(acc) << '\n';
	//std::cout << "Mean " << ba::extract_result<ba::tag::mean>(acc) << '\n';
}

void imageStatistics(std::string aPath, const ScalarPixelType aPixelType)
{
	static const std::map<ScalarPixelType, std::function<void (std::string)>> functions = {
		{ itk::ImageIOBase::UCHAR, &imageStatisticsImplementation<unsigned char> },
		{ itk::ImageIOBase::CHAR, &imageStatisticsImplementation<signed char> },
		{ itk::ImageIOBase::USHORT, &imageStatisticsImplementation<unsigned short> },
		{ itk::ImageIOBase::SHORT, &imageStatisticsImplementation<short> },
		{ itk::ImageIOBase::UINT, &imageStatisticsImplementation<unsigned int> },
		{ itk::ImageIOBase::INT, &imageStatisticsImplementation<int> },
		{ itk::ImageIOBase::ULONG, &imageStatisticsImplementation<unsigned long> },
		{ itk::ImageIOBase::LONG, &imageStatisticsImplementation<long> },
		{ itk::ImageIOBase::FLOAT, &imageStatisticsImplementation<float> },
		{ itk::ImageIOBase::DOUBLE, &imageStatisticsImplementation<double> }
	};
	auto it = functions.find(aPixelType);
	if (it != functions.end()) {
		it->second(aPath);
	} else {
		std::cout << "Statistics not available for element type.\n";
	}
}

int main(int argc, char *argv[])
{
	std::string input_file;
	bool enableStatistics = false;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<std::string>(&input_file), "input file")
		("stats,s", po::bool_switch(&enableStatistics), "compute statistics");
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 1;
	}

	if (vm.count("input") == 0) {
		std::cout << "Missing input filename\n" << desc << "\n";
		return 1;
	}

	try {
		itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(input_file.c_str(), itk::ImageIOFactory::ReadMode);
		imageIO->SetFileName(input_file);
		imageIO->ReadImageInformation();
		const ScalarPixelType pixelType = imageIO->GetComponentType();

		std::cout << boost::str(boost::format(
			"Component type: %1%\n"
			"Component count: %2%\n"
			"Dimensions: [%3%, %4%, %5%]\n")
			% imageIO->GetComponentType()
			% imageIO->GetNumberOfComponents()
			% imageIO->GetDimensions(0)
			% imageIO->GetDimensions(1)
			% imageIO->GetDimensions(2));
		if (enableStatistics) {
			imageStatistics(input_file, pixelType);
		}
	}
	catch( itk::ExceptionObject & err )
	{
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

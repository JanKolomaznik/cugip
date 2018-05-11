#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <itkSpatialObjectToImageFilter.h>
#include <itkEllipseSpatialObject.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <boost/spirit/home/x3.hpp>
#include <boost/format.hpp>
#include <boost/fusion/adapted/std_tuple.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/image_locator.hpp>
#include <cugip/for_each.hpp>
#include <cugip/fill.hpp>

#include <limits>
#include <random>

using namespace cugip;
namespace x3 = boost::spirit::x3;
namespace ascii = boost::spirit::x3::ascii;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

const unsigned int Dimension = 3;
typedef float                              PixelType;
typedef itk::Image< PixelType, Dimension > ImageType;


void computeDistanceField(cugip::host_image_view<float, 3> field);


template<typename TView>
void generateRandomSeeds(TView data, int seedCount)
{
	std::default_random_engine generator;
	using Distribution = std::uniform_int_distribution<int>;
	Distribution distributions[3] = {
		Distribution(0, data.dimensions()[0] - 1),
		Distribution(0, data.dimensions()[1] - 1),
		Distribution(0, data.dimensions()[2] - 1) };

	for (int i = 0; i < seedCount; ++i) {
		auto coordinates = cugip::vect3i_t(
			distributions[0](generator),
			distributions[1](generator),
			distributions[2](generator));

		data[coordinates] = 0;
	}
}


template<typename TImageType>
void generateData(
	fs::path aOutputPath,
	typename TImageType::SizeType size,
	int seedCount
	)
{
	typename TImageType::RegionType region;
	typename TImageType::IndexType start;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;

	region.SetSize(size);
	region.SetIndex(start);

	typename TImageType::Pointer image = TImageType::New();
	image->SetRegions(region);
	image->Allocate();

	cugip::simple_vector<int, 3> tmpSize;
	for (int i = 0; i < 3; ++i) {
		tmpSize[i] = image->GetLargestPossibleRegion().GetSize()[i];
	}
	auto outView = makeHostImageView(image->GetPixelContainer()->GetBufferPointer(), tmpSize);

	cugip::fill(outView, std::numeric_limits<float>::max());
	generateRandomSeeds(outView, seedCount);
	std::cout << "Seeds generated\n";
	computeDistanceField(outView);

	typedef itk::ImageFileWriter<TImageType> WriterType;
	typename WriterType::Pointer writer = WriterType::New();

	std::cout << "Writing file " << aOutputPath.string() << '\n';
	writer->SetFileName(aOutputPath.string());
	writer->SetInput(image);
	writer->Update();
}


typedef std::tuple<int, int, int> Triple;
Triple parseResolution(std::string aResolution) {
	using x3::int_;
	using ascii::blank;

//	auto parserRule = x3::rule<class size_tag, Triple>{} = int_ >> 'x' >> int_ >> 'x' >> int_;
	auto iter = aResolution.begin();

	Triple result;

	bool const res = x3::phrase_parse(iter, aResolution.end(), int_ >> 'x' >> int_ >> 'x' >> int_, blank, result);
	//bool const res = x3::phrase_parse(iter, aResolution.end(), parserRule, blank, result);
	if (!res || iter != aResolution.end()) {
		throw 1;
	}

	return result;
}

int main( int argc, char* argv[] )
{
	fs::path input_file;
	fs::path output_file;
	double sigma;

	std::string resolution;

	int seedCount;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("output,o", po::value<fs::path>(&output_file), "output file")
		("resolution,r", po::value<std::string>(&resolution), "Resolution WIDTHxHEIGHTxDEPTH")
		("seed,s", po::value<int>(&seedCount)->default_value(1), "Seed count")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);


	if (vm.count("help")) {
	    std::cout << desc << "\n";
	    return 1;
	}

	if (vm.count("output") == 0) {
	    std::cout << "Missing output filename\n" << desc << "\n";
	    return 1;
	}

	Triple size;
	if (vm.count("resolution") > 0) {
		try {
			size = parseResolution(resolution);
		} catch (...) {
		    std::cout << "Resolution in wrong format 20x20x20 expected\n" << desc << "\n";
		    return 1;
		}
	} else {
		std::get<0>(size) = 400;
		std::get<1>(size) = 400;
		std::get<2>(size) = 400;
	}


	try {
		std::cout << boost::format("Generating image of size [%1%, %2%, %3%] ...\n") % std::get<0>(size) % std::get<1>(size) % std::get<2>(size);
		ImageType::SizeType imageSize;
		imageSize[0] = std::get<0>(size);
		imageSize[1] = std::get<1>(size);
		imageSize[2] = std::get<2>(size);
		generateData<ImageType>(output_file, imageSize, seedCount);
	} catch( itk::ExceptionObject & error ) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <cugip/host_image_view.hpp>

#include "watershed_options.hpp"

#include <unordered_map>
#include <limits>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace cugip;

void runWatershedTransformation(
		const_host_image_view<const float, 2> aInput,
		host_image_view<int32_t, 2> aOutput,
		const WatershedOptions &aOptions);

void runWatershedTransformation(
		const_host_image_view<const float, 3> aInput,
		host_image_view<int32_t, 3> aOutput,
		const WatershedOptions &aOptions);

void runWatershedTransformation(
		const_host_image_view<const float, 2> aInput,
		host_image_view<int64_t, 2> aOutput,
		const WatershedOptions &aOptions);

void runWatershedTransformation(
		const_host_image_view<const float, 3> aInput,
		host_image_view<int64_t, 3> aOutput,
		const WatershedOptions &aOptions);

template<int tDimension, typename TLabel>
void processImage(fs::path aInput, fs::path aOutput, const WatershedOptions &aOptions)
{
	typedef itk::Image<float, tDimension> InputImageType;
	typedef itk::Image<TLabel, tDimension> OutputImageType;

	typedef itk::ImageFileReader<InputImageType>  ReaderType;
	typedef itk::ImageFileWriter<OutputImageType> WriterType;

	typename ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(aInput.string());
	reader->Update();

	typename InputImageType::Pointer image = reader->GetOutput();

	typename OutputImageType::Pointer output_image = OutputImageType::New();
	output_image->SetRegions(image->GetLargestPossibleRegion());
	output_image->Allocate();
	output_image->SetSpacing(image->GetSpacing());

	cugip::simple_vector<int, tDimension> size;
	for (int i = 0; i < tDimension; ++i) {
		size[i] = image->GetLargestPossibleRegion().GetSize()[i];
	}

	auto inView = makeConstHostImageView(image->GetPixelContainer()->GetBufferPointer(), size);
	auto outView = makeHostImageView(output_image->GetPixelContainer()->GetBufferPointer(), size);

	runWatershedTransformation(inView, outView, aOptions);

	typename WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(aOutput.string());
	writer->SetInput(output_image);
	writer->Update();
}

static const std::map<std::string, WatershedVariant> cMethodMapping = []() {
	std::map<std::string, WatershedVariant> map;
	map["distance"] = WatershedVariant::DistanceBased;
	map["distance_async"] = WatershedVariant::DistanceBasedAsync;
	map["distance_async_lim"] = WatershedVariant::DistanceBasedAsyncLimited;
	map["descent_pointer"] = WatershedVariant::SteepestDescentPointer;
	map["descent_pointer_2"] = WatershedVariant::SteepestDescentPointerTwoPhase;
	map["descent_simple"] = WatershedVariant::SteepestDescentSimple;
	map["descent_simple_async"] = WatershedVariant::SteepestDescentSimpleAsync;
	map["descent_gs"] = WatershedVariant::SteepestDescentGlobalState;

	return map;
}();

WatershedVariant getWatershedVariantFromString(const std::string &token)
{
	auto it = cMethodMapping.find(token);
	if (it == cMethodMapping.end()) {
		throw po::validation_error(po::validation_error::invalid_option_value);
	}
	return it->second;
}

struct HashTuple {
	size_t operator()(const std::tuple<int, bool> &t) const
	{
		std::hash<int> hash1;

		std::hash<bool> hash2;

		return (hash1(get<0>(t)) << 1) + hash2((get<1>(t)));
	}
};

int main( int argc, char* argv[] )
{
	using ProcessImageFunction = void (*)(fs::path, fs::path, const WatershedOptions &);
	static const std::unordered_map<std::tuple<int, bool>, ProcessImageFunction, HashTuple> cProcessImageFunctions = {
			{{2, false}, processImage<2, int32_t>},
			{{2, true}, processImage<2, int64_t>},
			{{3, false}, processImage<3, int32_t>},
			{{3, true}, processImage<3, int64_t>},
		};


	fs::path inputFile;
	fs::path outputFile;

	WatershedOptions options;
	std::string watershedName;
	bool force64bit = false;
	bool useDeviceMemory = false;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<fs::path>(&inputFile), "input file")
		("output,o", po::value<fs::path>(&outputFile), "output file")
		("watershed,w", po::value<std::string>(&watershedName)->default_value("distance"),
			[](){
				std::string res;
				for (auto name : cMethodMapping) {
					res += name.first + ", ";
				}
				return res;
			}().data())
		//("normalize,n", po::value<bool>(&normalize)->default_value(false), "normalize")
		("device,d", po::value<bool>(&useDeviceMemory)->default_value(false), "Use device memory.")
		("label64,l", po::value<bool>(&force64bit)->default_value(false), "force 64 bit labels even when not needed")
		;
		options.useUnifiedMemory = !useDeviceMemory;

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

	if (vm.count("output") == 0) {
		std::cout << "Missing output filename\n" << desc << "\n";
		return 1;
	}

	options.wshedVariant = getWatershedVariantFromString(watershedName);

	typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

	try {
		itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(inputFile.c_str(), itk::ImageIOFactory::ReadMode);
		imageIO->SetFileName(inputFile.string());
		imageIO->ReadImageInformation();
		const ScalarPixelType pixelType = imageIO->GetComponentType();
		std::cout << "Pixel Type is " << imageIO->GetComponentTypeAsString(pixelType) << std::endl;

		const size_t numDimensions =  imageIO->GetNumberOfDimensions();
		std::cout << "numDimensions: " << numDimensions << std::endl;

		if (imageIO->GetNumberOfComponents () != 1) {
			std::cerr << "Error: Only scalar images supported.\n";
			return EXIT_FAILURE;
		}

		int64_t elementCount = 1;
		for (int i = 0; i < numDimensions; ++i) {
			elementCount *= imageIO->GetDimensions(i);
		}
		bool need64bitLabels = force64bit;
		if (elementCount >= std::numeric_limits<int32_t>::max()) {
			need64bitLabels = true;
		}

		std::cout << "64 bit labels: " << (need64bitLabels ? "ON\n" : "OFF\n");
		std::cout << "Unified memory buffers: " << (options.useUnifiedMemory ? "ON\n" : "OFF\n");
		auto it = cProcessImageFunctions.find(std::make_tuple(numDimensions, need64bitLabels));
		if (it == cProcessImageFunctions.end()) {
			std::cerr << "Error: Unsupported image dimension.\n";
			throw 1;
		}

		it->second(inputFile, outputFile, options);

		/*switch (numDimensions) {
		case 2:
			processImage<2, int64_t>(inputFile, outputFile, options);
			break;
		case 3:
			processImage<3, int64_t>(inputFile, outputFile, options);
			break;
		default:
			std::cerr << "Error: Unsupported image dimension.\n";
			return EXIT_FAILURE;
		}*/

	} catch( itk::ExceptionObject & error ) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	} catch(...) {
		std::cerr << "Error: TODO" << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

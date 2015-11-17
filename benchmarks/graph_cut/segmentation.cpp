#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/filesystem.hpp>
#include <cugip/itk_utils.hpp>

#include <boost/log/trivial.hpp>
#include <string>

namespace po = boost::program_options;

typedef itk::Image<float, 3> ImageType;
typedef itk::Image<uint8_t, 3> MarkersType;
typedef itk::Image<uint8_t, 3> MaskType;


void
computeBoykovKolmogorovGrid(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::const_host_image_view<const uint8_t, 3> aMarkers,
	cugip::host_image_view<uint8_t, 3> aOutput,
	float aSigma,
	uint8_t aMaskValue);

void
computeGridCut(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::const_host_image_view<const uint8_t, 3> aMarkers,
	cugip::host_image_view<uint8_t, 3> aOutput,
	float aSigma,
	uint8_t aMaskValue);

void
computeCudaGraphCut(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::const_host_image_view<const uint8_t, 3> aMarkers,
	cugip::host_image_view<uint8_t, 3> aOutput,
	float aSigma,
	uint8_t aMaskValue);

enum class Algorithm {
	BoykovKolmogorov,
	GridCut,
	CudaCut
};

Algorithm getAlgorithmFromString(const std::string &token)
{
	if (token == "boykov-kolmogorov") {
		return Algorithm::BoykovKolmogorov;
	} else if (token == "gridcut") {
		return Algorithm::GridCut;
	} else if (token == "cudacut") {
		return Algorithm::CudaCut;
	} else {
		throw po::validation_error(po::validation_error::invalid_option_value);
	}
	return Algorithm::BoykovKolmogorov;
}

int
main(int argc, char* argv[])
{
	boost::filesystem::path inputFile;
	boost::filesystem::path markersFile;
	boost::filesystem::path outputFile;
	float sigma;
	uint8_t maskValue;

	Algorithm algorithm;
	std::string algorithmName;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<boost::filesystem::path>(&inputFile), "input file")
		("markers,m", po::value<boost::filesystem::path>(&markersFile), "markers file")
		("algorithm,a", po::value<std::string>(&algorithmName)->default_value("boykov-kolmogorov"), "boykov-kolmogorov, gridcut, cudacut")
		("output,o", po::value<boost::filesystem::path>(&outputFile), "output mask file")
		("sigma,s", po::value<float>(&sigma)->default_value(1.0f), "input noise deviation")
		("mask-value,v", po::value<uint8_t>(&maskValue)->default_value(255), "mask value")
		;
	/*std::string inputFile;
	std::string markersFile;
	std::string outputFile;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<std::string>(&inputFile), "input file")
		("markers,m", po::value<std::string>(&markersFile), "markers file")
		("output,o", po::value<std::string>(&outputFile), "output mask file")
		;*/

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	algorithm = getAlgorithmFromString(algorithmName);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 1;
	}

	if (vm.count("input") == 0) {
		std::cout << "Missing input filename\n" << desc << "\n";
		return 1;
	}

	if (vm.count("markers") == 0) {
		std::cout << "Missing markers filename\n" << desc << "\n";
		return 1;
	}

	if (vm.count("output") == 0) {
		std::cout << "Missing output filename\n" << desc << "\n";
		return 1;
	}

	typedef itk::ImageFileReader<ImageType>  ImageReaderType;
	typedef itk::ImageFileReader<MarkersType>  MarkersReaderType;
	typedef itk::ImageFileWriter<MaskType>  MaskWriterType;

	BOOST_LOG_TRIVIAL(info) << "Loading inputs ...";
	ImageReaderType::Pointer imageReader = ImageReaderType::New();
	MarkersReaderType::Pointer markersReader = MarkersReaderType::New();
	imageReader->SetFileName(inputFile.string());
	markersReader->SetFileName(markersFile.string());
	imageReader->Update();
	markersReader->Update();

	ImageType::Pointer image = imageReader->GetOutput();
	MarkersType::Pointer markers = markersReader->GetOutput();
	BOOST_LOG_TRIVIAL(info) << "Input size: ["
		<< image->GetLargestPossibleRegion().GetSize()[0] << ", "
		<< image->GetLargestPossibleRegion().GetSize()[1] << ", "
		<< image->GetLargestPossibleRegion().GetSize()[2] << "]";

	BOOST_LOG_TRIVIAL(info) << "Allocating output ...";
	MaskType::Pointer outputImage = MaskType::New();
	outputImage->SetRegions(image->GetLargestPossibleRegion());
	outputImage->Allocate();
	outputImage->SetSpacing(image->GetSpacing());

	switch (algorithm) {
	case Algorithm::BoykovKolmogorov:
		BOOST_LOG_TRIVIAL(info) << "Running Boykov-Kolmogorov graph cut ...";
		computeBoykovKolmogorovGrid(
			cugip::const_view(*(image.GetPointer())),
			cugip::const_view(*(markers.GetPointer())),
			cugip::view(*(outputImage.GetPointer())),
			sigma,
			maskValue);
		break;
	case Algorithm::GridCut:
		BOOST_LOG_TRIVIAL(info) << "Running GridCut ...";
		computeGridCut(
			cugip::const_view(*(image.GetPointer())),
			cugip::const_view(*(markers.GetPointer())),
			cugip::view(*(outputImage.GetPointer())),
			sigma,
			maskValue);
		break;
	case Algorithm::CudaCut:
		BOOST_LOG_TRIVIAL(info) << "Running CUDACut ...";
		computeCudaGraphCut(
			cugip::const_view(*(image.GetPointer())),
			cugip::const_view(*(markers.GetPointer())),
			cugip::view(*(outputImage.GetPointer())),
			sigma,
			maskValue);
		break;
	default:
		BOOST_LOG_TRIVIAL(error) << "Unknown algorithm";
	}

	BOOST_LOG_TRIVIAL(info) << "Saving output `" << outputFile << "` ...";
	MaskWriterType::Pointer writer = MaskWriterType::New();
	writer->SetFileName(outputFile.string());
	writer->SetInput(outputImage);
	try {
		writer->Update();
	} catch (itk::ExceptionObject & error) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

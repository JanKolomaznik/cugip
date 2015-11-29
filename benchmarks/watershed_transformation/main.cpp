#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMorphologicalWatershedImageFilter.h>
#include <itkWatershedImageFilter.h>

#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/filesystem.hpp>
#include <cugip/itk_utils.hpp>

#include <boost/log/trivial.hpp>
#include <boost/timer/timer.hpp>

#include "watershed_transformation.hpp"

typedef itk::Image<float, 3> ImageType;
typedef itk::Image<int, 3> LabeledImageType;
typedef itk::MorphologicalWatershedImageFilter<ImageType, LabeledImageType> WatershedFilterType;
typedef itk::ImageFileWriter<LabeledImageType>  WriterType;

namespace po = boost::program_options;

enum class Algorithm {
	ItkImplementation,
	TopoDistance
};

Algorithm getAlgorithmFromString(const std::string &token)
{
	if (token == "itk-implementation") {
		return Algorithm::ItkImplementation;
	} else if (token == "topo-distance") {
		return Algorithm::TopoDistance;
	} else {
		throw po::validation_error(po::validation_error::invalid_option_value);
	}
	return Algorithm::ItkImplementation;
}

int
main(int argc, char* argv[])
{
	boost::filesystem::path inputFile;
	boost::filesystem::path outputFile;

	Algorithm algorithm;
	std::string algorithmName;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<boost::filesystem::path>(&inputFile), "input file")
		("algorithm,a", po::value<std::string>(&algorithmName)->default_value("itk-implementation"), "itk-implementation")
		("output,o", po::value<boost::filesystem::path>(&outputFile), "output mask file")
		;

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

	if (vm.count("output") == 0) {
		std::cout << "Missing output filename\n" << desc << "\n";
		return 1;
	}

	typedef itk::ImageFileReader<ImageType>  ImageReaderType;

	BOOST_LOG_TRIVIAL(info) << "Loading inputs ...";
	ImageReaderType::Pointer imageReader = ImageReaderType::New();
	imageReader->SetFileName(inputFile.string());
	imageReader->Update();

	ImageType::Pointer image = imageReader->GetOutput();

	LabeledImageType::Pointer outputImage;

	switch (algorithm) {
	case Algorithm::ItkImplementation: {
		BOOST_LOG_TRIVIAL(info) << "Running ITK version of watershed transformation ...";
			WatershedFilterType::Pointer filter = WatershedFilterType::New();
			filter->SetInput(image);
			filter->MarkWatershedLineOff();

			boost::timer::cpu_timer computationTimer;
			computationTimer.start();
			filter->Update();
			BOOST_LOG_TRIVIAL(info) << "Computation time: " << computationTimer.format(9, "%w");

			outputImage = filter->GetOutput();
		}
		break;
	default:
		BOOST_LOG_TRIVIAL(info) << "Allocating output ...";
		outputImage = LabeledImageType::New();
		outputImage->SetRegions(image->GetLargestPossibleRegion());
		outputImage->Allocate();
		outputImage->SetSpacing(image->GetSpacing());
		BOOST_LOG_TRIVIAL(info) << "Running CUDA version of watershed transformation (topographical distance)...";
		boost::timer::cpu_timer computationTimer;
		computationTimer.start();
		watershedTransformation(
			cugip::const_view(*(image.GetPointer())),
			cugip::view(*(outputImage.GetPointer())),
			Options()
			);
		BOOST_LOG_TRIVIAL(info) << "Computation time: " << computationTimer.format(9, "%w");
		//BOOST_LOG_TRIVIAL(error) << "Unknown algorithm";
	}

	BOOST_LOG_TRIVIAL(info) << "Saving output `" << outputFile << "` ...";
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(outputFile.string());
	writer->SetInput(outputImage);
	try {
		writer->Update();
	} catch (itk::ExceptionObject & error) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}

	return 0;
}

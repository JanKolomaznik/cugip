
#include <cugip/image_traits.hpp>
#include <boost/program_options.hpp>
#include <boost/variant.hpp>
#include <boost/filesystem.hpp>

#include <cugip/host_image.hpp>
#include <cugip/host_image_view.hpp>

#include <cugip/image_dumping.hpp>
#include <cugip/multiresolution_pyramid.hpp>


#include <cugip/3rd_party/free_image_utils.hpp>
#include <cugip/any_image.hpp>
#include <cugip/transform.hpp>
#include <cugip/color_spaces.hpp>
#include <cugip/geometry_transformation.hpp>
#include <cugip/interpolated_view.hpp>
#include <cugip/advanced_operations/fast_corner_detector.hpp>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace cugip;

void runConnectedComponentLabeling(
		const_host_image_view<const int8_t, 2> aInput,
		host_image_view<int32_t, 2> aOutput,
		const std::string &aName);

void runConnectedComponentLabeling(
		const_host_image_view<const int8_t, 3> aInput,
		host_image_view<int32_t, 3> aOutput,
		const std::string &aName);


/*template<int tDimension>
void processImage(fs::path aInput, fs::path aOutput, std::string aMethodName)
{
	typedef itk::Image<int8_t, tDimension> InputImageType;
	typedef itk::Image<int32_t, tDimension> OutputImageType;

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

	runConnectedComponentLabeling(inView, outView, aMethodName);

	typename WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(aOutput.string());
	writer->SetInput(output_image);
	writer->Update();
}*/



//struct ProcessImage : public boost::static_visitor<>

multiresolution_levels<2>
generate_level_infos(int aLevelCount, vect2i_t aSize)
{
	CUGIP_ASSERT(aLevelCount >= 2);

	multiresolution_levels<2> levels(aLevelCount);
	levels[0].scale = scale_type(1);
	levels[1].scale = scale_type(2,3);
	for (int i = 2; i < aLevelCount; ++i) {
		levels[i].scale = levels[i-2].scale * scale_type(1, 2);
	}

	levels[0].resolution = aSize;
	for (int i = 1; i < aLevelCount; ++i) {
		levels[i].resolution = round(vect2f_t(
			boost::rational_cast<float>(aSize[0] * levels[i].scale),
			boost::rational_cast<float>(aSize[1] * levels[i].scale)
			));
	}

	return levels;
}

template<typename TPyramidView>
void
generate_multiresolution_octaves_and_intra_octaves(TPyramidView &aPyramid)
{
	//TODO - can be executed in two streams, factors should be 1/1.5 and 1/2.0, speedup for factor 1/2
	float factor = boost::rational_cast<float>(aPyramid.level_info(1).scale / aPyramid.level_info(0).scale);
	scale(
		make_interpolated_view(aPyramid.level(0)),
		aPyramid.level(1),
		factor);

	for (int i = 2; i < aPyramid.count(); ++i) {
		float factor = boost::rational_cast<float>(aPyramid.level_info(i).scale / aPyramid.level_info(i-2).scale);
		scale(
			make_interpolated_view(aPyramid.level(i - 2)),
			aPyramid.level(i),
			factor);
	}
}


struct ProcessImage : public boost::static_visitor<>
{
	ProcessImage(const fs::path &aOutput)
		: outputFile(aOutput)
	{}

	template<typename TImageView>
	void operator()(TImageView aInput) const {


		host_image<float, 2> intensity(aInput.dimensions());

		transform(
			aInput,
			view(intensity),
			[](auto color) {
				return 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
			});

		using Image = host_image<float, 2>; //typename image_view_traits<TImageView>::image_t;
		multiresolution_levels<2> scalingConfig = generate_level_infos(6, intensity.dimensions()); //TODO - set level count
		multiresolution_pyramid<Image> pyramid(std::move(intensity), scalingConfig);
		multiresolution_pyramid<Image> processedPyramid(scalingConfig);

		auto pyramidView = view(pyramid);
		generate_multiresolution_octaves_and_intra_octaves(pyramidView);
		dump_pyramid(pyramidView, "./testPyramid");

		auto pyramidView2 = view(processedPyramid);
		apply_to_all_levels(
			pyramidView,
			pyramidView2,
			[](auto from, auto to)
			{
				fast_corner_saliency(from, to, 0);
			});
		dump_pyramid(pyramidView2, "./saliencePyramid");

			/*host_image<float, 2> salience(aInput.dimensions());
			fast_corner_saliency(const_view(intensity), view(salience), 0);

			host_image<float, 2> maxima(aInput.dimensions());
			transform_locator(
				const_view(salience),
				view(maxima),
				[](auto aLocator) {
					for (int i = -1; i <= 1; ++i) {
						for (int j = -1; j <= 1; ++j) {
							if (aLocator[vect2i_t(i, j)] > aLocator.get()) {
								return 0.0f;
							}
						}
					}
					return aLocator.get();
				});

			dump_view(view(maxima), "salience_");
		}*/

	//	compute_multiresolution_pyramid(aInput, pyramidView, tmpView, Postprocess{});
	}



	fs::path outputFile;
};

int main( int argc, char* argv[] )
{
	fs::path inputFile;
	fs::path outputFile;

	std::string methodName;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<fs::path>(&inputFile), "input file")
		("output,o", po::value<fs::path>(&outputFile), "output file")
		//("type,t", po::value<std::string>(&methodName)->default_value("async_gs"), "async_gs, sync_gs, sync_gs2, sync_gs4, sync_gs8")
		//("normalize,n", po::value<bool>(&normalize)->default_value(false), "normalize")
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

	if (vm.count("output") == 0) {
		std::cout << "Missing output filename\n" << desc << "\n";
		return 1;
	}


	auto loadedImage = readImageData(inputFile);

	auto loadedImageView = cugip::const_view(loadedImage);
	boost::apply_visitor(ProcessImage(outputFile), loadedImageView);
/*
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

		switch (numDimensions) {
		case 2:
			processImage<2>(inputFile, outputFile, methodName);
			break;
		case 3:
			processImage<3>(inputFile, outputFile, methodName);
			break;
		default:
			std::cerr << "Error: Unsupported image dimension.\n";
			return EXIT_FAILURE;
		}

	} catch( itk::ExceptionObject & error ) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}*/

	return 0;
}

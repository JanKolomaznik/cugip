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

using namespace cugip;

namespace x3 = boost::spirit::x3;
namespace ascii = boost::spirit::x3::ascii;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

struct Ellipsoid
{
	Int3 center;
	float radius;
	float value;
};

struct Tube
{
	Int3 center;
	Float3 direction;
	float radius;
	float length;
	float value;
};

struct Plate
{
	Int3 center;
	Float3 normal;
	float radius;
	float thickness;
	float value;
};

template<typename TCallable>
void for_each_in_radius(TCallable aCallable, int aRadius)
{
	simple_vector<int, 3> index;
	for(index[2] = -aRadius; index[2] <= aRadius; ++index[2]) {
		for(index[1] = -aRadius; index[1] <= aRadius; ++index[1]) {
			for(index[0] = -aRadius; index[0] <= aRadius; ++index[0]) {
				aCallable(index);
			}
		}
	}
}
template<typename TView>
void putObject(const Ellipsoid &aEllipsoid, TView aView)
{
	auto loc = image_locator<TView, BorderHandlingTraits<border_handling_enum::REPEAT>>(aView, aEllipsoid.center);
	for_each_in_radius(
		[&](Int3 coord) {
			auto res = sqr(coord[0]) + sqr(coord[1]) + sqr(coord[2]);
			if (res <= sqr(aEllipsoid.radius)) {
				loc[coord] = static_cast<typename TView::value_type>(aEllipsoid.value);
			}
		},
		int(aEllipsoid.radius) + 1);
}

template<typename TView>
void putObject(const Tube &aTube, TView aView)
{
	auto loc = image_locator<TView, BorderHandlingTraits<border_handling_enum::REPEAT>>(aView, aTube.center);
	for_each_in_radius(
		[&](Int3 coord) {
			Float3 pos(coord);
			float proj = dot(pos, aTube.direction);
			auto norm = pos - proj * aTube.direction;
			auto dist = magnitude(norm);
			if (dist <= aTube.radius) {
				loc[coord] = static_cast<typename TView::value_type>(aTube.value);
			}
		},
		int(aTube.length / 2) + 1);
}

template<typename TView>
void putObject(const Plate &aPlate, TView aView)
{
	auto loc = image_locator<TView, BorderHandlingTraits<border_handling_enum::REPEAT>>(aView, aPlate.center);
	for_each_in_radius(
		[&](Int3 coord) {
			Float3 pos(coord);
			float proj = abs(dot(pos, aPlate.normal));
			auto dist = proj/magnitude(aPlate.normal);
			if (dist <= aPlate.thickness / 2.0f) {
				loc[coord] = static_cast<typename TView::value_type>(aPlate.value);
			}
		},
		int(aPlate.radius));
}

template<typename TObject, typename TView>
void generateObjects(const std::vector<TObject> &aObjects, TView aView)
{
	for (const auto &obj : aObjects) {
		putObject(obj, aView);
	}
}

template<typename TView>
void fillBackground(TView aView, int aBackgroundValue)
{
	cugip::for_each(aView, [=](typename TView::value_type &aValue){ return aValue = aBackgroundValue; });
}

template<typename TImageType>
void generateData(
	fs::path aOutputPath,
	typename TImageType::SizeType size,
	const std::vector<Ellipsoid> &aEllipsoids,
	const std::vector<Tube> &aTubes,
	const std::vector<Plate> &aPlates,
	int aBackgroundValue
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

	fillBackground(outView, aBackgroundValue);

	generateObjects(aEllipsoids, outView);
	generateObjects(aTubes, outView);
	generateObjects(aPlates, outView);

	typedef itk::ImageFileWriter<TImageType> WriterType;
	typename WriterType::Pointer writer = WriterType::New();

	std::cout << "Writing file " << aOutputPath.string() << '\n';
	writer->SetFileName(aOutputPath.string());
	writer->SetInput(image);
	writer->Update();
}

template<typename TImageType>
void generateData(
	fs::path aOutputPath,
	typename TImageType::SizeType size
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

	auto center = div(tmpSize, 2);
	cugip::for_each_position(
		outView,
		[=](auto &aValue, auto aPosition){
			return aValue = magnitude(aPosition - center);
		});

	typedef itk::ImageFileWriter<TImageType> WriterType;
	typename WriterType::Pointer writer = WriterType::New();

	std::cout << "Writing file " << aOutputPath.string() << '\n';
	writer->SetFileName(aOutputPath.string());
	writer->SetInput(image);
	writer->Update();
}

std::vector<Ellipsoid> ellipsoids = {
	Ellipsoid{ Int3(300, 80, 100), 20.0f, 100.0f },
	Ellipsoid{ Int3(300, 180, 100), 10.0f, 100.0f },

	Ellipsoid{ Int3(320, 260, 80), 5.0f, 100.0f },
	Ellipsoid{ Int3(280, 260, 80), 5.0f, 100.0f },
	Ellipsoid{ Int3(320, 260, 120), 5.0f, 100.0f },
	Ellipsoid{ Int3(280, 260, 120), 5.0f, 100.0f },

	Ellipsoid{ Int3(320, 310, 80), 2.0f, 100.0f },
	Ellipsoid{ Int3(280, 310, 80), 2.0f, 100.0f },
	Ellipsoid{ Int3(320, 310, 120), 2.0f, 100.0f },
	Ellipsoid{ Int3(280, 310, 120), 2.0f, 100.0f },

	Ellipsoid{ Int3(320, 350, 80), 2.0f, 100.0f },
	Ellipsoid{ Int3(280, 350, 80), 2.0f, 100.0f },
	Ellipsoid{ Int3(320, 350, 120), 2.0f, 100.0f },
	Ellipsoid{ Int3(280, 350, 120), 2.0f, 100.0f },
};

std::vector<Tube> tubes = {
	Tube{
		Int3(60, 250, 100),
		Float3(0.0f, 0.0f, 1.0f),
		15.0f,
		80.0f,
		100.0f },
	Tube{
		Int3(140, 250, 100),
		Float3(0.0f, 0.0f, 1.0f),
		15.0f,
		80.0f,
		100.0f },

	Tube{
		Int3(60, 320, 100),
		Float3(0.0f, 0.0f, 1.0f),
		5.0f,
		80.0f,
		100.0f },
	Tube{
		Int3(140, 320, 100),
		Float3(0.0f, 0.0f, 1.0f),
		5.0f,
		80.0f,
		100.0f },

	Tube{
		Int3(60, 370, 100),
		Float3(0.0f, 0.0f, 1.0f),
		2.0f,
		80.0f,
		100.0f },
	Tube{
		Int3(140, 370, 100),
		Float3(0.0f, 0.0f, 1.0f),
		2.0f,
		80.0f,
		100.0f },
};

std::vector<Plate> plates = {
	Plate{
		Int3(100, 40, 100),
		Float3(0.0f, 1.0f, 0.0f),
		50.0f,
		30.0f,
		100.0f },
	Plate{
		Int3(100, 120, 100),
		Float3(0.0f, 1.0f, 0.0f),
		50.0f,
		8.0f,
		100.0f },
	Plate{
		Int3(100, 160, 100),
		Float3(0.0f, 1.0f, 0.0f),
		40.0f,
		1.0f,
		100.0f },
};

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

void initObjects() {
	int count = plates.size();
	for (int i = 0; i < count; ++i) {
		plates.push_back(plates[i]);
		plates[count + i].center[2] += 200;
		plates[count + i].value = 90;
	}
	count = tubes.size();
	for (int i = 0; i < count; ++i) {
		tubes.push_back(tubes[i]);
		tubes[count + i].center[2] += 200;
		tubes[count + i].value = 90;
	}
	count = ellipsoids.size();
	for (int i = 0; i < count; ++i) {
		ellipsoids.push_back(ellipsoids[i]);
		ellipsoids[count + i].center[2] += 200;
		ellipsoids[count + i].value = 90;
	}
}

Tube parseTube(const std::string &aLine)
{
	using x3::int_;
	using x3::lit;
	using x3::char_;
	using x3::float_;
	using ascii::blank;

	std::tuple<int, int, int, float, float, float, float, float, float> result;
	auto rule = lit('t') >>
		lit('[') >> int_ >> ',' >> int_ >> ',' >> int_ >> lit(']') >>
		lit('[') >> float_ >> ',' >> float_ >> ',' >> float_ >> lit(']') >>
		float_ >>
		float_ >>
		float_;
	bool const res = x3::phrase_parse(aLine.begin(), aLine.end(), rule, blank, result);
	return Tube{
		Int3(get<0>(result), get<1>(result), get<2>(result)),
		Float3(get<3>(result), get<4>(result), get<5>(result)),
		get<6>(result),
		get<7>(result),
		get<8>(result)};
}

Plate parsePlate(const std::string &aLine)
{
	using x3::int_;
	using x3::lit;
	using x3::char_;
	using x3::float_;
	using ascii::blank;

	std::tuple<int, int, int, float, float, float, float, float, float> result;
	auto rule = lit('p') >>
		lit('[') >> int_ >> lit(',') >> int_ >> lit(',') >> int_ >> lit(']') >>
		lit('[') >> float_ >> lit(',') >> float_ >> lit(',') >> float_ >> lit(']') >>
		float_ >>
		float_ >>
		float_;
	bool const res = x3::phrase_parse(aLine.begin(), aLine.end(), rule, blank, result);
	return Plate{
		Int3(get<0>(result), get<1>(result), get<2>(result)),
		Float3(get<3>(result), get<4>(result), get<5>(result)),
		get<6>(result),
		get<7>(result),
		get<8>(result)};
}

Ellipsoid parseEllipsoid(const std::string &aLine)
{
	using x3::int_;
	using x3::lit;
	using x3::char_;
	using x3::float_;
	using ascii::blank;

	std::tuple<int, int, int, float, float> result;
	auto rule = lit('e') >>
		lit('[') >> int_ >> lit(',') >> int_ >> lit(',') >> int_ >> lit(']') >>
		float_ >>
		float_;
	bool const res = x3::phrase_parse(aLine.begin(), aLine.end(), rule, blank, result);
	if (!res) {
		std::cout << "XXX " << aLine << "\n";
	}
	return Ellipsoid{
		Int3(get<0>(result), get<1>(result), get<2>(result)),
		get<3>(result),
		get<4>(result)};
}

void loadConfiguration(
	const std::string &aInputPath,
	std::vector<Ellipsoid> &aEllipsoids,
	std::vector<Tube> &aTubes,
	std::vector<Plate> &aPlates)
{
	std::ifstream file(aInputPath, std::ifstream::in);
	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		switch (line[0]) {
		case 't':
			aTubes.push_back(parseTube(line));
			break;
		case 'e':
			aEllipsoids.push_back(parseEllipsoid(line));
			break;
		case 'p':
			aPlates.push_back(parsePlate(line));
			break;
		default:
			std::cout << "Unknown line identifier '" << line[0] << "'\n";
		}
	}
}


int main( int argc, char* argv[] )
{
	fs::path input_file;
	fs::path output_file;
	double sigma;

	std::string resolution;

	int background;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("input,i", po::value<fs::path>(&input_file), "configuration file")
		("output,o", po::value<fs::path>(&output_file), "output file")
		("resolution,r", po::value<std::string>(&resolution), "Resolution WIDTHxHEIGHTxDEPTH")
		("background,b", po::value<int>(&background)->default_value(0), "background value")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);


	if (vm.count("help")) {
	    std::cout << desc << "\n";
	    return 1;
	}
	/*if (vm.count("input") == 0) {
	    std::cout << "Missing input filename\n" << desc << "\n";
	    return 1;
    }*/

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
		std::cout << "Loading configuration ...\n";
		//initObjects();
		loadConfiguration(input_file.string(), ellipsoids, tubes, plates);

		const unsigned int Dimension = 3;

		//typedef uint8_t                              PixelType;
		typedef float                              PixelType;
		typedef itk::Image< PixelType, Dimension > ImageType;


		std::cout << boost::format("Generating image of size [%1%, %2%, %3%] ...\n") % std::get<0>(size) % std::get<1>(size) % std::get<2>(size);
		ImageType::SizeType imageSize;
		imageSize[0] = std::get<0>(size);
		imageSize[1] = std::get<1>(size);
		imageSize[2] = std::get<2>(size);
		generateData<ImageType>(output_file, imageSize, ellipsoids, tubes, plates, background);
		//generateData<ImageType>(output_file, imageSize);
		//generateEllipsoids<ImageType>(output_file, imageSize, sigma);
	} catch( itk::ExceptionObject & error ) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

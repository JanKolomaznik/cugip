#define BOOST_TEST_MODULE TransformTest
//#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <cugip/reduce.hpp>
#include <cugip/math.hpp>
#include <cugip/image.hpp>
#include <cugip/host_image.hpp>
#include <cugip/copy.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/subview.hpp>
#include <cugip/image_dumping.hpp>
/*#include <cugip/view_arithmetics.hpp>*/
#include <cugip/transform.hpp>
/*#include <thrust/device_vector.h>
#include <thrust/reduce.h>*/

using namespace cugip;

//static const float cEpsilon = 0.00001;

#include <QApplication>
#include <QMainWindow>
#include <QImage>
#include <QtWidgets>

class QAction;
class QLabel;
class QMenu;
class QScrollArea;
class QScrollBar;

template<typename TView>
QImage
fillQImage(TView aView)
{
	static_assert(dimension<TView>::value == 2, "Only 2d images can be converted to QImage");

	auto qimage = QImage(aView.dimensions()[0], aView.dimensions()[1], QImage::Format_Grayscale8);
	auto qview = makeHostImageView(qimage.bits(), aView.dimensions(), {1, qimage.bytesPerLine()});

	copy(aView, qview);

	return qimage;
}


class ImageViewer : public QMainWindow
{
//Q_OBJECT

public:
	ImageViewer();
	//bool loadFile(const QString &);

	template <typename TView>
	void showView(TView aView)
	{
		setImage(fillQImage(aView));
	}

private slots:
	//void open();
	//void saveAs();
	//void print();
	//void copy();
	//void paste();
	void zoomIn();
	void zoomOut();
	void normalSize();
	void fitToWindow();
	void about();

private:
	void createActions();
	void createMenus();
	void updateActions();
	//bool saveFile(const QString &fileName);
	void setImage(QImage &&newImage);
	void scaleImage(double factor);
	void adjustScrollBar(QScrollBar *scrollBar, double factor);

	QImage image;
	QLabel *imageLabel;
	QScrollArea *scrollArea;
	double scaleFactor;

	QAction *zoomInAct;
	QAction *zoomOutAct;
	QAction *normalSizeAct;
	QAction *fitToWindowAct;
};



ImageViewer::ImageViewer()
	: imageLabel(new QLabel)
	, scrollArea(new QScrollArea)
	, scaleFactor(1)
{
	imageLabel->setBackgroundRole(QPalette::Base);
	imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	imageLabel->setScaledContents(true);

	scrollArea->setBackgroundRole(QPalette::Dark);
	scrollArea->setWidget(imageLabel);
	scrollArea->setVisible(false);
	setCentralWidget(scrollArea);

	createActions();

	resize(QGuiApplication::primaryScreen()->availableSize() * 3 / 5);
}


void ImageViewer::setImage(QImage &&newImage)
{
	image = std::move(newImage);
	imageLabel->setPixmap(QPixmap::fromImage(image));
	scaleFactor = 1.0;

	scrollArea->setVisible(true);
	fitToWindowAct->setEnabled(true);
	updateActions();

	if (!fitToWindowAct->isChecked())
		imageLabel->adjustSize();
}

void ImageViewer::zoomIn()
{
	scaleImage(1.25);
}

void ImageViewer::zoomOut()
{
	scaleImage(0.8);
}

void ImageViewer::normalSize()
{
	imageLabel->adjustSize();
	scaleFactor = 1.0;
}

void ImageViewer::fitToWindow()
{
	bool fitToWindow = fitToWindowAct->isChecked();
	scrollArea->setWidgetResizable(fitToWindow);
	if (!fitToWindow)
		normalSize();
	updateActions();
}

void ImageViewer::about()
{
	QMessageBox::about(this, tr("About Image Viewer"),
			tr("<p>The <b>Image Viewer</b> example shows how to combine QLabel "
				"and QScrollArea to display an image. QLabel is typically used "
				"for displaying a text, but it can also display an image. "
				"QScrollArea provides a scrolling view around another widget. "
				"If the child widget exceeds the size of the frame, QScrollArea "
				"automatically provides scroll bars. </p><p>The example "
				"demonstrates how QLabel's ability to scale its contents "
				"(QLabel::scaledContents), and QScrollArea's ability to "
				"automatically resize its contents "
				"(QScrollArea::widgetResizable), can be used to implement "
				"zooming and scaling features. </p><p>In addition the example "
				"shows how to use QPainter to print an image.</p>"));
}

void ImageViewer::createActions()
{
	QMenu *viewMenu = menuBar()->addMenu(tr("&View"));

	zoomInAct = viewMenu->addAction(tr("Zoom &In (25%)"), this, &ImageViewer::zoomIn);
	zoomInAct->setShortcut(QKeySequence::ZoomIn);
	zoomInAct->setEnabled(false);

	zoomOutAct = viewMenu->addAction(tr("Zoom &Out (25%)"), this, &ImageViewer::zoomOut);
	zoomOutAct->setShortcut(QKeySequence::ZoomOut);
	zoomOutAct->setEnabled(false);

	normalSizeAct = viewMenu->addAction(tr("&Normal Size"), this, &ImageViewer::normalSize);
	normalSizeAct->setShortcut(tr("Ctrl+S"));
	normalSizeAct->setEnabled(false);

	viewMenu->addSeparator();

	fitToWindowAct = viewMenu->addAction(tr("&Fit to Window"), this, &ImageViewer::fitToWindow);
	fitToWindowAct->setEnabled(false);
	fitToWindowAct->setCheckable(true);
	fitToWindowAct->setShortcut(tr("Ctrl+F"));

	QMenu *helpMenu = menuBar()->addMenu(tr("&Help"));

	helpMenu->addAction(tr("&About"), this, &ImageViewer::about);
	helpMenu->addAction(tr("About &Qt"), &QApplication::aboutQt);
}

void ImageViewer::updateActions()
{
	zoomInAct->setEnabled(!fitToWindowAct->isChecked());
	zoomOutAct->setEnabled(!fitToWindowAct->isChecked());
	normalSizeAct->setEnabled(!fitToWindowAct->isChecked());
}

void ImageViewer::scaleImage(double factor)
{
	Q_ASSERT(imageLabel->pixmap());
	scaleFactor *= factor;
	imageLabel->resize(scaleFactor * imageLabel->pixmap()->size());

	adjustScrollBar(scrollArea->horizontalScrollBar(), factor);
	adjustScrollBar(scrollArea->verticalScrollBar(), factor);

	zoomInAct->setEnabled(scaleFactor < 3.0);
	zoomOutAct->setEnabled(scaleFactor > 0.333);
}

void ImageViewer::adjustScrollBar(QScrollBar *scrollBar, double factor)
{
	scrollBar->setValue(int(factor * scrollBar->value()
				+ ((factor - 1) * scrollBar->pageStep()/2)));
}


static int argc = 1;
static char *argv[] = { "debug", NULL };
class DebugViewer {
public:
	DebugViewer()
		: app(argc, argv)
	{
	}
	template <typename ...TViews>
	DebugViewer(TViews... aViews)
		: app(argc, argv)
	{
		std::initializer_list<int>{ (addViewer(aViews), 0) ... };
	}


	template <typename TView>
	void addViewer(TView aView)
	{
		viewers.emplace_back(new ImageViewer());
		viewers.back()->showView(aView);
	}

	int exec() {
		for (auto & viewer : viewers) {
			viewer->show();
		}
		return app.exec();
	}

	QApplication app;
	std::vector<std::unique_ptr<ImageViewer>> viewers;
};
//***************************************************************************
struct TestTransformFunctor
{
	template<typename TLocator>
	CUGIP_DECL_HYBRID int //typename TLocator::value_type
	operator()(TLocator aLocator) const
	{
		using Diff = typename TLocator::diff_t;
		//return aLocator[Diff(-1, FillFlag())] + aLocator[Diff(1, FillFlag())];
		return aLocator[Diff(0, -1, 0)];

	}
};
/*
BOOST_AUTO_TEST_CASE(TransformDevice)
{

	device_image<int, 3> deviceImage1(300, 300, 4);
	device_image<int, 3> deviceImage2(300, 300, 4);

	auto input = constantImage(25, deviceImage1.dimensions());

	copy(input, view(deviceImage1));

	transform(const_view(deviceImage1), view(deviceImage2), []__device__(const int &value) { return value + 1; });

	auto difference = sum_differences(const_view(deviceImage2), constantImage(26, deviceImage1.dimensions()), 0);
	BOOST_CHECK_EQUAL(difference, 0);
}

BOOST_AUTO_TEST_CASE(TransformDeviceWithSharedMemory)
{
	auto default_policy = DefaultTransformLocatorPolicy<3>{};
	auto preloading_policy = PreloadingTransformLocatorPolicy<3, 1>{};

	device_image<int, 3> deviceImage1(300, 300, 4);
	device_image<int, 3> deviceImage2(300, 300, 4);

	auto input = constantImage(25, deviceImage1.dimensions());

	copy(input, view(deviceImage1));

	auto ftor = []__device__(auto locator) { return locator[vect3i_t(1, 1, 1)] + locator[vect3i_t(-1, -1, -1)]; };
	transform_locator(input, view(deviceImage1), ftor, default_policy);
	transform_locator(input, view(deviceImage2), ftor, preloading_policy);

	auto difference = sum_differences(const_view(deviceImage1), const_view(deviceImage2), 0);
	BOOST_CHECK_EQUAL(difference, 0);
}*/

/*
BOOST_AUTO_TEST_CASE(BoundedTransformWithPreload)
{

	device_image<int8_t, 3> deviceImage1(30, 30, 4);
	device_image<int8_t, 3> deviceImage2(30, 30, 4);

	auto input = UniqueIdDeviceImageView<3>(deviceImage1.dimensions());

	transform_locator(input, view(deviceImage1), TestTransformFunctor());
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	transform_locator(input, view(deviceImage2), TestTransformFunctor(), PreloadingTransformLocatorPolicy<3, 1>());
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());

	auto difference = sum_differences(view(deviceImage1), view(deviceImage2), int{0});

	//DebugViewer().exec();
	BOOST_CHECK_EQUAL(difference, 0);
}*/

BOOST_AUTO_TEST_CASE(BoundedTransformWithPreload2)
{

	device_image<int, 3> deviceImage1(30, 30, 4);
	device_image<int, 3> deviceImage2(deviceImage1.dimensions());
	host_image<int, 3> hostImage1(deviceImage1.dimensions());
	host_image<int, 3> hostImage2(deviceImage1.dimensions());

	auto input = UniqueIdDeviceImageView<3>(deviceImage1.dimensions());
	//auto input = constantImage(100, deviceImage1.dimensions());

	transform_locator(input, view(deviceImage1), TestTransformFunctor());
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	transform_locator(input, view(deviceImage2), TestTransformFunctor(), PreloadingTransformLocatorPolicy<3, 1>());
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());

	copy(view(deviceImage1), view(hostImage1));
	copy(view(deviceImage2), view(hostImage2));
	auto difference = sum_differences(view(deviceImage1), view(deviceImage2), int{0});

	//DebugViewer(view(hostImage1), view(hostImage2)).exec();
	//dump_view(view(hostImage1), "./normal");
	//dump_view(view(hostImage2), "./preloaded");
	BOOST_CHECK_EQUAL(difference, 0);
}

struct TestTransformFunctor2
{
	// template<typename ...TArgs>
	// CUGIP_DECL_HYBRID void
	// operator()(TArgs... aArgs) const
	// {
	// }

	template<typename TPreloadedView, typename TInView, typename TOutView>
	CUGIP_DECL_DEVICE void
	operator()(vect3i_t tileCorner, vect3i_t haloTileCorner, TPreloadedView dataView, TInView aInView, TOutView aOutView) const
	{
		auto threadIndex = currentThreadIndexDim<3>();

		auto idx = tileCorner + threadIndex;

		aOutView[idx] = aInView[idx];
		idx[0] += 16;
		aOutView[idx] = 2 * aInView[idx];
	}
};

template <typename TTileSize, typename THaloTileSize>
struct TestPolicy
{
	static constexpr int cDimension = TTileSize::cDimension;
	using TileSize = TTileSize;
	using HaloTileSize = THaloTileSize;
#if defined(__CUDACC__)
	CUGIP_DECL_HYBRID dim3 blockSize() const
	{
		return dim3{ 16, 4, 4}; //detail::defaultBlockDimForDimension<tDimension>();
	}

	template<int tDimension>
	CUGIP_DECL_HYBRID dim3 gridSize(const region<tDimension> aRegion) const
	{
		//TODO
		return {1,1,1};//detail::defaultGridSizeForBlockDim(aRegion.size, tileSize());
	}

	/*CUGIP_DECL_HYBRID region<cDimension> regionForBlock() const
	{
		return region<cDimension>{
			simple_vector<int, cDimension>(-tRadius, FillFlag()),
			dim3_to_vector<cDimension>(this->blockSize()) + simple_vector<int, cDimension>(2*tRadius, FillFlag())
			};
	}*/

	CUGIP_DECL_HYBRID
	vect3i_t corner1()
	{
       		return vect3i_t(-1, 0, 0);
	}
#endif //defined(__CUDACC__)

};

BOOST_AUTO_TEST_CASE(TileProcessing)
{
	device_image<int, 3> deviceImage1(300, 300, 40);
	device_image<int, 3> deviceImage2(deviceImage1.dimensions());
	host_image<int, 3> hostImage1(deviceImage1.dimensions());
	host_image<int, 3> hostImage2(deviceImage1.dimensions());

	//auto input = UniqueIdDeviceImageView<3>(deviceImage1.dimensions());
	auto input = constantImage(1, deviceImage1.dimensions());
	//auto input = checkerBoard(1, 2, vect3i_t(300, 32, 4), deviceImage1.dimensions());

	using Policy = TestPolicy<
				StaticSize<32, 4, 4>,
				StaticSize<34, 4, 4>>;

	cugip::detail::TileCoverImplementation::run
	//	<TestTransformFunctor2, Policy, decltype(input), decltype(view(deviceImage1))>
		(TestTransformFunctor2(), Policy(), 0, input, view(deviceImage1));

	auto board = checkerBoard(2, 1, vect3i_t(16, 300, 40), deviceImage1.dimensions());
	auto difference = sum_differences(view(deviceImage1), board, int{0});
	BOOST_CHECK_EQUAL(difference, 0);

	// copy(view(deviceImage1), view(hostImage1));
	// copy(board, view(deviceImage2));
	// copy(view(deviceImage2), view(hostImage2));
	// dump_view(view(hostImage1), "./aaaaa");
	// dump_view(view(hostImage2), "./bbbbb");
}

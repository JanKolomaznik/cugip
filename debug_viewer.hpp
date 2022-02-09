#pragma once

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


class DebugViewer {
public:

	static int argc = 1;
	static char *argv[] = { "debug", NULL };

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


#include "debug_viewer.tcc"

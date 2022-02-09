#pragma once


inline ImageViewer::ImageViewer()
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


inline void ImageViewer::setImage(QImage &&newImage)
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

inline void ImageViewer::zoomIn()
{
	scaleImage(1.25);
}

inline void ImageViewer::zoomOut()
{
	scaleImage(0.8);
}

inline void ImageViewer::normalSize()
{
	imageLabel->adjustSize();
	scaleFactor = 1.0;
}

inline void ImageViewer::fitToWindow()
{
	bool fitToWindow = fitToWindowAct->isChecked();
	scrollArea->setWidgetResizable(fitToWindow);
	if (!fitToWindow)
		normalSize();
	updateActions();
}

inline void ImageViewer::about()
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

inline void ImageViewer::createActions()
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

inline void ImageViewer::updateActions()
{
	zoomInAct->setEnabled(!fitToWindowAct->isChecked());
	zoomOutAct->setEnabled(!fitToWindowAct->isChecked());
	normalSizeAct->setEnabled(!fitToWindowAct->isChecked());
}

inline void ImageViewer::scaleImage(double factor)
{
	Q_ASSERT(imageLabel->pixmap());
	scaleFactor *= factor;
	imageLabel->resize(scaleFactor * imageLabel->pixmap()->size());

	adjustScrollBar(scrollArea->horizontalScrollBar(), factor);
	adjustScrollBar(scrollArea->verticalScrollBar(), factor);

	zoomInAct->setEnabled(scaleFactor < 3.0);
	zoomOutAct->setEnabled(scaleFactor > 0.333);
}

inline void ImageViewer::adjustScrollBar(QScrollBar *scrollBar, double factor)
{
	scrollBar->setValue(int(factor * scrollBar->value()
				+ ((factor - 1) * scrollBar->pageStep()/2)));
}

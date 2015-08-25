#include "MainWindow.hpp"
#include "ui_MainWindow.h"

#include <QPixmap>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	mAutomataWrappers.push_back(getConwaysAutomatonWrapper());
	mAutomataWrappers.push_back(getCCLAutomatonWrapper());
	QGraphicsScene *scene = new QGraphicsScene (this);
	ui->mGraphicsView->setScene(scene);

	mGraphicsItem = new QGraphicsPixmapItem();
	scene->addItem(mGraphicsItem);

	mTimer.setInterval(50);

	QObject::connect(&mTimer, &QTimer::timeout, this, &MainWindow::runIteration);
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::runIteration()
{
	getCurrentAutomaton().runIteration();

	getCurrentAutomaton().getCurrentImage(mOutputImage.bits(), mOutputImage.width(), mOutputImage.height(), mOutputImage.bytesPerLine());
	mGraphicsItem->setPixmap(QPixmap::fromImage(mOutputImage));
}

void MainWindow::openImage()
{
	QFileDialog mFileDialog(this, QString("Open Image"));
	mFileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	mFileDialog.setOption(QFileDialog::DontUseNativeDialog, true);
	//QString file = QFileDialog::getOpenFileName(nullptr, QString("Open Image"), QString(), QString());
	if (mFileDialog.exec()) {
		QStringList fileNames = mFileDialog.selectedFiles();
		mInputImage = QImage(fileNames.front());
		if (mInputImage.format() != QImage::Format_RGB888) {
			mInputImage = mInputImage.convertToFormat(QImage::Format_RGB888);
		}
		mGraphicsItem->setPixmap(QPixmap::fromImage(mInputImage));
		//ui->mGraphicsView->scene()->addPixmap(QPixmap::fromImage(mImage));
		mOutputImage = QImage(mInputImage.width(), mInputImage.height(), QImage::Format_RGB888);

		getCurrentAutomaton().setStartImage(mInputImage.bits(), mInputImage.width(), mInputImage.height(), mInputImage.bytesPerLine());
	}
}

void MainWindow::toggleRun(bool aRun)
{
	if (aRun) {
		mTimer.start();
	} else {
		mTimer.stop();
	}
}

void MainWindow::zoomIn()
{
	ui->mGraphicsView->scale(1.1, 1.1);
}

void MainWindow::zoomOut()
{
	ui->mGraphicsView->scale(1 / 1.1, 1/ 1.1);
}

AAutomatonWrapper &MainWindow::getCurrentAutomaton()
{
	return *(mAutomataWrappers[1]);
}

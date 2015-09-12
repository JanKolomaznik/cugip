#include "MainWindow.hpp"
#include "ui_MainWindow.h"

#include <QPixmap>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	mTimer.setInterval(50);

	QObject::connect(&mTimer, &QTimer::timeout, this, &MainWindow::runIteration);

	AutomatonView *view = new AutomatonView();
	ui->mdiArea->addSubWindow(view);
	mAutomataViews.push_back(view);
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::runIteration()
{
	for (auto view : mAutomataViews) {
		view->runIterations(1);
	}
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
		//mGraphicsItem->setPixmap(QPixmap::fromImage(mInputImage));
		//ui->mGraphicsView->scene()->addPixmap(QPixmap::fromImage(mImage));
		//mOutputImage = QImage(mInputImage.width(), mInputImage.height(), QImage::Format_RGB888);

		setImageToAutomata();
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
	for (auto view : mAutomataViews) {
		view->zoomIn();
	}
}

void MainWindow::zoomOut()
{
	for (auto view : mAutomataViews) {
		view->zoomOut();
	}
}

void MainWindow::fitView()
{
	for (auto view : mAutomataViews) {
		view->fitView();
	}
}

void MainWindow::setImageToAutomata()
{
	for (auto view : mAutomataViews) {
		view->setImage(mInputImage);
	}
}

void MainWindow::reset()
{
	for (auto view : mAutomataViews) {
		view->resetAutomaton();
	}
}

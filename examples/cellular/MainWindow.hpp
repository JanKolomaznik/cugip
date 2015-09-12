#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <memory>
#include <vector>
#include <QMainWindow>
#include <QTimer>
#include <QGraphicsPixmapItem>
#include <QFileDialog>

#include "AutomatonWrapper.hpp"
#include "AutomatonView.hpp"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();
public slots:
	void
	runIteration();

	void
	openImage();

	void
	toggleRun(bool aRun);

	void
	zoomIn();

	void
	zoomOut();

	void
	fitView();

	void
	setImageToAutomata();

	void
	reset();
private:
	Ui::MainWindow *ui;

	std::vector<AutomatonView *> mAutomataViews;

	QImage mInputImage;
	QImage mOutputImage;

	QTimer mTimer;
};

#endif // MAINWINDOW_HPP

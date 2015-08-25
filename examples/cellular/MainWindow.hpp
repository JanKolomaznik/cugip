#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <memory>
#include <vector>
#include <QMainWindow>
#include <QTimer>
#include <QGraphicsPixmapItem>
#include <QFileDialog>

#include "AutomatonWrapper.hpp"

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
private:
	Ui::MainWindow *ui;
	AAutomatonWrapper &
	getCurrentAutomaton();

	std::vector<std::unique_ptr<AAutomatonWrapper>> mAutomataWrappers;

	QImage mInputImage;
	QImage mOutputImage;
	QGraphicsPixmapItem *mGraphicsItem;

	QTimer mTimer;
};

#endif // MAINWINDOW_HPP

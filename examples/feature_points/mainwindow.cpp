#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "feature_point_item.h"

#include <QFileDialog>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::openFile()
{
    auto file = QFileDialog::getOpenFileName(this, "Open image...");
    if (file.size()) {
	mImage.load(file);
	mItem = new QGraphicsPixmapItem( QPixmap::fromImage(mImage));
	QGraphicsScene* scene = new QGraphicsScene;
	scene->addItem(mItem);

	scene->addItem(new FeaturePointItem(QPointF(100, 100), 20));

	ui->graphicsView->setScene(scene);
	ui->graphicsView->fitInView(mItem, Qt::KeepAspectRatio);
    }
}

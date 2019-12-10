#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QGraphicsPixmapItem>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void openFile();
private:
    Ui::MainWindow *ui;

    QImage mImage;
    QGraphicsPixmapItem *mItem;
};

#endif // MAINWINDOW_H

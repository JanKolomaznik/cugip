#ifndef AUTOMATONVIEW_HPP
#define AUTOMATONVIEW_HPP

#include "ui_AutomatonView.h"

#include <QGraphicsPixmapItem>

#include "AutomatonWrapper.hpp"


class AutomatonView : public QWidget, private Ui::AutomatonView
{
	Q_OBJECT

public:
	explicit AutomatonView(QWidget *parent = 0);

	void
	runIterations(int aIterationCount);

	void
	setImage(const QImage &aImage);

public slots:
	void
	zoomIn();

	void
	zoomOut();

	void
	fitView();

	void
	resetAutomaton();

	void
	selectAutomaton(int aIndex);
private:
	AAutomatonWrapper &
	getCurrentAutomaton();

	std::vector<std::unique_ptr<AAutomatonWrapper>> mAutomataWrappers;

	QImage mInputImage;
	QImage mOutputImage;
	QGraphicsPixmapItem *mGraphicsItem;
};

#endif // AUTOMATONVIEW_HPP

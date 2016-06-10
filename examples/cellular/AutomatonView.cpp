#include "AutomatonView.hpp"

AutomatonView::AutomatonView(QWidget *parent)
	: QWidget(parent)
{
	setupUi(this);

	mAutomataWrappers.push_back(getConwaysAutomatonWrapper());
	mAutomatonCombo->addItem("Conway's Game of Life");
	mAutomataWrappers.push_back(getCCLAutomatonWrapper());
	mAutomatonCombo->addItem("CCL");
	mAutomataWrappers.push_back(getCCLAutomatonWrapper2());
	mAutomatonCombo->addItem("CCL with global state");
	mAutomataWrappers.push_back(getWShedAutomatonWrapper());
	mAutomatonCombo->addItem("Watersheds");
	mAutomataWrappers.push_back(getWShedAutomaton2Wrapper());
	mAutomatonCombo->addItem("Watersheds 2");
	mAutomataWrappers.push_back(getReactionDiffusionWrapper());
	mAutomatonCombo->addItem("Reaction-Diffusion");

	mAutomatonCombo->setCurrentIndex(0);
	QGraphicsScene *scene = new QGraphicsScene (this);
	mGraphicsView->setScene(scene);

	mGraphicsItem = new QGraphicsPixmapItem();
	scene->addItem(mGraphicsItem);
}

void AutomatonView::runIterations(int aIterationCount)
{
	getCurrentAutomaton().runIterations(aIterationCount);
	getCurrentAutomaton().getCurrentImage(mOutputImage.bits(), mOutputImage.width(), mOutputImage.height(), mOutputImage.bytesPerLine());

	mGraphicsItem->setPixmap(QPixmap::fromImage(mOutputImage));
}

void AutomatonView::setImage(const QImage &aImage)
{
	mInputImage = aImage;

	mGraphicsItem->setPixmap(QPixmap::fromImage(mInputImage));
	mOutputImage = QImage(mInputImage.width(), mInputImage.height(), QImage::Format_RGB888);

	resetAutomaton();
}

void AutomatonView::zoomIn()
{
	mGraphicsView->scale(1.1, 1.1);
}

void AutomatonView::zoomOut()
{
	mGraphicsView->scale(1 / 1.1, 1/ 1.1);
}

void AutomatonView::fitView()
{
	mGraphicsView->fitInView(mGraphicsItem, Qt::KeepAspectRatio);
}

void AutomatonView::resetAutomaton()
{
	if (!mInputImage.isNull()) {
		getCurrentAutomaton().setStartImage(mInputImage.bits(), mInputImage.width(), mInputImage.height(), mInputImage.bytesPerLine());
		getCurrentAutomaton().getCurrentImage(mOutputImage.bits(), mOutputImage.width(), mOutputImage.height(), mOutputImage.bytesPerLine());
		mGraphicsItem->setPixmap(QPixmap::fromImage(mOutputImage));
		mPreprocessingCheckBox->setChecked(getCurrentAutomaton().preprocessingEnabled());
	}
}

void AutomatonView::selectAutomaton(int aIndex)
{
	resetAutomaton();
}

void AutomatonView::enablePreprocessing(bool aEnabled)
{
	if (getCurrentAutomaton().preprocessingEnabled() != aEnabled) {
		getCurrentAutomaton().enablePreprocessing(aEnabled);
		resetAutomaton();
	}
}

AAutomatonWrapper &AutomatonView::getCurrentAutomaton()
{
	return *(mAutomataWrappers.at(mAutomatonCombo->currentIndex()));
	//return *(mAutomataWrappers.back());
}

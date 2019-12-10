#ifndef FEATURE_POINT_ITEM_H
#define FEATURE_POINT_ITEM_H


#include <QGraphicsItem>

class FeaturePointItem : public QGraphicsItem
{
public:
	FeaturePointItem(const QPointF &aPoint=QPointF(), double aRadius=1.0);

	QRectF boundingRect() const;

	// overriding paint()
	void paint(
		QPainter * painter,
		const QStyleOptionGraphicsItem * option,
		QWidget * widget) override;
private:
	QPointF mCenter;
	double mRadius = 1.0f;
};

#endif // FEATURE_POINT_ITEM_H

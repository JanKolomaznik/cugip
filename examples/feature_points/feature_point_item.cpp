#include "feature_point_item.h"

#include <QPen>
#include <QPainter>

FeaturePointItem::FeaturePointItem(const QPointF &aPoint, double aRadius)
	: mCenter(aPoint)
	, mRadius(aRadius)
{

}

QRectF FeaturePointItem::boundingRect() const
{
	return QRectF(mCenter - QPointF(mRadius, mRadius), mCenter + QPointF(mRadius, mRadius));
}

void FeaturePointItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
	QPen pen(Qt::green, 2);
	painter->setPen(pen);
	painter->drawEllipse(mCenter, mRadius, mRadius);
}


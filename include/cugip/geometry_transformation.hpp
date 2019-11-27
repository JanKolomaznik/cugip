#pragma once

#include <cugip/for_each.hpp>

namespace cugip {

template<typename TView, int tDim, typename std::enable_if<is_interpolated_view<TView>::value, int>::type = 0>
CUGIP_DECL_HYBRID simple_vector<float, tDim>
coordinatesFromIndex(TView aView, simple_vector<int, tDim> aIndex)
{
	return aView.coordinates_from_index(aIndex);
}

template<typename TView, int tDim>
CUGIP_DECL_HYBRID simple_vector<float, tDim>
coordinatesFromIndex(TView aView, simple_vector<int, tDim> aIndex)
{
	return aIndex + simple_vector<float, tDim>(0.5f, FillFlag());
}


template<typename TInputView, typename TOutputView, typename TInverseTransformation>
struct TransformationFunctor
{
	CUGIP_HD_WARNING_DISABLE
	template<typename TValue, typename TIndex>
	CUGIP_DECL_HYBRID
	void operator()(TValue &aValue, TIndex aIndex) const
	{
		//std::cout << "From " << aIndex << "to " << transformation(coordinatesFromIndex(output, aIndex)) << "\n";
		aValue = input.interpolated_value(transformation(coordinatesFromIndex(output, aIndex)));
	}

	TInputView input;
	TOutputView output;
	TInverseTransformation transformation;
};

template<typename TInputView, typename TOutputView, typename TInverseTransformation>
TransformationFunctor<TInputView, TOutputView, TInverseTransformation>
makeTransformationFunctor(TInputView aInput, TOutputView aOutput, TInverseTransformation aInverseTransformation)
{
	return TransformationFunctor<TInputView, TOutputView, TInverseTransformation>{ aInput, aOutput, aInverseTransformation };
}

template<typename TInputView, typename TOutputView, typename TInverseTransformation>
void geometry_transformation(TInputView aInput, TOutputView aOutput, TInverseTransformation aInverseTransformation)
{
	static_assert(is_interpolated_view<TInputView>::value, "Input view must provide interpolated access for the geometry transformation!");
	for_each_position(aOutput, makeTransformationFunctor(aInput, aOutput, aInverseTransformation));
}


template<typename TCoordType>
struct RotationTransformation
{
	CUGIP_DECL_HYBRID simple_vector<TCoordType, 2>
	operator()(const simple_vector<TCoordType, 2> &aPoint) const
	{
		return offset + rotate(aPoint - offset, angle);
	}
	simple_vector<TCoordType, 2> offset;
	double angle;
};

template<typename TCoordType>
RotationTransformation<TCoordType>
getInverseRotation(simple_vector<TCoordType, 2> aCenter, double aAngle)
{
	return RotationTransformation<TCoordType>{ aCenter, -aAngle };
}

template<typename TInputView, typename TOutputView, typename TOffsetCoordinates>
void rotate(TInputView aInput, TOutputView aOutput, TOffsetCoordinates aCenter, double aAngle)
{
	geometry_transformation(aInput, aOutput, getInverseRotation(aCenter, aAngle));
}

template<typename TScalingVector, typename TCoordinateType>
struct ScalingTransformation
{
	CUGIP_DECL_HYBRID TCoordinateType
	operator()(const TCoordinateType &aPoint) const
	{
		return offset + product(aPoint - offset, scale);
	}
	TCoordinateType offset;
	TScalingVector scale;
};

template<typename TScalingVector, typename TCoordinateType>
ScalingTransformation<TScalingVector, TCoordinateType>
getInverseScaling(TCoordinateType aAnchor, TScalingVector aScaleVector)
{
	return ScalingTransformation<TScalingVector, TCoordinateType>{ aAnchor, div(TScalingVector(1.0, FillFlag()), aScaleVector) };
}

template<typename TInputView, typename TOutputView, typename TOffsetCoordinates, typename TScaleVector>
void scale(TInputView aInput, TOutputView aOutput, TOffsetCoordinates aAnchor, TScaleVector aScaleVector)
{
	geometry_transformation(aInput, aOutput, getInverseScaling(aAnchor, aScaleVector));
}

template<typename TInputView, typename TOutputView>
void scale(TInputView aInput, TOutputView aOutput, float aFactor)
{
	static constexpr int cDimension = dimension<TInputView>::value;
	simple_vector<float, cDimension> anchor;
	simple_vector<float, cDimension> scalingVector(aFactor, FillFlag());
	geometry_transformation(aInput, aOutput, getInverseScaling(anchor, scalingVector));
}

}  // namespace cugip

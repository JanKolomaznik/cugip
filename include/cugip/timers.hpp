// Copyright (c) 2016  by FEI Company. All rights reserved.
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <chrono>
#include <ratio>
#include <bitset>
#include <sstream>
#include <vector>

namespace cugip {


template<typename TId>
struct DurationRecord {
	typedef std::chrono::duration<double> Duration;
	typedef TId Id;

	DurationRecord():
		sum(Duration::zero()),
		max_duration(Duration::min()),
		min_duration(Duration::max()),
		id_for_max(-1),
		id_for_min(-1),
		count(0)
	{}

	void Update(Duration duration, TId id) {
		sum += duration;
		++count;
		if (max_duration < duration) {
			max_duration = duration;
			id_for_max = id;
		}
		if (min_duration > duration) {
			min_duration = duration;
			id_for_min = id;
		}
	}

	Duration sum;
	Duration max_duration;
	Duration min_duration;
	TId id_for_max;
	TId id_for_min;
	int count;
};

template<typename TTimerSet, int ...tTimerId>
class MultiIntervalMeasurement {
public:
	MultiIntervalMeasurement():
		timer_set_(nullptr)
	{}

	MultiIntervalMeasurement(TTimerSet *timer_set, typename TTimerSet::Id id) :
		start_(std::chrono::high_resolution_clock::now()),
		id_(id),
		timer_set_(timer_set)
	{
		ignoreReturnValues((active_durations_.set(tTimerId), 0)...);
	}

	MultiIntervalMeasurement(MultiIntervalMeasurement &&other):
		start_(std::move(other.start_)),
		id_(std::move(other.id_)),
		timer_set_(other.timer_set_),
		active_durations_(std::move(other.active_durations_))
	{
		other.timer_set_ = nullptr;
	}

	~MultiIntervalMeasurement() {
		stopAll();
	}

	void stopAll() {
		if (!timer_set_) {
			return;
		}
		auto now = std::chrono::high_resolution_clock::now();
		ignoreReturnValues(handleTimerStop<tTimerId>(now)...);
	}


	template<int ...tStoppedTimer>
	void stop() {
		assert(timer_set_ != nullptr);
		auto now = std::chrono::high_resolution_clock::now();
		ignoreReturnValues(handleTimerStop<tStoppedTimer>(now)...);
	}

	template<int ...tResetedTimer>
	void reset() {
		if (!timer_set_) {
			return;
		}
		ignoreReturnValues((active_durations_[tResetedTimer] = false)...);
	}

	MultiIntervalMeasurement &operator=(MultiIntervalMeasurement &&other) {
		stopAll();
		start_ = std::move(other.start_);
		id_ = std::move(other.id_);
		timer_set_ = other.timer_set_;
		active_durations_ = std::move(other.active_durations_);
		other.timer_set_ = nullptr;
		return *this;
	}

protected:
	template<int tTimer>
	int handleTimerStop(std::chrono::high_resolution_clock::time_point now) {
		if (active_durations_[tTimer]) {
			active_durations_[tTimer] = false;
			auto duration = now - start_;
			timer_set_->template handleDurationForTimerS<tTimer>(duration, id_);
		}
		return 0;
	}

	std::chrono::high_resolution_clock::time_point start_;
	typename TTimerSet::Id id_;

	TTimerSet *timer_set_;

	std::bitset<TTimerSet::kTimerCount> active_durations_;
};

template<typename TTimerSet>
class IntervalMeasurement {
public:
	IntervalMeasurement() :
		timer_index_(-1)
	{}

	IntervalMeasurement(int timer_index, typename TTimerSet::Id id, TTimerSet *timer_set) :
		start_(std::chrono::high_resolution_clock::now()),
		id_(id),
		timer_index_(timer_index),
		timer_set_(timer_set)
	{
	}

	IntervalMeasurement(IntervalMeasurement &&interval) :
		start_(interval.start_),
		id_(interval.id_),
		timer_index_(interval.timer_index_),
		timer_set_(interval.timer_set_)
	{
		interval.timer_index_ = -1;
	}

	IntervalMeasurement &operator=(IntervalMeasurement &&interval) {
		start_ = interval.start_;
		id_ = interval.id_;
		timer_index_  = interval.timer_index_;
		timer_set_ = interval.timer_set_;
		interval.timer_index_ = -1;
		return *this;
	}


	~IntervalMeasurement() {
		stop();
	}

	void stop() {
		if (timer_index_ >= 0) {
			auto now = std::chrono::high_resolution_clock::now();
			auto duration = now - start_;
			timer_set_->handleDurationForTimer(timer_index_, duration, id_);
			timer_index_ = -1;
		}
	}

protected:
	std::chrono::high_resolution_clock::time_point start_;
	typename TTimerSet::Id id_;

	int timer_index_;

	TTimerSet *timer_set_;
};


template <int tTimerCount, typename TId = int>
class AggregatingTimerSet {
public:
	typedef AggregatingTimerSet<tTimerCount, TId> This;
	typedef IntervalMeasurement<This> SingleInterval;
	//template<int ...tTimerId>
	//friend class MultiIntervalMeasurement<AggregatingTimerSet<tTimerCount, TId>, tTimerId...>;

	typedef TId Id;
	static constexpr int kTimerCount = tTimerCount;

	/*template<int tTimerIndex>
	SingleInterval Start(TId interval_id) {
		return SingleInterval(tTimerIndex, interval_id, this);
	}*/

	SingleInterval start(int timer_index, TId interval_id) {
		return SingleInterval(timer_index, interval_id, this);
	}

	template<int ...tStartedTimer>
	MultiIntervalMeasurement<This, tStartedTimer...>
	start(TId interval_id) {
		return MultiIntervalMeasurement<This, tStartedTimer...>(this, interval_id);
	}

	template<int tTimerIndex>
	void handleDurationForTimerS(std::chrono::duration<double> duration, TId interval_id) {
		static_assert(tTimerIndex < tTimerCount && tTimerIndex >= 0, "Timer index out of range");
		timing_records_[tTimerIndex].Update(duration, interval_id);
	}

	void handleDurationForTimer(int timer_id, std::chrono::duration<double> duration, TId interval_id) {
		assert(timer_id < tTimerCount && timer_id >= 0);
		timing_records_[timer_id].Update(duration, interval_id);
	}

	std::string createReport(const std::array<std::string, tTimerCount> &names) {
		assert(int(names.size()) == tTimerCount);
		std::ostringstream stream;
		for (int i = 0; i < tTimerCount; ++i) {
			stream << "    " << names[i] << " -> " << createTimerReport(i) << '\n';
		};
		return stream.str();
	}
	std::string createCompactReport(const std::array<std::string, tTimerCount> &names) {
		assert(int(names.size()) == tTimerCount);
		std::ostringstream stream;
		for (int i = 0; i < tTimerCount; ++i) {
			stream << names[i] << ": " << createTimerCompactReport(i) << "| ";
		};
		return stream.str();
	}
protected:
	std::string createTimerReport(int index) {
		if (timing_records_[index].count == 0) {
			return "    NO DURATION MEASUREMENT";
		}
		return boost::str(boost::format("sum: %1%, avg: %2%, max: %3% (%4%), min: %5% (%6%), count: %7%")
			% timing_records_[index].sum.count()
			% (timing_records_[index].sum.count() / timing_records_[index].count)
			% timing_records_[index].max_duration.count()
			% timing_records_[index].id_for_max
			% timing_records_[index].min_duration.count()
			% timing_records_[index].id_for_min
			% timing_records_[index].count
			);
	}

	std::string createTimerCompactReport(int index) {
		if (timing_records_[index].count == 0) {
			return " NONE";
		}
		return boost::str(boost::format("%1%, %2%, %3% (%4%), %5% (%6%), %7%")
			% timing_records_[index].sum.count()
			% (timing_records_[index].sum.count() / timing_records_[index].count)
			% timing_records_[index].max_duration.count()
			% timing_records_[index].id_for_max
			% timing_records_[index].min_duration.count()
			% timing_records_[index].id_for_min
			% timing_records_[index].count
			);
	}

	std::array<DurationRecord<TId>, tTimerCount> timing_records_;
};

} //namespace cugip

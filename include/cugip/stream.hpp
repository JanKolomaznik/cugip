#pragma once

#include <cugip/cuda_utils.hpp>


namespace cugip {

struct MovedFromFlag {
	MovedFromFlag() :
		moved_from_(false)
	{}

	MovedFromFlag(MovedFromFlag &&other) :
		moved_from_(other.moved_from_)
	{
		other.moved_from_ = true;
	}

	MovedFromFlag &operator=(MovedFromFlag &&other) {
		moved_from_ = other.moved_from_;
		other.moved_from_ = true;
		return *this;
	}

	MovedFromFlag(const MovedFromFlag &other) = delete;
	MovedFromFlag &operator=(const MovedFromFlag &other) = delete;

	bool isMoved() const {
		return moved_from_;
	}

	bool isValid() const {
		return !moved_from_;
	}

	bool moved_from_;
};


/// RAII wrapper for cudaStream_t
class CudaStream {
	public:
	CudaStream() {
		CUGIP_CHECK_RESULT(cudaStreamCreate(&stream_));
	}

	explicit CudaStream(unsigned int flags) {
		CUGIP_CHECK_RESULT(cudaStreamCreateWithFlags(&stream_, flags));
	}

	CudaStream(unsigned int flags, int priority) {
		CUGIP_CHECK_RESULT(cudaStreamCreateWithPriority(&stream_, flags, priority));
	}

	CudaStream(CudaStream &&) = default;
	CudaStream &operator=(CudaStream &&) = default;

	/// Query the stream for completion status. True if all the work on the stream is completed
	bool query() {
		auto err = cudaStreamQuery(stream_);
		if (err == cudaSuccess) {
			return true;
		} else if (err == cudaErrorNotReady) {
			return false;
		} else {
			CUGIP_CHECK_ERROR_STATE("Error querying the stream for completion status");
			return false;
		}
	}

	/// waits for stream tasks to complete
	void synchronize() {
		CUGIP_CHECK_RESULT(cudaStreamSynchronize(stream_));
	}

	/// makes the stream wait for the event
	void waitForEvent(cudaEvent_t event) {
		// flag must be zero for now
		CUGIP_CHECK_RESULT(cudaStreamWaitEvent(stream_, event, 0));
	}

	/// Adds callback after the current items. The callback will block further work until finished,
	/// and must NOT make any CUDA API calls.
	void addCallback(cudaStreamCallback_t callback, void *userData) {
		// flag must be zero for now
		CUGIP_CHECK_RESULT(cudaStreamAddCallback(stream_, callback, userData, 0));
	}

	cudaStream_t get() {
		return stream_;
	}

	CudaStream(const CudaStream &) = delete;
	CudaStream &operator=(const CudaStream &) = delete;

	~CudaStream() {
		if (flag_.isValid()) {
			try {
				CUGIP_CHECK_RESULT(cudaStreamDestroy(stream_));
			} catch (ExceptionBase &e) {
				CUGIP_EFORMAT("cudaStream destruction failure.: %1%", boost::diagnostic_information(e));
			}
		}
	}

	private:
	cudaStream_t stream_;
	MovedFromFlag flag_;
};

}  // namespace cugip

// -----------------------------------------------------------------------------
// Shared GPU plumbing: the number type, error checking, owning device buffers
// and an event timer.
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace adi {

using real = double;  // every array on the GPU; a later FP32 experiment starts here

// Throws std::runtime_error naming the failed call, file and line.
inline void cuda_check(cudaError_t status, const char* call, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + call + ": " +
                                 cudaGetErrorString(status));
    }
}

}  // namespace adi

#define CUDA_CHECK(call) ::adi::cuda_check((call), #call, __FILE__, __LINE__)

// Use after every kernel launch. It reports launch errors at once; with ADI_DEBUG_SYNC it also waits
// for the kernel, so an error in it is reported at the launch that caused it.
#ifdef ADI_DEBUG_SYNC
#define CUDA_CHECK_LAST()                          \
    do {                                           \
        CUDA_CHECK(cudaGetLastError());            \
        CUDA_CHECK(cudaDeviceSynchronize());       \
    } while (0)
#else
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
#endif

namespace adi {

// Owns `count` elements of device memory: allocated in the constructor, freed in the destructor.
// Move-only, so every buffer has exactly one owner.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
    }
    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) { upload(host); }
    ~DeviceBuffer() {
        if (data_) cudaFree(data_);  // no CUDA_CHECK: destructors must not throw
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }

    void upload(const std::vector<T>& host) {
        if (host.size() != count_) throw std::runtime_error("DeviceBuffer::upload: size mismatch");
        if (count_ > 0) CUDA_CHECK(cudaMemcpy(data_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(std::vector<T>& host) const {
        host.resize(count_);
        if (count_ > 0) CUDA_CHECK(cudaMemcpy(host.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        if (count_ > 0) CUDA_CHECK(cudaMemset(data_, 0, count_ * sizeof(T)));
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// Times work on one stream with a pair of CUDA events.
class EventTimer {
public:
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    void start(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
    void stop(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(stop_, stream)); }

    // Waits for the stop event, then returns the milliseconds between the two events.
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

}  // namespace adi

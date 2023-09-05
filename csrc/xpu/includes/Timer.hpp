
#ifndef __TIMER_H__
#define __TIMER_H__

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <chrono>

class GPUTimer {
    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

public:
    GPUTimer() {}
    ~GPUTimer() {}

    inline void Record() { start_ct1 = std::chrono::steady_clock::now(); }
    inline void Elapsed(float& time_elapsed)
    {
        stop_ct1 = std::chrono::steady_clock::now();
        stop.wait_and_throw();
        time_elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    }
};

class CPUTimer {
    std::chrono::high_resolution_clock::time_point start;

public:
    CPUTimer() : start(std::chrono::high_resolution_clock::now()) {}
    inline void Reset() { start = std::chrono::high_resolution_clock::now(); }
    inline float Elapsed()
    {
        auto temp = start;
        start = std::chrono::high_resolution_clock::now();
        return (float)(std::chrono::duration_cast<std::chrono::microseconds>(start - temp).count() /
                       1e3);
    }
};

#endif

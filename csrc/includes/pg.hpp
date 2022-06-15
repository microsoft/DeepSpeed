#pragma once

#include <torch/python.h>

#include <pybind11/chrono.h>

enum class OpType : std::uint8_t {
    BROADCAST = 0,
    ALLREDUCE = 1,
    ALLREDUCE_COALESCED = 2,
    REDUCE = 3,
    ALLGATHER = 4,
    _ALLGATHER_BASE = 5,
    ALLGATHER_COALESCED = 6,
    GATHER = 7,
    SCATTER = 8,
    REDUCE_SCATTER = 9,
    ALLTOALL_BASE = 10,
    ALLTOALL = 11,
    SEND = 12,
    RECV = 13,
    RECVANYSOURCE = 14,
    BARRIER = 15,
    _REDUCE_SCATTER_BASE = 16,
    UNKNOWN = 100,
};

constexpr auto kNoTimeout = std::chrono::milliseconds(0);
constexpr auto kProcessGroupDefaultTimeout = std::chrono::milliseconds(30 * 60 * 1000);

class ProcessGroup {
public:
    virtual bool is Completed();
    virtual bool isSuccess();
    virtual std::exception_ptr exception() const;
    virtual int sourceRank() const;
    virtual std::vector<at::Tensor> result();
    virtual void synchronize();
    virtual bool wait(std::chrono::milliseconds timeout = kNoTimeout);
    virtual void abort();
    // virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture();
    OpType retrieveOpType();

protected:
    void init() void finish(std::exception_ptr exception = nullptr);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool completed_ = false;
    std::exception_ptr exception_;

    const int rank_;
    const int size_;
    OpType opType_;

    struct Options {
        std::string backend, std::chrono::milliseconds timeout = kProcessGroupDefaultTimeout
    };

    virtual std::string getBackendName() const = 0;
    int getRank() { return rank_; }
    int getSize() { return size_; }
}
#include <pg.hpp>

#include

OpType ProcessGroup::retrieveOpType() { return opType_; }

bool ProcessGroup::isCompleted()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
}

bool ProcessGroup::isSuccess() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return !exception_;
}

std::exception_ptr ProcessGroup::exception() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return exception_;
}

int ProcessGroup::sourceRank() const
{
    TORCH_CHECK(false,
                "sourceRank() may only be called on work objects "
                "that correspond to a recv or recv-from-any call.");
}

std::vector<at::Tensor> ProcessGroup::result() { TORCH_CHECK(false, "result() not implemented."); }

void ProcessGroup::synchronize() {}

bool ProcessGroup::wait(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout == kNoTimeout) {
        // This waits without a timeout.
        cv_.wait(lock, [&] { return completed_; });
    } else {
        // Waits for the user-provided timeout.
        cv_.wait_for(lock, timeout, [&] { return completed_; });
        if (!completed_) {
            // Throw exception if the wait operation timed out and the work was not
            // completed.
            TORCH_CHECK(false, "Operation timed out!");
        }
    }
    if (exception_) { std::rethrow_exception(exception_); }
    synchronize();
    // Always return true, because abort API is not implemented.
    return true;
}

void ProcessGroup::abort() { TORCH_CHECK(false, "ProcessGroup::abort not implemented."); }

// c10::intrusive_ptr<c10::ivalue::Future> ProcessGroup::getFuture() {
//  TORCH_CHECK(false, "ProcessGroup::getFuture not implemented.")
//}

void ProcessGroup::finish(std::exception_ptr exception)
{
    std::unique_lock<std::mutex> lock(mutex_);
    completed_ = true;
    exception_ = exception;
    if (recordFunctionEndCallback_) {
        recordFunctionEndCallback_();
        recordFunctionEndCallback_ = nullptr;
    }
    lock.unlock();
    cv_.notify_all();
}

void ProcessGroup::finishAndThrow(std::exception_ptr exception)
{
    std::unique_lock<std::mutex> lock(mutex_);
    completed_ = true;
    exception_ = exception;
    if (recordFunctionEndCallback_) {
        recordFunctionEndCallback_();
        recordFunctionEndCallback_ = nullptr;
    }
    if (exception_) { std::rethrow_exception(exception_); }
}

ProcessGroup::ProcessGroup(int rank, int size)
    : rank_(rank), size_(size), dist_debug_level_(debug_level())
{
}

ProcessGroup::~ProcessGroup() {}

void ProcessGroup::init() { cout << "init\n"; }
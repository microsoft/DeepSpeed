#include "nvme_device.h"

extern "C" DeepSpeedAIOBase* create_device() {
    return new NVMEDevice();
}

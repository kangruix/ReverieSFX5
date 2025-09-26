#pragma once

enum DeviceType : int {
    CPU = 0,
    CUDA = 1
};

struct Device {
    DeviceType type;
    int index;

    Device(DeviceType t = DeviceType::CPU, int idx = 0) : type(t), index(idx) {}

    bool is_cuda() const { return type == DeviceType::CUDA; }
    bool is_cpu() const { return type == DeviceType::CPU; }
};
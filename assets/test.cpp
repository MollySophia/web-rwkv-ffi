#include <iostream>
#include <dlfcn.h>
#include "web_rwkv_ffi.h"

int main() {
    // load libweb_rwkv_ffi.dylib
    void *handle = dlopen("../target/release/libweb_rwkv_ffi.dylib", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << std::endl;
        return 1;
    }

    void (*init)(uint64_t) = (void (*)(uint64_t))dlsym(handle, "init");
    void (*load_with_rescale)(const char *, uintptr_t, uintptr_t, uintptr_t) = (void (*)(const char *, uintptr_t, uintptr_t, uintptr_t))dlsym(handle, "load_with_rescale");
    uint16_t (*infer)(const uint16_t *, uintptr_t, struct Sampler) = (uint16_t (*)(const uint16_t *, uintptr_t, struct Sampler))dlsym(handle, "infer");

    init(0);
    load_with_rescale("../rwkv-sudoku.st", 0, 0, 999);
    uint16_t prompt[] = {102, 1, 1, 9, 2, 7, 8, 1, 3, 1, 132, 6, 1, 1, 3, 4, 1, 1, 1, 1, 132, 8, 7, 1, 1, 6, 5, 9, 1, 2, 132, 9, 8, 1, 1, 5, 1, 1, 1, 1, 132, 1, 3, 1, 1, 1, 1, 1, 1, 1, 132, 1, 1, 5, 1, 1, 4, 1, 10, 1, 132, 1, 1, 1, 1, 1, 1, 4, 8, 1, 132, 1, 5, 1, 1, 1, 1, 1, 9, 1, 132, 4, 2, 1, 9, 1, 7, 10, 1, 5, 132, 103};
    struct Sampler sampler = {1.0, 1.0, 0};
    uint16_t output = infer(prompt, sizeof(prompt) / sizeof(uint16_t), sampler);
    std::cout << "output: [" << output << ", ";
    for (int i = 0; i < 20; i++) {
        output = infer(&output, 1, sampler);
        std::cout << output << ", ";
    }
    return 0;
}
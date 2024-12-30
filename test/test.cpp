#include <iostream>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include "web_rwkv_ffi.h"
#include "trie.hpp"

int main() {
#ifdef _WIN32
    HMODULE handle = LoadLibrary("../target/release/web_rwkv_ffi.dll");
    if (handle == NULL) {
        std::cerr << "Cannot open library: " << GetLastError() << std::endl;
        return 1;
    }

    void (*init)(uint64_t) = (void (*)(uint64_t))GetProcAddress(handle, "init");
    void (*load_with_rescale)(const char *, uintptr_t, uintptr_t, uintptr_t) = (void (*)(const char *, uintptr_t, uintptr_t, uintptr_t))GetProcAddress(handle, "load_with_rescale");
    uint16_t (*infer)(const uint16_t *, uintptr_t, struct Sampler) = (uint16_t (*)(const uint16_t *, uintptr_t, struct Sampler))GetProcAddress(handle, "infer");
#else
    void *handle = dlopen("../target/release/libweb_rwkv_ffi.dylib", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << std::endl;
        return 1;
    }

    void (*init)(uint64_t) = (void (*)(uint64_t))dlsym(handle, "init");
    void (*load_with_rescale)(const char *, uintptr_t, uintptr_t, uintptr_t) = (void (*)(const char *, uintptr_t, uintptr_t, uintptr_t))dlsym(handle, "load_with_rescale");
    uint16_t (*infer)(const uint16_t *, uintptr_t, struct Sampler) = (uint16_t (*)(const uint16_t *, uintptr_t, struct Sampler))dlsym(handle, "infer");

#endif
    TRIE_TOKENIZER tokenizer("b_rwkv_vocab_v20230424.txt");
    init(0);
    load_with_rescale("../RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.st", 0, 0, 999);

    std::string prompt_str = "The Eiffel Tower is in the city of";
    auto prompt_ids = tokenizer.encode(prompt_str);
    std::vector<uint16_t> prompt(prompt_ids.begin(), prompt_ids.end());
    struct Sampler sampler = {1.0, 1.0, 0};
    uint16_t output = infer(prompt.data(), prompt.size(), sampler);
    std::cout << prompt_str << '[';
    for (int i = 0; i < 20; i++) {
        output = infer(&output, 1, sampler);
        std::cout << tokenizer.decode({output});
    }
    return 0;
}
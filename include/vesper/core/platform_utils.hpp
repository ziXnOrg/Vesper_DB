#pragma once

#include <optional>
#include <string>
#include <cstdlib>

namespace vesper::core {

// Cross-platform safe getenv wrapper.
// - Windows: uses _dupenv_s and frees the allocated buffer
// - POSIX/others: uses std::getenv (read-only)
// Returns std::nullopt if the variable is not set. If set but empty, returns an
// engaged optional with an empty string.
inline std::optional<std::string> safe_getenv(const char* name) noexcept {
    if (name == nullptr || *name == '\0') return std::nullopt;
#if defined(_WIN32)
    char* buf = nullptr;
    size_t len = 0;
    const errno_t err = _dupenv_s(&buf, &len, name);
    if (err != 0 || buf == nullptr) {
        if (buf) std::free(buf);
        return std::nullopt;
    }
    // _dupenv_s returns a NUL-terminated string; len includes the terminator.
    std::string value(buf);
    std::free(buf);
    return value;
#else
    const char* v = std::getenv(name);
    if (!v) return std::nullopt;
    return std::string(v);
#endif
}

} // namespace vesper::core


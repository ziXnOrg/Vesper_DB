#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <string>
#include "vesper/core/platform_utils.hpp"

#if defined(_WIN32)
  #include <cstdlib>
#else
  #include <cstdlib>
#endif

using vesper::core::safe_getenv;

static void set_env_var(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

static void unset_env_var(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

TEST_CASE("safe_getenv returns nullopt when unset", "[platform][env]") {
    const char* key = "VESPER_TEST_SAFE_GETENV_UNSET";
    unset_env_var(key);
    auto v = safe_getenv(key);
    REQUIRE_FALSE(v.has_value());
}

TEST_CASE("safe_getenv returns value when set", "[platform][env]") {
    const char* key = "VESPER_TEST_SAFE_GETENV_VALUE";
    set_env_var(key, "hello_world");
    auto v = safe_getenv(key);
    REQUIRE(v.has_value());
    REQUIRE(*v == std::string("hello_world"));
    unset_env_var(key);
}

TEST_CASE("safe_getenv returns empty string when set to empty (POSIX) or nullopt on Windows", "[platform][env]") {
    const char* key = "VESPER_TEST_SAFE_GETENV_EMPTY";
    set_env_var(key, "");
    auto v = safe_getenv(key);
#if defined(_WIN32)
    // On Windows CRT, setting to empty removes the variable
    REQUIRE_FALSE(v.has_value());
#else
    REQUIRE(v.has_value());
    REQUIRE(v->empty());
#endif
    unset_env_var(key);
}


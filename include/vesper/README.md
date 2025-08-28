# Vesper API (Phase 3 Planning)

This directory will contain public C++20 headers and the stable C ABI header. All public APIs use Doxygen comments with pre/post-conditions, complexity, thread-safety, and ownership.

Planned files
- vesper.hpp (umbrella)
- collection.hpp, segment.hpp, search_params.hpp, filter_expr.hpp
- error.hpp (std::expected error taxonomy)
- vesper_c.h (C ABI, POD types only)


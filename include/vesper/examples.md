# API Examples (illustrative)

## Open, insert, and search (C++)
```cpp
#include <vesper/collection.hpp>
#include <vesper/filter_expr.hpp>
using vesper::collection;

auto main() -> int {
  auto coll = collection::open("/tmp/vesper_coll");
  if (!coll) { /* handle coll.error() */ return 1; }
  float q[4]{1,2,3,4};
  (void)coll->insert(42, q, 4);
  vesper::search_params p{.metric="l2", .k=10};
  auto res = coll->search(q, 4, p, nullptr);
  return res ? 0 : 1;
}
```

## C ABI search
```c
#include <vesper/vesper_c.h>

int main(){
  vesper_collection_t* c = 0;
  if (vesper_open_collection("/tmp/vesper_coll", &c) != VESPER_OK) return 1;
  float q[4] = {1,2,3,4};
  vesper_search_params_t p = {"l2", 10, 0.95f, 8, 64, 0};
  vesper_search_result_t out[10]; size_t out_size=0;
  vesper_status_t st = vesper_search(c, q, 4, &p, out, 10, &out_size);
  vesper_close_collection(c);
  return st == VESPER_OK ? 0 : 1;
}
```


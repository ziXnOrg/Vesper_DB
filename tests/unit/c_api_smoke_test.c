#include <vesper/vesper_c.h>
#include <assert.h>
#include <stdio.h>

int main(void){
  vesper_collection_t* c = NULL;
  vesper_status_t st = vesper_open_collection("/tmp/vesper_coll_c", &c);
  assert(st == VESPER_OK);
  assert(c != NULL);
  vesper_close_collection(c);
  printf("C API smoke ok\n");
  return 0;
}


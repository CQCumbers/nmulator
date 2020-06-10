#include "r4300.h"

using namespace asmjit;
JitRuntime runtime;
Block empty = {};

int main(int argc, char* argv[]) {
  // initialize system components
  R4300::init(argv[1]);
  if (argc == 3) {
    int port = atoi(argv[2]);
    R4300::init_debug(port);
  }

  // add tasks to scheduler
  Sched::init(TASK_R4300, 0);
  Sched::add(TASK_VI, 6510);
  Sched::start_loop();
}

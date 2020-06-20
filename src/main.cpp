#include <stdlib.h>
#include "nmulator.h"

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
  Sched::add(TASK_CIC, 65100);
  Sched::start_loop();
}

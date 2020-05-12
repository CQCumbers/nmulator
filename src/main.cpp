#include "r4300.h"
#include "debugger.h"

int main(int argc, char* argv[]) {
  // initialize system components
  if (argc != 2 && argc != 3)// printf("error: must provide file\n"), exit(1);
    Vulkan::run_buffer();
  FILE *file = fopen(argv[1], "r");
  if (!file) printf("error: can't open file\n"), exit(1);
  R4300::init(file), fclose(file);
  if (argc == 3) Debugger::init(atoi(argv[2]));

  Sched::add(R4300::update, 0);
  Sched::add(R4300::vi_update, 6510);
  Sched::start_loop();
}

#include "nmulator.h"

/* === Global variables === */

namespace Sched { int64_t until; }
static uint64_t ev_array[16];
static uint64_t *events = ev_array + 15;
static uint64_t *const top = events;

typedef void (*task_t)();
static const task_t tasks[] = {
  RSP::update,
  R4300::update,
  RDP::update,
  R4300::vi_update,
  R4300::ai_update,
  R4300::pi_update,
  R4300::cic_update,
  R4300::timer_fire
};

/* === Internal helpers === */

// Compare event times without overflow
// returns true if lhs is after rhs
static bool after(uint64_t lhs, uint64_t rhs) {
  return (int64_t)rhs - (int64_t)lhs < 0;
}

// Insert event into sorted events array,
// searching forward from ptr
static void move_down(uint64_t *ptr, uint64_t event) {
  while (ptr < top && after(event, ptr[1]))
    ptr[0] = ptr[1], ++ptr;
  ptr[0] = event;
}

// Insert event into sorted events array,
// searching backwards from ptr
static void move_up(uint64_t *ptr, uint64_t event) {
  while (ptr + 1 >= events && after(ptr[0], event))
    ptr[1] = ptr[0], --ptr;
  ptr[1] = event;
}

/* === Public interface === */

// Add first task to event queue
void Sched::init(uint8_t task, int64_t time) {
  events[0] = (time << 8) | task, until = time;
}

// Schedule new task to be executed
// a certain time into the future
void Sched::add(uint8_t task, int64_t time) {
  uint64_t next = events[0] >> 8;
  uint64_t e = ((next - until + time) << 8) | task;
  if (time < until) until = time;
  move_down(--events, e);
}

// Change time of an already scheduled task
void Sched::move(uint8_t task, int64_t time) {
  uint64_t next = events[0] >> 8;
  uint64_t e = ((next - until + time) << 8) | task;
  if (time < until) until = time;
  for (uint64_t *i = events; i < top; ++i) {
    if ((*i & 0xff) != task) continue;
    after(e, *i) ? move_down(i, e) : move_up(i - 1, e);
    return;
  }
  move_down(--events, e);
}

// Repeatedly execute next scheduled task
void Sched::start_loop() {
  while (true) {
    uint64_t now = *events++;
    uint64_t next = events[0] >> 8;
    until = next - (now >> 8);
    tasks[now & 0xff]();
  }
}

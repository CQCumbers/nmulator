#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <algorithm>

namespace RSP {
  void update();
}

namespace RDP {
  void update();
}

namespace R4300 {
  void update();
  void vi_update();
  void ai_update();
  void pi_update();
  void timer_fire();
}

struct Event {
  uint64_t time;
  void (*func)();
};

struct Scheduler {
  uint8_t n_events;
  uint64_t now;
  Event next;
  Event events[16];
};

static Scheduler s;

bool operator<(const Event& lhs, const Event& rhs) {
  return (int64_t)lhs.time - (int64_t)rhs.time > 0;
}

inline Scheduler *scheduler() {
  return &s;
}

inline void sched(void (*func)(), uint64_t time) {
  Event &e = s.events[s.n_events++];
  e.func = func, e.time = s.now + time;
  std::push_heap(s.events, s.events + s.n_events);
}

inline bool still_top(uint64_t time) {
  uint64_t t = s.events[0].time;
  if (s.n_events == 0 || (int64_t)(s.now + time) - (int64_t)t <= 0) {
    s.now = s.next.time, s.next.time += time; return true;
  } else return false;
}

inline void resched(void (*func)(), uint64_t time) {
  for (Event *e = s.events; e < s.events + s.n_events; ++e) {
    if (e->func == func) {
      e->time = s.now + time;
      std::make_heap(s.events, s.events + s.n_events);
      return;
    }
  }
  sched(func, time);
}

inline void exec_next() {
  s.next = s.events[0];
  std::pop_heap(s.events, s.events + (s.n_events--));
  s.next.func(), s.now = s.next.time;
}

#endif
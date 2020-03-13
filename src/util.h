#ifndef UTIL_H
#define UTIL_H

#include <vector>

struct Event {
  void (*func)() = nullptr;
  uint64_t time = 0;
  bool valid = false;
};

bool operator<(const Event& lhs, const Event& rhs) {
  return (int64_t)lhs.time - (int64_t)rhs.time > 0;
}

static std::vector<Event> events;
Event event = {0};
static uint64_t now = 0;

void sched(void (*func)(), uint64_t time) {
  events.push_back({ func, now + time, true });
  std::push_heap(events.begin(), events.end());
}

inline bool still_top(uint64_t time) {
  bool is_top = (time == 0 || events.empty());
  if (!is_top) {
    int64_t next = events.front().time;
    is_top = (int64_t)(now + time) - next < 0;
  }
  if (is_top) now = event.time, event.time += time;
  return is_top;
}

void cancel(void (*func)()) {
  for (Event &e : events)
    if (e.func == func) e.valid = false;
}

void exec_next() {
  event = events.front();
  std::pop_heap(events.begin(), events.end());
  events.pop_back();
  if (event.valid) event.func(), now = event.time;
}

#endif

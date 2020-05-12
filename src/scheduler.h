#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <algorithm>

namespace Sched {
  struct Event {
    uint64_t time;
    void (*func)();
  };

  bool operator<(const Event& lhs, const Event& rhs) {
    return (int64_t)lhs.time - (int64_t)rhs.time > 0;
  }

  uint8_t n_events;
  uint64_t next;
  int64_t until;
  Event events[16];

  void add(void (*func)(), uint64_t time) {
    Event &e = events[n_events++];
    e.func = func, e.time = next - until + time;
    std::push_heap(events, events + n_events);
    until -= next - events[0].time, next = events[0].time;
  }

  void move(void (*func)(), uint64_t time) {
    for (Event *e = events; e < events + n_events; ++e) {
      if (e->func != func) continue;
      e->time = next - until + time;
      std::make_heap(events, events + n_events);
      until -= next - events[0].time, next = events[0].time;
      return;
    }
    add(func, time);
  }

  void start_loop() {
    while (true) {
      std::pop_heap(events, events + (n_events--));
      Event &now = events[n_events];
      next = events[0].time, until = next - now.time;
      now.func();
    }
  }
}

inline uint32_t pext_low(uint32_t val, uint32_t mask) {
    val &= 0x55555555;
    val = (val ^ (val >> 1)) & 0x33333333;
    val = (val ^ (val >> 2)) & 0x0f0f0f0f;
    val = (val ^ (val >> 4)) & 0x00ff00ff;
    val = (val ^ (val >> 8)) & 0x0000ffff;
    return val & mask;
}

#ifdef _MSC_VER
#  define bswap16(x)  _byteswap_ushort(x)
#  define bswap32(x)  _byteswap_ulong(x)
#  define bswap64(x)  _byteswap_uint64(x)
#else
#  define bswap16(x)  __builtin_bswap16(x)
#  define bswap32(x)  __builtin_bswap32(x)
#  define bswap64(x)  __builtin_bswap64(x)
#endif

#endif

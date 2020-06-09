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

static constexpr void (*funcs[7])() = {
  RSP::update,
  RDP::update,
  R4300::update,
  R4300::vi_update,
  R4300::ai_update,
  R4300::pi_update,
  R4300::timer_fire
};

namespace Sched {
  bool after(uint64_t lhs, uint64_t rhs) {
    return (int64_t)lhs - (int64_t)rhs < 0;
  }

  int64_t until;
  uint64_t ev_array[16];
  uint64_t *events = ev_array + 15;
  uint64_t *const top = events;

  void move_down(uint64_t *ptr, uint64_t event) {
    while (ptr < top && !after(event, ptr[1]))
      ptr[0] = ptr[1], ++ptr;
    ptr[0] = event;
  }

  void print_events() {
    for (uint64_t *e = events; e < top; ++e)
      printf("event: %llx\n", *e);
    printf("---\n");
  }

  void move_up(uint64_t *ptr, uint64_t event) {
    while (ptr + 1 >= events && !after(ptr[0], event))
      ptr[1] = ptr[0], --ptr;
    ptr[1] = event;
  }

  void add(void (*func)(), uint64_t time) {
    for (uint8_t i = 0; i < 7; ++i) {
      if (func != funcs[i]) continue;
      uint64_t next = events[0] >> 8;
      uint64_t e = ((next - until + time) << 8) | i;
      move_down(--events, e);
      until -= next - (events[0] >> 8);
      return;
    }
  }

  void move(void (*func)(), uint64_t time) {
    for (uint64_t *e = events; e < top; ++e) {
      if (funcs[*e & 0xff] != func) continue;
      uint64_t next = events[0] >> 8;
      //*e = ((next - until + time) << 8) | (*e & 0xff);
      uint64_t event = ((next - until + time) << 8) | (*e & 0xff);
      //std::sort(events, top + 1, after);
      if (after(event, *e)) move_down(e, event);
      else move_up(e - 1, event);
      until -= next - (events[0] >> 8); return;
    }
    add(func, time);
  }

  void start_loop() {
    while (true) {
      uint64_t now = *events++;
      uint64_t next = events[0] >> 8;
      until = next - (now >> 8);
      funcs[now & 0xff]();
    }
  }
}

inline uint64_t pext_low(uint64_t val, uint64_t mask) {
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

//#define printf(fmt, ...) (0)

#endif

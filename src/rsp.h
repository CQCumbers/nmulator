#ifndef RSP_H
#define RSP_H

#include <stdio.h>
#include <x86intrin.h>
#include <stdint.h>

namespace RSP {
  uint8_t *dmem = nullptr;
  uint8_t *imem = nullptr;
  constexpr uint32_t addr_mask = 0x1fff;

  template <typename T>
  int64_t read(uint32_t addr) {
    T *ptr = reinterpret_cast<T*>(dmem + (addr & addr_mask));
    switch (sizeof(T)) {
      case 1: return *ptr;
      case 2: return static_cast<T>(__builtin_bswap16(*ptr));
      case 4: return static_cast<T>(__builtin_bswap32(*ptr));
      case 8: return static_cast<T>(__builtin_bswap64(*ptr));
    }
  }

  template <typename T>
  void write(uint32_t addr, T val) {
    T *ptr = reinterpret_cast<T*>(dmem + (addr & addr_mask));
    switch (sizeof(T)) {
      case 1: *ptr = val; return;
      case 2: *ptr = __builtin_bswap16(val); return;
      case 4: *ptr = __builtin_bswap32(val); return;
      case 8: *ptr = __builtin_bswap64(val); return;
    }
  }
  
  uint32_t fetch(uint32_t addr) {
    uint32_t *ptr = reinterpret_cast<uint32_t*>(imem + (addr & 0xfff));
    return __builtin_bswap32(*ptr);
  }

  uint64_t reg_array[0x86] = {0};
  uint32_t pc = 0x0;
  constexpr uint8_t dev_cop0 = 0x20, dev_cop2 = 0x40;

  bool halted() {
    return reg_array[4 + dev_cop0] & 0x1;
  }

  bool broke() {
    return (reg_array[4 + dev_cop0] & 0x42) == 0x42;
  }

  void set_status(uint32_t val) {
    printf("Setting RSP STATUS to %x\n", val);
    reg_array[4 + dev_cop0] &= ~(val & 0x1);
    reg_array[4 + dev_cop0] |= (val & 0x2) >> 1;
    reg_array[4 + dev_cop0] &= ~(_pext_u32(val, 0xaaaaa0) << 5);
    reg_array[4 + dev_cop0] |= _pext_u32(val, 0x1555540) << 5;
  }
}

#endif

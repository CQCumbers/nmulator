class RSP {
  uint8_t *dmem = nullptr;
  uint8_t *imem = nullptr;
  constexpr uint32_t addr_mask = 0xfff;

  template <typename T>
  int64_t read(uint32_t addr) {
    T *ptr = reinterpret_cast<T*>(dmem + (addr & addr_mask));
    switch (sizeof(T)) {
      case 1: return *ptr;
      case 2: return __builtin_bswap16(*ptr);
      case 4: return __builtin_bswap32(*ptr);
    }
    // handle signed vs unsigned or implicit cast does this already?
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
    return *reinterpret_cast<uint32_t*>(imem + addr);
  }
}

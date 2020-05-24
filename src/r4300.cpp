#if defined(_WIN32)
#  include <SDL.h>
#  undef main
#  include <memory.h>
#  include <errhandlingapi.h>
#else
#  include <SDL2/SDL.h>
#  include <sys/mman.h>
#  include <signal.h>
#endif

#include "components.h"

namespace R4300 {
  uint8_t *pages[0x100] = {nullptr};
  uint32_t tlb[0x20][4] = {{0x80000000}};
  constexpr uint32_t addr_mask = 0x1fffffff;
  constexpr uint32_t page_mask = 0x1fffff;

  bool broke = false, moved = false, tlb_miss = false;
  robin_hood::unordered_map<uint32_t, bool> breaks;
  robin_hood::unordered_map<uint32_t, bool> watch_w;
  uint32_t pc = 0xbfc00000; //0xa4000040;
  uint64_t reg_array[0x63] = {0};
  constexpr uint8_t hi = 0x20, lo = 0x21;
  constexpr uint8_t dev_cop0 = 0x22, dev_cop1 = 0x42;

  uint64_t &count = reg_array[9 + dev_cop0];
  uint64_t &compare = reg_array[11 + dev_cop0];
  uint64_t &status = reg_array[12 + dev_cop0];
  uint64_t &cause = reg_array[13 + dev_cop0];

  template <typename T, bool map>
  int64_t read(uint32_t addr);
  template <typename T, bool map>
  void write(uint32_t addr, int64_t val);
  void joy_update(SDL_Event &event);
  void update();

  uint32_t crc32(uint8_t *bytes, uint32_t len);

  /* === MIPS Interface registers === */

  // put into array and read offset without switch?
  constexpr uint32_t mi_version = 0x01010101;
  uint32_t mi_irqs = 0x0, mi_mask = 0x0;

  void set_irqs(uint32_t mask) {
    mi_irqs |= mask;
    if (mi_irqs & mi_mask) {
      cause |= 0x400, cause &= ~0xff;
    }
  }

  void unset_irqs(uint32_t mask) {
    mi_irqs &= ~mask;
    if (!(mi_irqs & mi_mask)) cause &= ~0x400;
  }

  /* === RSP Interface registers === */

  uint64_t *rsp_cop0 = RSP::reg_array + RSP::dev_cop0;

  template <bool write>
  void rsp_dma(uint32_t val) {
    uint32_t skip = (val >> 20) & 0xfff, count = (val >> 12) & 0xff;
    uint32_t len = (val & 0xfff) + 8, m1 = 0x7ffffc, m0 = 0x1ffc;
    uint64_t &r0 = rsp_cop0[0], &r1 = rsp_cop0[1]; r1 &= ~0x7;
    if ((r0 & 0xffc) + (len &= ~0x7) > 0x1000) len = 0x1000 - (r0 & 0xffc);
    printf("[RSP] DMA with count %x, len %x, from %llx to %llx, at pc: %x\n", count + 1, len, r1, r0, RSP::pc);
    for (uint32_t i = 0; i <= count; ++i, r1 += len + skip, r0 += len) {
      uint8_t *ram = pages[0] + (r1 &= m1), *mem = RSP::dmem + (r0 &= m0);
      write ? memcpy(ram, mem, len) : memcpy(mem, ram, len);
    }
    /*if (r0 > 0x1000 && !write) {
      RSP::hash = crc32(RSP::imem, 0x1000);
      RSP::hash <<= 12, RSP::block->valid = false;
      //for (auto &block: RSP::blocks)
      //  block.second.valid = false;
      printf("Set hash to %x\n", RSP::hash);
    }*/
  }

  /* === Video Interface registers === */

  uint32_t vi_status = 0x0, vi_origin = 0x0;
  uint32_t vi_irq = 0x0, vi_line = 0x0;
  uint32_t vi_width = 640, vi_height = 480;
  uint32_t vi_line_progress = 0;
  bool vi_dirty = true;

  SDL_Window *window = nullptr;
  SDL_Renderer *renderer = nullptr;
  SDL_Texture *texture = nullptr;
  uint8_t *pixels = nullptr;

  void vi_update() {
    if (vi_line == vi_irq) set_irqs(0x8);
    vi_line += 0x2, sched(vi_update, 6510);

    if (vi_line < vi_height) return;
    bool interlaced = vi_status & 0x40;
    vi_line = interlaced & ~vi_line;
    if (vi_line == 0x1) return;
    uint8_t format = vi_status & 0x3;
    uint32_t height = vi_height >> !interlaced;

    if (vi_dirty) {
      if (texture) SDL_DestroyTexture(texture);
      texture = SDL_CreateTexture(renderer,
        (format == 2 ? SDL_PIXELFORMAT_RGBA5551 : SDL_PIXELFORMAT_RGBA8888),
        SDL_TEXTUREACCESS_STREAMING, vi_width, height);
      vi_dirty = false;
    }
    if (format == 2) {
      uint16_t *out = reinterpret_cast<uint16_t*>(pixels);
      for (uint32_t i = 0; i < vi_width * height; ++i)
        out[i] = read<uint16_t>(vi_origin + (i << 1));
      SDL_UpdateTexture(texture, nullptr, out, vi_width << 1);
    } else if (format == 3) {
      uint32_t *out = reinterpret_cast<uint32_t*>(pixels);
      for (uint32_t i = 0; i < vi_width * height; ++i)
        out[i] = read<uint32_t>(vi_origin + (i << 2));
      SDL_UpdateTexture(texture, nullptr, out, vi_width << 2);
    }
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);

    for (SDL_Event e; SDL_PollEvent(&e);) {
      if (e.type == SDL_QUIT) exit(0);
      joy_update(e);
    }
  }

  /* === Audio Interface registers === */

  constexpr uint32_t ntsc_clock = 48681812;
  uint32_t ai_status = 0x0, ai_ram = 0x0;
  uint32_t ai_len = 0, ai_start = 0;
  uint32_t ai_rate = 0, ai_bits = 0;
  bool ai_run = false, ai_dirty = false;
  SDL_AudioDeviceID audio_dev;

  void ai_update() {
    if (!ai_run || ai_len == 0) return;
    uint32_t new_len = SDL_GetQueuedAudioSize(audio_dev) >> ai_bits;
    if (new_len <= ai_start || new_len == 0)
      ai_status &= ~0x80000001, ai_start = 0, set_irqs(0x4);
    else sched(ai_update, (new_len - ai_start) << 10);
    ai_len = new_len - ai_start;
  }

  void ai_dma(uint32_t len) {
    if (!ai_run || ai_start != 0) return;
    //printf("[AI] DMA started with len %x\n", len);
    if (ai_dirty) {
      SDL_AudioFormat fmt = (ai_bits ? AUDIO_S16MSB : AUDIO_S8);
      SDL_AudioSpec spec = {
        .freq = static_cast<int>(ntsc_clock / (ai_rate + 1)) << !ai_bits,
        .format = fmt, .channels = 2, .samples = 0x100
      };
      audio_dev = SDL_OpenAudioDevice(nullptr, 0, &spec, nullptr, 0);
      SDL_PauseAudioDevice(audio_dev, 0); ai_dirty = false;
    }
    ai_start = ai_len, ai_len = len & 0x1fff8;
    if (ai_start != 0) ai_status |= 0x80000001;
    SDL_QueueAudio(audio_dev, pages[0] + ai_ram, ai_len);
    sched(ai_update, ai_len << 10);
  }

  /* === Peripheral & Serial Interface registers === */

  uint32_t pi_status = 0x0, pi_len = 0;
  uint32_t pi_ram = 0x0, pi_rom = 0x0;
  bool pi_write = false;
  int8_t stick_x = 0x0, stick_y = 0x0;
  uint16_t buttons = 0x0;
  uint32_t si_ram = 0x0;
  uint64_t eeprom[0x200], mempack[0x1000];

  void pi_update() {
    printf("[PI] DMA from %x to %x, of %x bytes\n", pi_ram, pi_rom, pi_len + 1);
    if (pi_write) memcpy(pages[0] + pi_ram, pages[0] + pi_rom, pi_len + 1);
    else memcpy(pages[0] + pi_rom, pages[0] + pi_ram, pi_len + 1);
    set_irqs(0x10); pi_status &= ~0x1;
  }

  void joy_status(uint32_t channel, uint32_t addr) {
    if (channel == 4) return write<uint32_t>(addr, 0x8000ff);
    if (channel != 0) return write<uint8_t>(addr - 2, 0x83);
    write<uint16_t>(addr, 0x0500);  // standard controller type
    write<uint8_t>(addr + 2, 0x0);  // mempack slot
  }

  void joy_read(uint32_t channel, uint32_t addr) {
    if (channel != 0) return write<uint8_t>(addr - 2, 0x84);
    //printf("[SI] buttons: %x stick: %x\n", buttons, stick_y);
    write<uint16_t>(addr, buttons);
    write<int8_t>(addr + 2, stick_x);
    write<int8_t>(addr + 3, stick_y);
  }

  void joy_update(SDL_Event &event) {
    if (event.type == SDL_KEYDOWN) {
      switch (event.key.keysym.sym) {
        case SDLK_x: buttons |= (1 << 15); break;  // A
        case SDLK_c: buttons |= (1 << 14); break;  // B
        case SDLK_z: buttons |= (1 << 13); break;  // Z
        case SDLK_RETURN: buttons |= (1 << 12); break;  // Start
        case SDLK_k: buttons |= (1 << 11); break;  // D Up
        case SDLK_j: buttons |= (1 << 10); break;  // D Down
        case SDLK_h: buttons |= (1 << 9); break;   // D Left
        case SDLK_l: buttons |= (1 << 8); break;   // D Right
        case SDLK_o: buttons |= (1 << 3); break;   // C Up
        case SDLK_i: buttons |= (1 << 2); break;   // C Down
        case SDLK_u: buttons |= (1 << 1); break;   // C Left
        case SDLK_p: buttons |= (1 << 0); break;   // C Right
        case SDLK_a: buttons |= (1 << 5); break;   // Trigger Left
        case SDLK_s: buttons |= (1 << 4); break;   // Trigger Right
        case SDLK_UP: stick_y = 80; break;         // Stick Up
        case SDLK_DOWN: stick_y = -80; break;      // Stick Down
        case SDLK_LEFT: stick_x = -80; break;      // Stick Left
        case SDLK_RIGHT: stick_x = 80; break ;     // Stick Right
      }
    } else if (event.type == SDL_KEYUP) {
      switch (event.key.keysym.sym) {
        case SDLK_x: buttons &= ~(1 << 15); break;  // A
        case SDLK_c: buttons &= ~(1 << 14); break;  // B
        case SDLK_z: buttons &= ~(1 << 13); break;  // Z
        case SDLK_RETURN: buttons &= ~(1 << 12); break;  // Start
        case SDLK_k: buttons &= ~(1 << 11); break;  // D Up
        case SDLK_j: buttons &= ~(1 << 10); break;  // D Down
        case SDLK_h: buttons &= ~(1 << 9); break;   // D Left
        case SDLK_l: buttons &= ~(1 << 8); break;   // D Right
        case SDLK_o: buttons &= ~(1 << 3); break;   // C Up
        case SDLK_i: buttons &= ~(1 << 2); break;   // C Down
        case SDLK_u: buttons &= ~(1 << 1); break;   // C Left
        case SDLK_p: buttons &= ~(1 << 0); break;   // C Right
        case SDLK_a: buttons &= ~(1 << 5); break;   // Trigger Left
        case SDLK_s: buttons &= ~(1 << 4); break;   // Trigger Right
        case SDLK_UP: stick_y = 0; break;           // Stick Up
        case SDLK_DOWN: stick_y = 0; break;         // Stick Down
        case SDLK_LEFT: stick_x = 0; break;         // Stick Left
        case SDLK_RIGHT: stick_x = 0; break;        // Stick Right
        case SDLK_q: Vulkan::dump_next = true; break;
        case SDLK_w: broke = true; break;
      }
    }
  }

  uint8_t mempack_crc(uint8_t len, uint32_t addr) {
    uint8_t crc = 0;
    for (uint32_t i = 0; i <= len; ++i) {
      for (uint8_t mask = 0x80; mask >= 1; mask >>= 1) {
        uint8_t xor_tap = (crc & 0x80) ? 0x85 : 0x00;
        uint8_t data = read<uint8_t>(addr + i) & mask;
        crc = (crc << 1) | (i < 0x20 && data);
        crc ^= xor_tap;
      }
    }
    return crc;
  }

  void mempack_read(uint8_t len, uint32_t addr) {
    uint16_t mem = read<uint16_t>(addr) & ~0x1f;
    for (uint32_t i = 0; i < len - 1; i += 8) {
      if (mem + i >= 0x8000) break;
      write<uint64_t>(addr + 2 + i, mempack[(mem + i) / 8]);
    }
    write<uint8_t>(addr + len + 1, mempack_crc(addr + 2, len - 1));
  }

  void mempack_write(uint8_t len, uint32_t addr) {
    uint16_t mem = read<uint16_t>(addr) & ~0x1f;
    for (uint32_t i = 0; i < len - 2; i += 8) {
      if (mem + i >= 0x8000) break;
      mempack[(mem + i) / 8] = read<uint64_t>(addr + 2 + i);
    }
    write<uint8_t>(addr + len, mempack_crc(addr + 2, len - 2));
  }

  void eeprom_read(uint8_t len, uint32_t &addr) {
    uint8_t mem = read<uint8_t>(addr);
    for (uint32_t i = 0; i < len; i += 8)
      write<uint64_t>(addr + 1 + i, eeprom[mem + i / 8]);
  }

  void eeprom_write(uint8_t len, uint32_t addr) {
    uint8_t mem = read<uint8_t>(addr);
    for (uint32_t i = 0; i < len - 1; i += 8)
      eeprom[mem + i / 8] = read<uint64_t>(addr + 1 + i);
    write<uint8_t>(addr + len, 0);
  }

  void si_update() {
    uint32_t busy = 0x1fc007ff;
    uint32_t pc = 0x1fc007c0, channel = 0;
    write<uint8_t>(busy, 0x0);
    while (pc < busy) {
      uint8_t t = read<uint8_t>(pc++);
      if (t == 0xfe) return;
      if (t == 0) ++channel, t = 0x80;
      if (t >> 7) continue;
      uint8_t r = read<uint8_t>(pc++);
      switch (t -= 1, read<uint8_t>(pc++)) {
        case 0x00: joy_status(channel, pc); break;
        case 0x01: joy_read(channel, pc); break;
        case 0x02: mempack_read(r, pc); break;
        case 0x03: mempack_write(t, pc); break;
        case 0x04: eeprom_read(r, pc); break;
        case 0x05: eeprom_write(t, pc); break;
        case 0xff: joy_status(channel, pc); break;
      }
      pc += t + r, ++channel;
    }
  }

  /* === Reading and Writing === */

  template <typename T>
  int64_t mmio_read(uint32_t addr) {
    if ((addr & addr_mask & ~0x1fff) == 0x4000000)
      return RSP::read<T, true>(addr);
    switch (addr & addr_mask) {
      default: /*printf("[MMIO] read from %x\n", addr);*/ return 0;
      // RSP Interface
      case 0x4040000: return rsp_cop0[0];
      case 0x4040004: return rsp_cop0[1];
      case 0x4040010: return rsp_cop0[4];
      case 0x404001c: return (rsp_cop0[7] ? 0x1 : rsp_cop0[7]++);
      case 0x4080000: return RSP::pc & 0xffc;
      // RDP Interface
      case 0x4100000: return RDP::pc_start;
      case 0x4100004: return RDP::pc_end;
      case 0x4100008: return RDP::pc;
      case 0x410000c: return RDP::status;
      // MIPS Interface
      case 0x4300004: return mi_version;
      case 0x4300008: return mi_irqs;
      case 0x430000c: return mi_mask;
      // Video Interface
      case 0x4400000: return vi_status;
      case 0x4400004: return vi_origin;
      case 0x4400008: return vi_width;
      case 0x440000c: return vi_irq;
      case 0x4400010: return vi_line;
      case 0x4400018: return vi_height;
      // Audio Interface
      case 0x4500004: return (ai_status & 0x1 ? ai_len : 0);
      case 0x450000c: return ai_status;
      // Peripheral Interface
      case 0x4600000: return pi_ram;
      case 0x4600004: return pi_rom;
      case 0x4600010: return pi_status;
      // RDRAM Interface
      case 0x4700000: return 0xe;      // RI_MODE
      case 0x4700004: return 0x40;     // RI_CONFIG
      case 0x470000c: return 0x14;     // RI_SELECT
      case 0x4700010: return 0x63634;  // RI_REFRESH
      // Serial Interface
      case 0x4800018: return (mi_irqs & 0x2) << 11;
    }
  }

  template <typename T>
  void mmio_write(uint32_t addr, uint32_t val) {
    if ((addr & addr_mask & ~0x1fff) == 0x4000000) {
      printf("RSP MMIO Write of %x to %x, block: %x\n", val, addr & 0x1fff, pc);
      return RSP::write<T, true>(addr, val);
    }
    switch (addr & addr_mask) {
      default: /*printf("[MMIO] write to %x: %x\n", addr, val);*/ return;
      // RSP Interface
      case 0x4040000: rsp_cop0[0] = val & 0x1fff; return;
      case 0x4040004: rsp_cop0[1] = val & 0xffffff; return;
        printf("[RSP] Writing %llx to DMA_SRC\n", rsp_cop0[1]);
        //printf("DMEM 4e0 is %llx %llx\n", RSP::read<uint32_t>(0x4e0), RSP::read<uint32_t>(0x4e4));
        //printf("DMEM 5f8 is %llx %llx\n", RSP::read<uint32_t>(0x5f8), RSP::read<uint32_t>(0x5fc));
        //if (rsp_cop0[1] == 0x3bb790 || rsp_cop0[1] == 0x3aa538*)
        //  printf("First word: %x\n", *(uint32_t*)(pages[0] + rsp_cop0[1]));
        return;
      case 0x4040008: rsp_dma<false>(val); return;
      case 0x404000c: rsp_dma<true>(val); return;
      case 0x4040010: RSP::set_status(val); return;
      case 0x404001c: rsp_cop0[7] = 0x0; return;
      case 0x4080000: RSP::pc = val & 0xffc; return;
      // RDP Interface
      case 0x4100000:
        RDP::pc_start = val & addr_mask;
        RDP::pc = RDP::pc_start; return;
      case 0x4100004:
        Sched::add(RDP::update, 0);
        RDP::pc_end = val & addr_mask; return;
      case 0x410000c:
        RDP::status &= ~pext_low(val >> 0, 0x7);
        RDP::status |= pext_low(val >> 1, 0x7); return;
      // MIPS Interface
      case 0x4300000:
        if (val & 0x800) unset_irqs(0x20); return;
      case 0x430000c:
        mi_mask &= ~pext_low(val >> 0, 0x3f);
        mi_mask |= pext_low(val >> 1, 0x3f);
        if (mi_irqs & mi_mask) {
          cause |= 0x400, cause &= ~0xff;
        } else cause &= ~0x400; return;
      // Video Interface
      case 0x4400000: 
        if (val == vi_status) return;
        vi_status = val, vi_dirty = true; return;
      case 0x4400004: vi_origin = val & 0xffffff; return;
      case 0x4400008:
        if ((val & 0xfff) == vi_width) return;
        vi_width = val & 0xfff, vi_dirty = true; return;
      case 0x440000c: vi_irq = val & 0x3ff; return;
      case 0x4400010: unset_irqs(0x8); return;
      case 0x4400018:
        if ((val & 0x3ff) == vi_height) return;
        vi_height = val & 0x3ff, vi_dirty = true; return;
      // Audio Interface
      case 0x4500000: ai_ram = val & 0xfffff8; return;
      case 0x4500004: ai_dma(val); return;
      case 0x4500008: ai_run = val & 0x1; return;
      case 0x450000c: unset_irqs(0x4); return;
      case 0x4500010:
        if ((val & 0xfff) == ai_rate) return;
        ai_rate = val & 0xfff, ai_dirty = true; return;
      case 0x4500014:
        if (((val >> 3) & 0x1) == ai_bits) return;
        ai_bits = (val >> 3) & 0x1, ai_dirty = true; return;
      // Peripheral Interface
      case 0x4600000: pi_ram = val & 0xffffff; return;
      case 0x4600004: pi_rom = val & addr_mask; return;
      case 0x4600008:
        if (pi_status & 0x1) return;
        pi_len = val, pi_write = false;
        sched(pi_update, 0/*pi_len / 2*/);
        pi_status |= 0x1; return;
      case 0x460000c:
        if (pi_status & 0x1) return;
        pi_len = val, pi_write = true;
        sched(pi_update, 0/*pi_len / 2*/);
        pi_status |= 0x1; return;
      case 0x4600010: unset_irqs(0x10); return;
      // Serial Interface
      case 0x4800000: si_ram = val & 0xffffff; return;
      case 0x4800004: si_update();
        printf("a43bf: %x\n", pages[0][0xa43bf]);
        memcpy(pages[0] + si_ram, pages[0xfe] + 0x7c0, 0x40);
        printf("SI writing to %x, a43bf: %x\n", si_ram, pages[0][0xa43bf]);
        set_irqs(0x2); return;
      case 0x4800010:
        memcpy(pages[0xfe] + 0x7c0, pages[0] + si_ram, 0x40);
        set_irqs(0x2); return;
      case 0x4800018: unset_irqs(0x2); return;
    }
  }

  uint32_t tlb_map(uint32_t addr) {
    // assumes all global, valid, and dirty
    printf("COP0 2: %llx, 3: %llx\n", reg_array[2 + dev_cop0], reg_array[3 + dev_cop0]);
    printf("Checking TLB for addr: %x\n", addr);
    for (uint32_t i = 0; i < 32; ++i) {
      uint32_t mask = ~(tlb[i][0] | 0x1fff);
      printf("TLB[%x] with mask: %x, vaddr: %x, paddr0: %x, paddr1: %x\n",
        i, mask, tlb[i][1] & mask, (tlb[i][2] << 6) & mask, (tlb[i][3] << 6) & mask);
      if ((addr & mask) == (tlb[i][1] & mask)) {
        mask = (mask >> 1) | 0x80000000;
        bool odd = (addr & -mask) != 0;
        uint32_t page = tlb[i][2 + odd] << 6;
        return (page & mask) | (addr & ~mask);
      }
    }
    tlb_miss = true;
    // load COP0 context, bad_vaddr, entry_hi
    uint64_t &ctx = reg_array[4 + dev_cop0];
    ctx &= ~0x7ffff0, ctx |= (addr >> 9) & 0x7ffff0;
    reg_array[8 + dev_cop0] = addr;
    reg_array[10 + dev_cop0] = addr & 0xffffe000;
    printf("TLB miss for addr: %x\n", addr);
    return addr;
  }

  template <typename T, bool map>
  int64_t read(uint32_t addr) {
    //printf("Reading from %x\n", addr);
    if (map && addr >> 30 != 0x2) addr = tlb_map(addr);
    uint8_t *page = pages[(addr >> 21) & 0xff];
    if ((addr & addr_mask) == 0x1fc007e4) {
      printf("Hacky CIC emulation\n");
      pages[0][0x1fc007ff] = 0x80;  // Hacky CIC emulation
    }
    if (!page) return mmio_read<T>(addr);
    T *ptr = reinterpret_cast<T*>(page + (addr & page_mask));
    switch (sizeof(T)) {
      case 1: return *ptr;
      case 2: return static_cast<T>(bswap16(*ptr));
      case 4: return static_cast<T>(bswap32(*ptr));
      case 8: return static_cast<T>(bswap64(*ptr));
    }
  }

  template <typename T, bool map>
  void write(uint32_t addr, int64_t val) {
    if (((addr >> 4) & 0xffff) == 0xa442)
      printf("Writing %llx to %x, block: %x\n", val, addr, pc);
    if (map && addr >> 30 != 0x2) addr = tlb_map(addr);
    //broke |= watch_w[addr];
    uint8_t *page = pages[(addr >> 21) & 0xff];
    if (!page) return mmio_write<T>(addr, val);
    T *ptr = reinterpret_cast<T*>(page + (addr & page_mask));
    switch (sizeof(T)) {
      case 1: *ptr = val; return;
      case 2: *ptr = bswap16(val); return;
      case 4: *ptr = bswap32(val); return;
      case 8: *ptr = bswap64(val); return;
    }
  }

  const uint32_t hpage_mask = addr_mask & ~0xfff;
  uint32_t last_pg = -1;  // reset to -1 before calling compile

  uint32_t fetch(uint32_t addr) {
    uint32_t pg = pc & hpage_mask;
    if (pg != last_pg) protect(last_pg = pg);
    return read<uint32_t, true>(addr);
  }

  /* === Actual CPU Functions === */

  robin_hood::unordered_node_map<uint32_t, Block> blocks;
  robin_hood::unordered_map<uint32_t, std::vector<uint32_t>> prot_pages;
  bool modified = false; Block *block = &empty;

  bool is_break(uint32_t pc) {
    if (!moved) { moved = true; return false; }
    return (broke = broke || breaks[pc]);
  }

  void mtc0(uint32_t reg, uint32_t val) {
    regs[reg + cfg.cop0] = val;
    if (reg != 9 && reg != 11) return;
    if (reg == 11) cause &= ~0x8000;
    uint32_t cycles = (compare - count) * 2;
    Sched::add(timer_fire, cycles);
  }

  void emit_link(ReadPtr &out) {
    using namespace asmjit;

    JitRuntime runtime;
    CodeHolder code;
    code.init(rt.codeInfo());
    x86::Assembler as(&code);

    as.mov(x86::rax, reinterpret_cast<uint64_t>(block));
    as.cmp(x86::rdi, x86::qword_ptr(x86::rax, npc)); as.jne(exit_label);

    rt.add(&out, &code);
  }

  JitConfig cfg = {
    .read = {
      read<int8_t>, read<int16_t>,
      read<int32_t>, read<int64_t>
    },
    .write = {
      write<int8_t>, write<int16_t>,
      write<int32_t>, write<int64_t>
    },
    .fetch = fetch, .link = link,
    .pc = 0xbfc00000,
    .cop0 = 0x22, .cop1 = 0x42,
    .regs = regs, .tlb = tlb,
  };


  void timer_fire() {
    cause |= 0x8000, cause &= ~0xff;
  }

  void timer_update() {
  }

  inline void irqs_update() {
    uint8_t ip = ((cause & status) >> 8) & 0xff;
    if ((status & 0x3) != 0x1 || ip == 0x0) return;

    // if interrupt enabled and triggered, set pc to handler address
    printf("Jumping to interrupt instead of %x: %x\n", pc, fetch(pc));
    printf("IP: %x MI: %x MASK: %x VI_INTR %x\n", ip, mi_irqs, mi_mask, vi_irq);
    reg_array[14 + dev_cop0] = pc; pc = 0x80000180; status |= 0x2;
  }

#if defined(_WIN32)
  uint8_t *alloc_pages(uint32_t size) {
    return reinterpret_cast<uint8_t*>(VirtualAlloc(
      nullptr, size, MEM_COMMIT, PAGE_READWRITE
    ));
  }

  void protect(uint32_t hpage) {
    DWORD old;
    if (prot_pages[hpage].empty()) {
      VirtualProtect(pages[0] + hpage, 0x1000, PAGE_READONLY, &old);
    }
    prot_pages[hpage].push_back(pc & addr_mask);
  }

  void unprotect(uint32_t hpage) {
    DWORD old;
    for (uint32_t addr : prot_pages[hpage]) blocks[addr].valid = false;
    VirtualProtect(pages[0] + hpage, 0x1000, PAGE_READWRITE, &old);
    prot_pages[hpage].clear(); modified = true;
  }
  
  LONG WINAPI handle_fault(_EXCEPTION_POINTERS *info) {
    DWORD sig = info->ExceptionRecord->ExceptionCode;
    if (sig != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    uint8_t *addr = (uint8_t*)info->ExceptionRecord->ExceptionInformation[1];
    int64_t hpage = addr - pages[0];
    if (!(0 <= hpage && hpage <= addr_mask)) exit(1);
    unprotect(hpage & hpage_mask); return EXCEPTION_CONTINUE_EXECUTION;
  }

  void setup_fault_handler() {
    AddVectoredExceptionHandler(true, handle_fault);
  }
#else
  uint8_t *alloc_pages(uint32_t size) {
    return reinterpret_cast<uint8_t*>(mmap(
      nullptr, size, PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_SHARED, 0, 0
    ));
  }

  void protect(uint32_t hpage) {
    if (prot_pages[hpage].empty())
      mprotect(pages[0] + hpage, 0x1000, PROT_READ);
    prot_pages[hpage].push_back(pc & addr_mask);
  }

  void unprotect(uint32_t hpage) {
    for (uint32_t addr : prot_pages[hpage]) blocks[addr].valid = false;
    mprotect(pages[0] + hpage, 0x1000, PROT_READ | PROT_WRITE);
    prot_pages[hpage].clear(); modified = true;
  }

  void handle_fault(int sig, siginfo_t *info, void *raw_ctx) {
    if (sig != SIGBUS && sig != SIGSEGV) return;
    int64_t hpage = (uint8_t*)info->si_addr - pages[0];
    if (!(0 <= hpage && hpage <= addr_mask)) exit(1);
    unprotect(hpage & hpage_mask);
  }

  void setup_fault_handler() {
    struct sigaction act;
    memset(&act, 0, sizeof(act));
    act.sa_sigaction = handle_fault;
    act.sa_flags = SA_RESTART | SA_SIGINFO;
    sigemptyset(&act.sa_mask);
    sigaction(SIGBUS, &act, nullptr);
  }
#endif

  void update() {
    uint32_t cycles = 0;
    while (still_top(cycles)) {
      bool run = block->valid;
      if (run) {
        pc = block->code();
        if (broke) {
          Debugger::update();
          blocks.clear();
          block->next_pc = 0;
        }
        if (block->next_pc != pc)
          block->next_pc = pc, block->next = &blocks[pc & addr_mask];
        block = block->next;
      }

      if (!block->valid) {
        CodeHolder code; 
        code.init(runtime.codeInfo());
        MipsJit<Device::r4300> jit(code);

        block->cycles = jit.jit_block();
        runtime.add(&block->code, &code);
        block->valid = true;
      }
      cycles = run ? block->cycles : 0;
    }
    sched(update, cycles);
  }

  uint32_t crc32(uint8_t *bytes, uint32_t len) {
    uint32_t crc = 0xffffffff;
    for (uint32_t i = 0; i < len; ++i) {
      crc ^= bytes[i];
      for (uint32_t j = 0; j < 8; ++j) {
         uint32_t mask = -(crc & 0x1);
         crc = (crc >> 1) ^ (0xedb88320 & mask);
      }
    }
    return ~crc;
  }

  void init(FILE *file) {
    // allocate memory, setup change detection
    pages[0] = alloc_pages(0x20000000);
    setup_fault_handler();
    // specially handle SP/DP, DP/MI, VI/AI, PI/RI, and SI ranges
    for (uint32_t i = 0x1; i < 0x100; ++i) {
      if (i < 0x20 || i > 0x24) pages[i] = pages[0] + i * (page_mask + 1);
    }

    // read ROM file into memory
    fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    fseek(file, 0, SEEK_SET);
    fread(pages[0] + 0x10000000, 1, fsize, file);
    memcpy(pages[0] + 0x04000000, pages[0] + 0x10000000, 0x40000);

    // read PIF boot rom, setup ram based on CIC
    FILE *file2 = fopen("../cen64/pifdata.bin", "r");
    fread(pages[0] + 0x1fc00000, 1, 0x7c0, file2);
    fclose(file2);

    write<uint32_t>(0xa0000318, 0x800000);  // 8MB RDRAM
    switch (crc32(pages[0] + 0x04000040, 0xfc0)) {
      case 0x6170a4a1: write<uint32_t>(0xbfc007e4, 0x43f3f); break;  // 6101
      case 0x90bb6cb5: write<uint32_t>(0xbfc007e4, 0x3f3f); break;   // 6102
      case 0x0b050ee0: write<uint32_t>(0xbfc007e4, 0x783f); break;   // 6103
      case 0x98bc2c86: write<uint32_t>(0xbfc007e4, 0x913f); break;   // 6105
      case 0xacc8580a: write<uint32_t>(0xbfc007e4, 0x853f); break;   // 6106
      default: printf("No compatible CIC chip found\n"), exit(1);
    }

    // setup VI
    pixels = alloc_pages(0x200000);
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO);
    SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");
    SDL_CreateWindowAndRenderer(640, 480, SDL_WINDOW_ALLOW_HIGHDPI, &window, &renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);
    Vulkan::init();

    // setup RSP
    RSP::mem = pages[0] + 0x04000000;
    rsp_cop0[4] = 0x1, rsp_cop0[11] = 0x80;
  }
}

#endif

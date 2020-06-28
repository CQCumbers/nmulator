#ifdef _WIN32
#  include <SDL.h>
#  undef main
#  define _AMD64_
#  include <memoryapi.h>
#  include <errhandlingapi.h>
#  include <io.h>
#else
#  include <SDL2/SDL.h>
#  include <sys/mman.h>
#  include <signal.h>
#  include <unistd.h>
#endif

#include <vector>
#include "robin_hood.h"

#include <fcntl.h>
#include <errno.h>
#include <nmmintrin.h>
#include "nmulator.h"

namespace R4300 { uint8_t *ram; }
static uint32_t pages[0x100000];
static uint32_t tlb[0x20][4];

static uint32_t pc = 0xbfc00000;
static uint64_t regs[99 + 52];
static uint64_t *const cop0 = regs + 34;
static uint64_t *const cop1 = regs + 66;

/* === MIPS Interface registers === */

static uint32_t mi_irqs, mi_mask;
static const uint32_t mi_version = 0x01010101;

// raise MI pending interrupt bits
void R4300::set_irqs(uint32_t mask) {
  if ((mi_irqs |= mask) & mi_mask)
    cop0[13] |= 0x400, cop0[13] &= ~0xff;
}

// lower MI pending interrupt bits
void R4300::unset_irqs(uint32_t mask) {
  if (!((mi_irqs &= ~mask) & mi_mask))
    cop0[13] &= ~0x400;
}

/* === Video Interface registers === */

static uint32_t vi_status, vi_origin;
static uint32_t vi_width, vi_height;
static uint32_t vi_irq_line, vi_line;
static bool vi_dirty;

static const char *title;
static SDL_Window *window;
static SDL_Renderer *renderer;
static SDL_Texture *texture;
static void joy_update(SDL_Event event);

// setup SDL renderer for display
static void vi_init() {
  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO);
  SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");
  const uint32_t flags = SDL_WINDOW_ALLOW_HIGHDPI;
  SDL_CreateWindowAndRenderer(640, 480, flags, &window, &renderer);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
  SDL_RenderClear(renderer), SDL_SetWindowTitle(window, title);
}

// convert rgba5551 to bgra8888, 16 byte aligned len
static void convert16(uint8_t *dst, const uint8_t *src, uint32_t len) {
  __m128i imm0xf8 = _mm_set1_epi16(0xf8);
  __m128i shuffle = _mm_set_epi8(
    14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
  for (uint32_t i = 0; i < len; i += 16, src += 16, dst += 32) {
    __m128i data = _mm_loadu_si128((__m128i*)src);
    __m128i pack = _mm_shuffle_epi8(data, shuffle);
    __m128i r = _mm_and_si128(_mm_srli_epi16(pack, 8), imm0xf8);
    __m128i g = _mm_and_si128(_mm_srli_epi16(pack, 3), imm0xf8);
    __m128i b = _mm_and_si128(_mm_slli_epi16(pack, 2), imm0xf8);
    __m128i gb = _mm_or_si128(_mm_slli_epi16(g, 8), b);
    _mm_storeu_si128((__m128i*)dst, _mm_unpacklo_epi16(gb, r));
    _mm_storeu_si128((__m128i*)(dst + 16), _mm_unpackhi_epi16(gb, r));
  }
}

// convert rgba8888 to bgra8888, 16 byte aligned len
static void convert32(uint8_t *dst, const uint8_t *src, uint32_t len) {
  __m128i shuffle = _mm_set_epi8(
    15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2);
  for (uint32_t i = 0; i < len; i += 16, src += 16, dst += 16) {
    __m128i data = _mm_loadu_si128((__m128i*)src);
    __m128i pack = _mm_shuffle_epi8(data, shuffle);
    _mm_storeu_si128((__m128i*)dst, pack);
  }
}

// update VI regs and display framebuffer
void R4300::vi_update() {
  if (vi_line == vi_irq_line) set_irqs(0x8);
  vi_line += 0x2, Sched::add(TASK_VI, 6510);
  if (vi_line < vi_height) return;

  // reset scanline to 0 (1 on odd interlaced frames)
  bool interlaced = vi_status & 0x40;
  vi_line = interlaced & ~vi_line;
  if (vi_line == 0x1) return;
  uint8_t format = vi_status & 0x3;
  uint32_t height = vi_height >> !interlaced;

  if (vi_dirty) {
    if (texture) SDL_DestroyTexture(texture);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
      SDL_TEXTUREACCESS_STREAMING, vi_width, height);
    vi_dirty = false;
  }

  bool fmt16 = (format == 2);
  uint8_t *pixels; int pitch, len = vi_width * height;
  SDL_LockTexture(texture, NULL, (void**)&pixels, &pitch);
  if (fmt16) convert16(pixels, ram + vi_origin, len * 2);
  else convert32(pixels, ram + vi_origin, len * 4);
  SDL_UnlockTexture(texture);

  SDL_RenderCopy(renderer, texture, NULL, NULL);
  SDL_RenderPresent(renderer);

  for (SDL_Event e; SDL_PollEvent(&e);) {
    if (e.type == SDL_QUIT) exit(0);
    joy_update(e);
  }
}

/* === Audio Interface registers === */

static uint32_t ai_status, ai_rate;
static uint32_t ai_ram, ai_len;
static uint32_t ai_start, ai_end;

static bool ai_run, ai_dirty, ai_16bit;
static const uint32_t ntsc_clock = 48681812;
static const uint32_t audio_delay = 2048;
static SDL_AudioDeviceID audio_dev;

void R4300::ai_update() {
  // calculate samples remaining in play buffer
  uint32_t prev_len = ai_len;
  ai_len = SDL_GetQueuedAudioSize(audio_dev) >> ai_16bit;
  ai_len = ai_len > audio_delay ? ai_len - audio_delay : 0;
  if (ai_len > 0) return Sched::move(TASK_AI, 1);
  if (prev_len > 0) set_irqs(0x4);

  // play samples at saved address, set param buffer empty
  if (~ai_status & 0x80000001) return;
  SDL_QueueAudio(audio_dev, ram + ai_start, ai_end);
  ai_len = ai_end >> ai_16bit, ai_status &= ~0x80000001;
  Sched::move(TASK_AI, ai_len << 12);
}

// configure new SDL audio device
static void ai_config() {
  SDL_AudioFormat fmt = (ai_16bit ? AUDIO_S16MSB : AUDIO_S8);
  SDL_AudioSpec spec = {
    .freq = (int)(ntsc_clock / (ai_rate + 1)) << !ai_16bit,
    .format = fmt, .channels = 2, .samples = 256,
  };
  audio_dev = SDL_OpenAudioDevice(NULL, 0, &spec, NULL, 0);
  SDL_PauseAudioDevice(audio_dev, 0);
}

// save address of samples, set param buffer full
static void ai_dma(uint32_t len) {
  if (!ai_run || len == 0) return;
  if (ai_dirty) ai_config(), ai_dirty = false;
  ai_start = ai_ram, ai_end = len & 0x1fff8;
  ai_status |= 0x80000001, R4300::ai_update();
}

/* === Peripheral Interface registers === */

static uint32_t pi_status, pi_len;
static uint32_t pi_ram, pi_rom;
static bool pi_to_rom;

// DMA bytes from cartridge to RDRAM, or vice-versa
void R4300::pi_update() {
  uint8_t *src = ram + (pi_to_rom ? pi_ram : pi_rom);
  uint8_t *dst = ram + (pi_to_rom ? pi_rom : pi_ram);
  memcpy(dst, src, pi_len), set_irqs(0x10), pi_status &= ~0x1;
}

// hack to pass PIF bootrom
void R4300::cic_update() {
  ram[0x1fc007ff] |= 0x80;
}

/* === Serial Interface registers === */

static uint64_t *mempak;
static uint64_t *eeprom;

static uint16_t buttons;
static uint8_t joy_x, joy_y;
static uint32_t si_ram;

static const uint8_t crc8_table[256] = {
  0x00, 0x85, 0x8f, 0x0a, 0x9b, 0x1e, 0x14, 0x91,
  0xb3, 0x36, 0x3c, 0xb9, 0x28, 0xad, 0xa7, 0x22,
  0xe3, 0x66, 0x6c, 0xe9, 0x78, 0xfd, 0xf7, 0x72,
  0x50, 0xd5, 0xdf, 0x5a, 0xcb, 0x4e, 0x44, 0xc1,
  0x43, 0xc6, 0xcc, 0x49, 0xd8, 0x5d, 0x57, 0xd2,
  0xf0, 0x75, 0x7f, 0xfa, 0x6b, 0xee, 0xe4, 0x61,
  0xa0, 0x25, 0x2f, 0xaa, 0x3b, 0xbe, 0xb4, 0x31,
  0x13, 0x96, 0x9c, 0x19, 0x88, 0x0d, 0x07, 0x82,
  0x86, 0x03, 0x09, 0x8c, 0x1d, 0x98, 0x92, 0x17,
  0x35, 0xb0, 0xba, 0x3f, 0xae, 0x2b, 0x21, 0xa4,
  0x65, 0xe0, 0xea, 0x6f, 0xfe, 0x7b, 0x71, 0xf4,
  0xd6, 0x53, 0x59, 0xdc, 0x4d, 0xc8, 0xc2, 0x47,
  0xc5, 0x40, 0x4a, 0xcf, 0x5e, 0xdb, 0xd1, 0x54,
  0x76, 0xf3, 0xf9, 0x7c, 0xed, 0x68, 0x62, 0xe7,
  0x26, 0xa3, 0xa9, 0x2c, 0xbd, 0x38, 0x32, 0xb7,
  0x95, 0x10, 0x1a, 0x9f, 0x0e, 0x8b, 0x81, 0x04,
  0x89, 0x0c, 0x06, 0x83, 0x12, 0x97, 0x9d, 0x18,
  0x3a, 0xbf, 0xb5, 0x30, 0xa1, 0x24, 0x2e, 0xab,
  0x6a, 0xef, 0xe5, 0x60, 0xf1, 0x74, 0x7e, 0xfb,
  0xd9, 0x5c, 0x56, 0xd3, 0x42, 0xc7, 0xcd, 0x48,
  0xca, 0x4f, 0x45, 0xc0, 0x51, 0xd4, 0xde, 0x5b,
  0x79, 0xfc, 0xf6, 0x73, 0xe2, 0x67, 0x6d, 0xe8,
  0x29, 0xac, 0xa6, 0x23, 0xb2, 0x37, 0x3d, 0xb8,
  0x9a, 0x1f, 0x15, 0x90, 0x01, 0x84, 0x8e, 0x0b,
  0x0f, 0x8a, 0x80, 0x05, 0x94, 0x11, 0x1b, 0x9e,
  0xbc, 0x39, 0x33, 0xb6, 0x27, 0xa2, 0xa8, 0x2d,
  0xec, 0x69, 0x63, 0xe6, 0x77, 0xf2, 0xf8, 0x7d,
  0x5f, 0xda, 0xd0, 0x55, 0xc4, 0x41, 0x4b, 0xce,
  0x4c, 0xc9, 0xc3, 0x46, 0xd7, 0x52, 0x58, 0xdd,
  0xff, 0x7a, 0x70, 0xf5, 0x64, 0xe1, 0xeb, 0x6e,
  0xaf, 0x2a, 0x20, 0xa5, 0x34, 0xb1, 0xbb, 0x3e,
  0x1c, 0x99, 0x93, 0x16, 0x87, 0x02, 0x08, 0x8d
};

// calculate mempak data crc8, P = 0x85
static uint8_t crc8(const uint8_t *msg, uint32_t len) {
  uint8_t crc = 0;
  for (uint32_t i = 0; i < len; ++i)
    crc = crc8_table[crc ^ msg[i]];
  return crc;
}

static const uint8_t mempak_blank[272] = {
  0x81, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
  0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
  0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
  0xff, 0xff, 0xff, 0xff, 0x05, 0x1a, 0x5f, 0x13,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0x01, 0xff, 0x66, 0x25, 0x99, 0xcd,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x05, 0x1a, 0x5f, 0x13,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0x01, 0xff, 0x66, 0x25, 0x99, 0xcd,
  0xff, 0xff, 0xff, 0xff, 0x05, 0x1a, 0x5f, 0x13,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0x01, 0xff, 0x66, 0x25, 0x99, 0xcd,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x05, 0x1a, 0x5f, 0x13,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0x01, 0xff, 0x66, 0x25, 0x99, 0xcd,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x71, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03,
  0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03
};

#ifdef _WIN32

static void mempak_init(const char *name) {
  // open or create mempak file
  int mode = _S_IREAD | _S_IWRITE, exists;
  int file = _open(name, _O_RDWR | _O_CREAT | _O_EXCL, mode);
  if ((exists = file < 0 && errno == EEXIST)) file = _open(name, _O_RDWR);
  if (file < 0) printf("Can't open %s\n", name), exit(1);

  // map mempak file into memory
  if (_chsize(file, 0x8000) < 0) printf("Can't modify %s\n", name), exit(1);
  mempak = (uint64_t*)MapViewOfFile((HANDLE)_get_osfhandle(file),
    FILE_MAP_ALL_ACCESS, 0, 0, 0x8000), _close(file);
  if (!exists) memcpy(mempak, mempak_blank, 272);
}

static void eeprom_init(char *name) {
  // change extension to .eep, open file
  strcpy(name + strlen(name) - 4, ".eep");
  int file = _open(name, _O_RDWR | _O_CREAT, _S_IREAD | _S_IWRITE);
  if (file < 0) printf("Can't open %s\n", name), exit(1);

  // map 4kbit eeprom file into memory
  if (_chsize(file, 0x200) < 0) printf("Can't modify %s\n", name), exit(1);
  mempak = (uint64_t*)MapViewOfFile((HANDLE)_get_osfhandle(file),
    FILE_MAP_ALL_ACCESS, 0, 0, 0x200), _close(file);
}

#else

static void mempak_init(const char *name) {
  // open or create mempak file
  int file = open(name, O_RDWR | O_CREAT | O_EXCL, 0644), exists;
  if ((exists = file < 0 && errno == EEXIST)) file = open(name, O_RDWR);
  if (file < 0) printf("Can't open %s\n", name), exit(1);

  // map mempak file into memory
  if (ftruncate(file, 0x8000) < 0) printf("Can't modify %s\n", name), exit(1);
  mempak = (uint64_t*)mmap(NULL, 0x8000, PROT_READ | PROT_WRITE,
    MAP_SHARED, file, 0), close(file);
  if (!exists) memcpy(mempak, mempak_blank, 272);
}

static void eeprom_init(char *name) {
  // change extension to .eep, open file
  strcpy(name + strlen(name) - 4, ".eep");
  int file = open(name, O_RDWR | O_CREAT, 0644);
  if (file < 0) printf("Can't open %s\n", name), exit(1);

  // map 4kbit eeprom file into memory
  if (ftruncate(file, 0x200) < 0) printf("Can't modify %s\n", name), exit(1);
  eeprom = (uint64_t*)mmap(NULL, 0x200, PROT_READ | PROT_WRITE,
    MAP_SHARED, file, 0), close(file);
}

#endif

// mempak to pifram, len = read length + 1 crc byte
static void mempak_read(uint32_t pc, uint8_t len) {
  uint32_t mem = read16(R4300::ram + pc) & ~0x1f;
  memcpy(R4300::ram + pc + 2, mempak + mem / 8, --len);
  R4300::ram[pc + 2 + len] = crc8(R4300::ram + pc + 2, len);
}

// pifram to mempak, len = 2 address bytes + write data length
static void mempak_write(uint32_t pc, uint8_t len) {
  uint32_t mem = read16(R4300::ram + pc) & ~0x1f;
  if (mem == 0x8000) return;  // ignore enable/disable
  memcpy(mempak + mem / 8, R4300::ram + pc + 2, len - 2);
  R4300::ram[pc + len] = crc8(R4300::ram + pc + 2, len - 2);
}

// eeprom to pifram, len = read length
static void eeprom_read(uint32_t pc, uint8_t len) {
  if (!eeprom) eeprom_init(strdup(title));
  uint64_t *src = eeprom + R4300::ram[pc];
  memcpy(R4300::ram + pc + 1, src, len);
}

// pifram to eeprom, len = read length
static void eeprom_write(uint32_t pc, uint8_t len) {
  if (!eeprom) eeprom_init(strdup(title));
  uint64_t *dst = eeprom + R4300::ram[pc];
  memcpy(dst, R4300::ram + pc + 1, len);
}

// read type of connected controllers
static void joy_status(uint32_t pc, uint32_t channel) {
  if (channel == 4) return write32(R4300::ram + pc, 0x8000ff);
  if (channel != 0) { R4300::ram[pc - 2] = 0x83; return; }
  write16(R4300::ram + pc, 0x500), R4300::ram[pc + 2] = 0x1;
}

// read inputs from connected controllers
static void joy_read(uint32_t pc, uint32_t channel) {
  if (channel != 0) { R4300::ram[pc - 2] = 0x84; return; }
  write16(R4300::ram + pc, buttons);
  R4300::ram[pc + 2] = joy_x, R4300::ram[pc + 3] = joy_y;
}

// interpret SI commands in pifram
static void si_update() {
  const uint32_t busy = 0x1fc007ff;
  uint32_t channel = R4300::ram[busy] = 0;
  for (uint32_t pc = 0x1fc007c0; pc < busy;) {
    uint8_t t = R4300::ram[pc++];
    if (t == 0xfe) return;
    if (t == 0) ++channel, t = 0x80;
    if (t >> 7) continue;
    uint8_t r = R4300::ram[pc++];
    switch (t -= 1, R4300::ram[pc++]) {
      case 0x00: joy_status(pc, channel); break;
      case 0x01: joy_read(pc, channel); break;
      case 0x02: mempak_read(pc, r); break;
      case 0x03: mempak_write(pc, t); break;
      case 0x04: eeprom_read(pc, r); break;
      case 0x05: eeprom_write(pc, t); break;
      case 0xff: joy_status(pc, channel); break;
    }
    pc += t + r, ++channel;
  }
}

// read controller inputs from SDL
static void joy_update(SDL_Event event) {
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
      case SDLK_UP: joy_y = 80; break;           // Stick Up
      case SDLK_DOWN: joy_y = -80; break;        // Stick Down
      case SDLK_LEFT: joy_x = -80; break;        // Stick Left
      case SDLK_RIGHT: joy_x = 80; break ;       // Stick Right
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
      case SDLK_UP: joy_y = 0; break;             // Stick Up
      case SDLK_DOWN: joy_y = 0; break;           // Stick Down
      case SDLK_LEFT: joy_x = 0; break;           // Stick Left
      case SDLK_RIGHT: joy_x = 0; break;          // Stick Right
      case SDLK_q: RDP::dump = true; break;
    }
  }
}

/* === R4300 memory access === */

// handle MMIO read from physical address
static int64_t read(uint32_t addr) {
  switch (addr & R4300::mask) {
    default: /*printf("[MMIO] read from %x\n", addr);*/ return 0;
    // RSP Interface
    case 0x4040000: return RSP::cop0[0];
    case 0x4040004: return RSP::cop0[1];
    case 0x4040010: return RSP::cop0[4];
    case 0x404001c: return (RSP::cop0[7] ? 0x1 : RSP::cop0[7]++);
    case 0x4080000: return RSP::pc & 0xffc;
    // RDP Interface
    case 0x4100000: return RSP::cop0[8];
    case 0x4100004: return RSP::cop0[9];
    case 0x4100008: return RSP::cop0[10];
    case 0x410000c: return RSP::cop0[11];
    // MIPS Interface
    case 0x4300004: return mi_version;
    case 0x4300008: return mi_irqs;
    case 0x430000c: return mi_mask;
    // Video Interface
    case 0x4400000: return vi_status;
    case 0x4400004: return vi_origin;
    case 0x4400008: return vi_width;
    case 0x440000c: return vi_irq_line;
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

// handle MMIO write to physical address
static void write(uint32_t addr, uint64_t val) {
  uint8_t *ram = R4300::ram;
  switch (addr & R4300::mask) {
    default: /*printf("[MMIO] write to %x: %x\n", addr, val);*/ return;
    // RSP Interface
    case 0x4040000: RSP::cop0[0] = val & 0x1fff; return;
    case 0x4040004: RSP::cop0[1] = val & 0xffffff; return;
    case 0x4040008: RSP::dma(val, false); return;
    case 0x404000c: RSP::dma(val, true); return;
    case 0x4040010: RSP::set_status(val); return;
    case 0x404001c: RSP::cop0[7] = 0x0; return;
    case 0x4080000: RSP::pc = val & 0xffc; return;
    // RDP Interface
    case 0x4100000:
      // set RDP_PC_START
      RSP::cop0[8] = val & R4300::mask;
      RSP::cop0[10] = RSP::cop0[8]; return;
    case 0x4100004:
      // set RDP_PC_END
      Sched::add(TASK_RDP, 0);
      RSP::cop0[9] = val & R4300::mask; return;
    case 0x410000c:
      // update RDP_STATUS
      RSP::cop0[11] &= ~pext(val >> 0, 0x7);
      RSP::cop0[11] |= pext(val >> 1, 0x7); return;
    // MIPS Interface
    case 0x4300000:
      if (val & 0x800) R4300::unset_irqs(0x20); return;
    case 0x430000c:
      mi_mask &= ~pext(val >> 0, 0x3f);
      mi_mask |= pext(val >> 1, 0x3f);
      if (mi_irqs & mi_mask) {
        cop0[13] |= 0x400, cop0[13] &= ~0xff;
      } else cop0[13] &= ~0x400; return;
    // Video Interface
    case 0x4400000:
      if (val == vi_status) return;
      vi_status = val, vi_dirty = true; return;
    case 0x4400004: vi_origin = val & 0xffffff; return;
    case 0x4400008:
      if ((val & 0xfff) == vi_width) return;
      vi_width = val & 0xfff, vi_dirty = true; return;
    case 0x440000c: vi_irq_line = val & 0x3ff; return;
    case 0x4400010: R4300::unset_irqs(0x8); return;
    case 0x4400018:
      if ((val & 0x3ff) == vi_height) return;
      vi_height = val & 0x3ff, vi_dirty = true; return;
    // Audio Interface
    case 0x4500000: ai_ram = val & 0xfffff8; return;
    case 0x4500004: ai_dma(val); return;
    case 0x4500008: ai_run = val & 0x1; return;
    case 0x450000c: R4300::unset_irqs(0x4); return;
    case 0x4500010:
      if ((val & 0xfff) == ai_rate) return;
      ai_rate = val & 0xfff, ai_dirty = true; return;
    case 0x4500014:
      if (((val >> 3) & 0x1) == ai_16bit) return;
      ai_16bit = (val >> 3) & 0x1, ai_dirty = true; return;
    // Peripheral Interface
    case 0x4600000: pi_ram = val & 0xffffff; return;
    case 0x4600004: pi_rom = val & R4300::mask; return;
    case 0x4600008:
      if (pi_status & 0x1) return;
      pi_len = val + 1, pi_to_rom = true;
      Sched::add(TASK_PI, 0);
      pi_status |= 0x1; return;
    case 0x460000c:
      if (pi_status & 0x1) return;
      pi_len = val + 1, pi_to_rom = false;
      Sched::add(TASK_PI, 0);
      pi_status |= 0x1; return;
    case 0x4600010: R4300::unset_irqs(0x10); return;
    // Serial Interface
    case 0x4800000: si_ram = val & 0xffffff; return;
    case 0x4800004: si_update();
      memcpy(ram + si_ram, ram + 0x1fc007c0, 0x40);
      R4300::set_irqs(0x2); return;
    case 0x4800010:
      memcpy(ram + 0x1fc007c0, ram + si_ram, 0x40);
      R4300::set_irqs(0x2); return;
    case 0x4800018: R4300::unset_irqs(0x2); return;
  }
}

// set bits 19/18 for physical pages with MMIO
static uint32_t mmio_bit(uint32_t pg, uint32_t len) {
  if (pg >= 0x3f00 && pg < 0x4000) pg |= 0xc0000;
  if (pg >= 0x4002 && pg < 0x5000) pg |= 0xc0000;
  return pg & ~(len - 1);  // align page address
}

// write to TLB from cop0, updating page table
static void tlb_write(uint32_t idx, uint64_t) {
  uint32_t pg = (tlb[idx][1] >> 12) & ~1;
  uint32_t len = (tlb[idx][0] >> 13) + 1;

  // unmap previous page in slot
  for (uint32_t i = 0; i < len * 2; ++i, ++pg) {
    if ((pg >> 18) == 0x2) break;
    pages[pg] = ((pg >> 18) - 0x2) << 30;
  }

  // calculate physical address diff
  pg = (uint32_t)cop0[10] >> 12;
  len = ((uint32_t)cop0[5] >> 13) + 1;
  uint32_t d1 = (pg &= ~1) - mmio_bit(cop0[2] >> 6, len);
  uint32_t d2 = (pg + len) - mmio_bit(cop0[3] >> 6, len);

  // map new page in slot
  const uint8_t entry[4] = {5, 10, 2, 3};
  for (uint8_t i = 0; i < 4; ++i)
    tlb[idx][i] = (uint32_t)cop0[entry[i]];
  for (uint32_t i = 0; i < len * 2; ++i, ++pg) {
    if ((pg >> 18) == 0x2) break;
    pages[pg] = (i < len ? d1 : d2) << 12;
  }
}

// read instruction from vaddr
static uint32_t fetch(uint32_t addr) {
  //printf("R4300 PC: %x\n", addr);
  addr = addr - pages[addr >> 12];
  if (addr >> 31) printf("Invalid fetch of %x\n", addr);
  return read32(R4300::ram + addr);
}

/* === Code change detection === */

static CodePtr lookup[0x20000000 / 4];
robin_hood::unordered_map<uint32_t, std::vector<uint32_t>> prot_pages;
uint8_t *sram;

#ifdef _WIN32

static uint8_t *alloc_pages(uint32_t size) {
  return (uint8_t*)VirtualAlloc(NULL, size,
    MEM_COMMIT, PAGE_READWRITE);
}

static void sram_protect() {
  DWORD old = PAGE_READWRITE;
  VirtualProtect(R4300::ram + 0x8000000, 0x10000, PAGE_NOACCESS, &old);
}

static void sram_init(char *name) {
  strcpy(name + strlen(name) - 4, ".sra");
  int file = _open(name, _O_RDWR | _O_CREAT, 0644);
  if (file < 0 || _chsize(file, 0x10000) < 0) exit(1);
  sram = (uint8_t*)MapViewOfFileEx((HANDLE)_get_osfhandle(file),
    FILE_MAP_ALL_ACCESS, 0, 0, 0x10000, R4300::ram + 0x8000000);
  _close(file);
}

static void protect(uint32_t pg) {
  DWORD old = PAGE_READWRITE;
  if (prot_pages[pg].empty())
    VirtualProtect(R4300::ram + (pg << 12), 0x1000, PAGE_READONLY, &old);
  uint32_t ppc = pc - pages[pc >> 12];
  prot_pages[pg].push_back(ppc / 4);
}

static void unprotect(uint32_t pg) {
  DWORD old = PAGE_READONLY;
  for (uint32_t addr : prot_pages[pg]) lookup[addr] = NULL;
  VirtualProtect(R4300::ram + (pg << 12), 0x1000, PAGE_READWRITE, &old);
  prot_pages[pg].clear();
}

static LONG WINAPI handle_fault(_EXCEPTION_POINTERS *info) {
  DWORD sig = info->ExceptionRecord->ExceptionCode;
  if (sig != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
  uint8_t *addr = (uint8_t*)info->ExceptionRecord->ExceptionInformation[1];
  int64_t pg = addr - R4300::ram;
  if (!(0 <= pg && pg <= R4300::mask)) exit(1);
  unprotect((uint32_t)pg >> 12); return EXCEPTION_CONTINUE_EXECUTION;
}

static void setup_fault_handler() {
  AddVectoredExceptionHandler(true, handle_fault);
}

#else

static uint8_t *alloc_pages(uint32_t size) {
  return (uint8_t*)mmap(NULL, size, PROT_READ | PROT_WRITE,
    MAP_ANONYMOUS | MAP_SHARED, 0, 0);
}

static void sram_protect() {
  mprotect(R4300::ram + 0x8000000, 0x10000, PROT_NONE);
}

static void sram_init(char *name) {
  strcpy(name + strlen(name) - 4, ".sra");
  int file = open(name, O_RDWR | O_CREAT, 0644);
  if (file < 0 || ftruncate(file, 0x10000) < 0) exit(1);
  sram = (uint8_t*)mmap(R4300::ram + 0x8000000, 0x10000,
    PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, file, 0);
  close(file);
}

static void protect(uint32_t pg) {
  if (prot_pages[pg].empty())
    mprotect(R4300::ram + (pg << 12), 0x1000, PROT_READ);
  uint32_t ppc = pc - pages[pc >> 12];
  prot_pages[pg].push_back(ppc / 4);
}

static void unprotect(uint32_t pg) {
  if (0x8000 <= pg && pg < 0x8010 && !sram) sram_init(strdup(title));
  for (uint32_t addr : prot_pages[pg]) lookup[addr] = NULL;
  mprotect(R4300::ram + (pg << 12), 0x1000, PROT_READ | PROT_WRITE);
  prot_pages[pg].clear();
}

static void handle_fault(int sig, siginfo_t *info, void*) {
  if (sig != SIGBUS && sig != SIGSEGV) return;
  int64_t pg = (uint8_t*)info->si_addr - R4300::ram;
  if (!(0 <= pg && pg <= R4300::mask)) exit(1);
  unprotect((uint32_t)pg >> 12);
}

static void setup_fault_handler() {
  struct sigaction act;
  memset(&act, 0, sizeof(act));
  act.sa_sigaction = handle_fault;
  act.sa_flags = SA_RESTART | SA_SIGINFO;
  sigemptyset(&act.sa_mask);
  sigaction(SIGBUS, &act, NULL);
}

#endif

/* === Debugger interface === */

static robin_hood::unordered_map<uint32_t, bool> breaks;
static bool broke, step;

// read registers in gdb MIPS order
static uint64_t read_reg(uint32_t idx) {
  if (idx < 32) return regs[idx];
  if (idx >= 38) return cop1[idx - 38];
  switch (idx) {
    default: printf("Invalid reg %x\n", idx), exit(1);
    case 32: return cop0[12];
    case 33: return regs[33];
    case 34: return regs[32];
    case 35: return cop0[8];
    case 36: return cop0[13];
    case 37: return pc;
  }
}

// read from virtual address
static uint64_t read_mem(uint32_t addr) {
  addr = addr - pages[addr >> 12];
  if (addr >> 31) return read(addr);
  return bswap64(*(uint64_t*)(R4300::ram + addr));
}

// create or delete breakpoint
static void set_break(uint32_t addr, bool active) {
  breaks[addr] = active;
}

const DbgConfig dbg = {
  .read_reg = read_reg,
  .read_mem = read_mem,
  .set_break = set_break
};

void R4300::init_debug(uint32_t port) {
  Debugger::init(port);
  broke = step = Debugger::update(&dbg);
}

/* === Recompiler interface === */

// fire timer interrupt
void R4300::timer_fire() {
  cop0[13] |= 0x8000, cop0[13] &= ~0xff;
}

// update timer on cop0 write
static void mtc0(uint32_t idx, uint64_t val) {
  cop0[idx &= 0x1f] = val;
  printf("Ran mtc0\n");
  if (idx != 9 && idx != 11) return;
  if (idx == 11) cop0[13] &= ~0x8000;
  uint32_t cycles = cop0[11] - cop0[9];
  Sched::move(TASK_TIMER, cycles * 2);
}

// stop compiler at 4kb boundaries
static int64_t stop_at(uint32_t addr) {
  if (!(addr & 0xfff)) return true;
  return broke = step || breaks[addr];
}

// mem pointer filled in on init()
static MipsConfig cfg = {
  .regs = regs, .cop0 = 34,
  .cop1 = 66, .pool = 99,
  .lookup = lookup, .fetch = fetch,
  .mtc0 = mtc0, .stop_at = stop_at,

  .pages = pages, .tlb = tlb[0],
  .read = read, .write = write,
  .tlbwi = tlb_write
};

static uint32_t crc32(uint8_t *bytes, uint32_t len) {
  uint32_t crc = 0, *msg = (uint32_t*)bytes;
  for (uint32_t i = 0; i < len / 4; ++i)
    crc = _mm_crc32_u32(crc, msg[i]);
  return crc;
}

void R4300::update() {
  while (Sched::until >= 0) {
    uint32_t ppc = pc - pages[pc >> 12];
    CodePtr code = lookup[ppc / 4];
    if (code) {
      pc = code();
      if (!broke) continue;
      step = Debugger::update(&dbg);
      memset(lookup, 0, sizeof(lookup));
      broke = false;
    } else {
      protect(ppc >> 12);
      Mips::jit(&cfg, pc, lookup + ppc / 4);
    }
  }
  Sched::add(TASK_R4300, 0);
}

void R4300::init(const char *name) {
  // allocate memory, setup change detection
  ram = cfg.mem = alloc_pages(0x20000000);
  setup_fault_handler();

  // (paddr >> 18) == 3 for unmapped regions
  // (paddr >> 18) == 2 for MMIO regions
  // page[vaddr >> 12] = vaddr - paddr
  for (uint32_t pg = 0; pg < 0x80000; ++pg)
    pages[pg] = ((pg >> 18) - 0x2) << 30;
  for (uint32_t pg = 0xc0000; pg < 0xe0000; ++pg)
    pages[pg] = ((pg >> 18) - 0x2) << 30;
  for (uint32_t pg = 0; pg < 0x20000; ++pg) {
    uint32_t v1 = 0x80000 + pg, v2 = 0xa0000 + pg;
    pages[v1] = (v1 - mmio_bit(pg, 1)) << 12;
    pages[v2] = (v2 - mmio_bit(pg, 1)) << 12;
  }

  // read ROM file into memory
  FILE *rom = fopen(title = name, "r");
  if (!rom) printf("Can't find rom %s\n", name), exit(1);
  fread(ram + 0x10000000, 1, 0x4000000, rom), fclose(rom);
  memcpy(ram + 0x4000000, ram + 0x10000000, 0x40000);
  mempak_init("mempak.mpk"), sram_protect();

  // read PIF boot rom, setup ram based on CIC
  FILE *pifrom = fopen("pifdata.bin", "r");
  if (!pifrom) printf("Can't find pifdata.bin\n"), exit(1);
  fread(ram + 0x1fc00000, 1, 0x7c0, pifrom), fclose(pifrom);

  write32(ram + 0x318, 0x800000);  // 8MB RDRAM
  switch (crc32(ram + 0x04000040, 0xfc0)) {
    case 0x583af077: write32(ram + 0x1fc007e4, 0x43f3f); break;  // 6101
    case 0x98a02fa9: write32(ram + 0x1fc007e4, 0x3f3f); break;   // 6102
    case 0x04e7fe6d: write32(ram + 0x1fc007e4, 0x783f); break;   // 6103
    case 0x035e73e4: write32(ram + 0x1fc007e4, 0x913f); break;   // 6105
    case 0x0f727fb1: write32(ram + 0x1fc007e4, 0x853f); break;   // 6106
    default: printf("No compatible CIC chip found\n"), exit(1);
  }

  // setup other components
  Mips::init_pool(regs + 99);
  vi_init(), RDP::init();
  RSP::init(ram + 0x04000000);
}

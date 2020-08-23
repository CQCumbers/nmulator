/* === Bitfield Flags === */

#define M0_FILL      0x00300000
#define M0_COPY      0x00200000
#define M0_2CYCLE    0x00100000
#define M0_PERSP     0x00080000
#define M0_TLUT_EN   0x00008000
#define M0_TLUT_IA   0x00004000

#define M1_BLEND     0x00004000
#define M1_CVG2ALPHA 0x00002000
#define M1_ALPHA2CVG 0x00001000
#define M1_ZMODE     0x00000c00
#define ZTRANS       0x00000800
#define ZDECAL       0x00000c00
#define M1_ON_CVG    0x00000080
#define M1_ZWRITE    0x00000020
#define M1_ZCMP      0x00000010
#define M1_AA        0x00000008
#define M1_ZSRC      0x00000004
#define M1_ALPHA     0x00000001

#define T_ZBUF       0x01
#define T_TEX        0x02
#define T_SHADE      0x04
#define T_RECT       0x08
#define T_LMAJOR     0x10

/* === Struct Definitions === */

struct TileData {
  uint cmd_idxs[64];
};

struct GlobalData {
  uint width, size;
  uint n_cmds, pad;
  uint2 zenc[128];
  uint2 zdec[8];
};

struct RDPCommand {
  uint type, tile, pad;
  int xh, xm, xl;
  int yh, ym, yl;
  int sh, sm, sl;

  int4 shade, sde, sdx;
  int4 tex, tde, tdx;

  int sxh, sxl, syh, syl;
  uint modes[2], mux[2];
  uint tlut, tmem, texes;
  uint fill, fog, blend;
  uint env, prim, lodf;
  uint zprim, keys, keyc;
};

struct RDPTex {
  uint format, size;
  uint width, addr;
  int2 sth, stl, shift;
  uint pal, pad;
};

/* === Resource Buffers === */

[[vk::binding(0)]] StructuredBuffer<RDPCommand> cmds;
[[vk::binding(1)]] StructuredBuffer<TileData> tiles;
[[vk::binding(2)]] StructuredBuffer<RDPTex> texes;
[[vk::binding(3)]] StructuredBuffer<GlobalData> globals;

[[vk::binding(4)]] StructuredBuffer<uint16_t> tmem;
[[vk::binding(5)]] RWStructuredBuffer<uint16_t> pix16;
[[vk::binding(6)]] RWStructuredBuffer<uint16_t> zbuf;

/* === Format Conversions === */

uint read_rgba16(uint enc) {
  // bswapped RGBA5551 to ABGR8888
  enc = ((enc << 8) | (enc >> 8)) & 0xffff;
  uint dec = -(enc & 0x1) & (0xff << 24);
  dec |= (enc & 0x3e) << 18;
  dec |= (enc & 0x7c0) << 5;
  return dec | ((enc & 0xf800) >> 8);
}

uint16_t write_rgba16(uint dec) {
  // ABGR8888 to bswapped RGBA5551
  uint enc = (dec >> 30) & 0x1;
  enc |= (dec >> 18) & 0x3e;
  enc |= (dec >> 5) & 0x7c0;
  enc |= (dec << 8) & 0xf800;
  return (uint16_t)((enc << 8) | (enc >> 8));
}

uint read_ia16(uint enc) {
  // 8 bit intensity, 8 bit alpha
  uint i = enc & 0xff, a = enc >> 8;
  return (a << 24) | (i << 16) | (i << 8) | i;
}

uint read_i4(uint enc) {
  // 4 bit intensity
  uint i = (enc << 4) | enc;
  return (i << 24) | (i << 16) | (i << 8) | i;
}

/* === Utility Functions === */

uint4 usadd(uint4 a, uint4 b) {
  return b == 0 ? a : a + max(b, -a);
}

int mul16(int a, int b) {
   // multiply 16.16 ints, with 16.16 result
   int a1 = a >> 16, a2 = a & 0xffff;
   int b1 = b >> 16, b2 = b & 0xffff;
   return (a1 * b1 << 16) + a1 * b2 + a2 * b1;
}

uint sel(uint idx, uint4 inputs) {
  switch (idx & 0x3) {
    default: return inputs.x;
    case 1: return inputs.y;
    case 2: return inputs.z;
    case 3: return inputs.w;
  }
}

uint sel(uint idx, uint4 i0, uint4 i1) {
   if (idx >= 8) return 0;
   uint4 inp = (idx & 0x4) ? i1 : i0;
   return sel(idx, inp);
}

uint sel(uint idx, uint4 i0, uint4 i1, uint4 i2, uint4 i3) {
   if (idx >= 16) return 0;
   uint4 inp0 = (idx & 0x8) ? i2 : i0;
   uint4 inp1 = (idx & 0x8) ? i3 : i1;
   return sel(idx & 0x7, inp0, inp1);
}

uint pack32(uint4 color) {
  uint4 c = (color >> 16) & 0xff;
  return (c.w << 24) | (c.z << 16) | (c.y << 8) | c.x;
}

uint4 unpack32(uint color) {
  uint a = color >> 24, b = color >> 16;
  uint4 c = uint4(color, color >> 8, b, a);
  return (c & 0xff) << 16;
}

uint unpacka(uint color) {
  return (color >> 24) << 16;
}

uint unpackz(uint enc) {
  enc = ((enc << 8) | (enc >> 8)) & 0xffff;
  // 14-bit float Z to 18-bit uint
  int mant = (enc >> 2) & 0x7ff, exp_ = enc >> 13;
  uint shift = max(6 - exp_, 0);
  uint base = 0x40000 - (0x40000 >> exp_);
  uint dec = (mant << shift) + base;
  // load upper 2 bits of dz only
  uint dz = (enc & 0x3) << 2;
  return ((dec & 0x3ffff) << 4) | dz;
}

uint packz(uint dec) {
  int exp_ = firstbithigh(max(dec ^ 0x3fffff, 1));
  exp_ = clamp(21 - exp_, 0, 7);
  uint shift = max(10 - exp_, 4);
  uint mant = (dec >> shift) & 0x7ff;
  // only upper 2 bits of dz are stored
  uint dz = (dec >> 2) & 0x3;
  uint enc = (exp_ << 13) | (mant << 2) | dz;
  return ((enc << 8) | (enc >> 8)) & 0xffff;
}

uint tmem4r(uint pos, uint off, uint idx) {
  pos = (pos & 0xfff) | (idx << 12);
  uint val = tmem[pos >> 1];
  uint shift = ((off & 0x3) ^ 0x1) * 4;
  return (val >> shift) & 0xf;
}

uint tmem8r(uint pos, uint idx) {
  pos = (pos & 0xfff) | (idx << 12);
  uint val = tmem[pos >> 1];
  uint shift = (pos & 0x1) * 8;
  return (val >> shift) & 0xff;
}

uint tmem16r(uint pos, uint idx) {
  pos = (pos & 0xfff) | (idx << 12);
  return tmem[pos >> 1];
}

/* === Pipeline Stages === */

uint texel(uint2 st, uint2 mask, RDPTex tex, RDPCommand cmd) {
  // apply mirror/mask to ST coords
  if (cmd.modes[0] & M0_TLUT_EN) tex.format = 2;
  bool2 mst = tex.shift & 0x100, en = (mask != 0);
  st ^= -(en & (st >> mask) & mst), st &= (en << mask) - 1;

  // see manual 13.8 for tmem layouts
  uint s = st.x, t = st.y, wd = tex.width;
  uint flip = (st.y & 0x1) << 2;
  bool ia = cmd.modes[0] & M0_TLUT_IA;

  if (tex.format == 0 && tex.size == 0) {        // 4 bit RGBA
    uint pos = (tex.addr + t * wd + s / 2) ^ flip;
    return read_i4(tmem4r(pos, s, cmd.tmem));
  } else if (tex.format == 0 && tex.size == 1) { // 8 bit RGBA
    uint pos = (tex.addr + t * wd + s) ^ flip;
    uint i = tmem8r(pos, cmd.tmem);
    return (i << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 0 && tex.size == 2) { // 16 bit RGBA
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    return read_rgba16(tmem16r(pos, cmd.tmem));
  } else if (tex.format == 0 && tex.size == 3) { // 32 bit RGBA
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    uint hi = tmem16r(pos | 0x800, cmd.tmem);
    uint lo = tmem16r(pos & 0x7ff, cmd.tmem);
    return (hi << 16) | (lo & 0xffff);
  } else if (tex.format == 2 && tex.size == 0) { // 4 bit CI
    uint pos = (tex.addr + t * wd + s / 2) ^ flip;
    uint idx = (tex.pal << 4) | tmem4r(pos & 0x7ff, s, cmd.tmem);
    if (~cmd.modes[0] & M0_TLUT_EN) return idx;
    uint entry = tmem16r(0x800 | (idx << 3), cmd.tmem);
    return ia ? read_ia16(entry) : read_rgba16(entry);
  } else if (tex.format == 2 && tex.size == 1) { // 8 bit CI
    uint pos = (tex.addr + t * wd + s) ^ flip;
    uint idx = tmem8r(pos & 0x7ff, cmd.tmem);
    if (~cmd.modes[0] & M0_TLUT_EN) return idx;
    uint entry = tmem16r(0x800 | (idx << 3), cmd.tmem);
    return ia ? read_ia16(entry) : read_rgba16(entry);
  } else if (tex.format == 2) {                  // 32 bit CI
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    uint idx = tmem16r(pos & 0x7ff, cmd.tmem);
    if (~cmd.modes[0] & M0_TLUT_EN) return (idx << 16) | idx;
    uint entry = tmem16r(0x800 | ((idx >> 8) << 3), cmd.tmem);
    return ia ? read_ia16(entry) : read_rgba16(entry);
  } else if (tex.format == 3 && tex.size == 0) { // 4 bit IA
    uint pos = (tex.addr + t * wd + s / 2) ^ flip;
    uint val = tmem4r(pos, s, cmd.tmem);
    uint i = val & 0xe, a = (val & 0x1) * 0xff;
    i = (i << 4) | (i << 1) | (i >> 2);
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 1) { // 8 bit IA
    uint pos = (tex.addr + t * wd + s) ^ flip;
    uint val = tmem8r(pos, cmd.tmem);
    uint i = (val & 0xf0) | (val >> 4);
    uint a = (val << 4) | (val & 0x0f);
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 2) { // 16 bit IA
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    return read_ia16(tmem16r(pos, cmd.tmem));
  } else if (tex.format == 3 && tex.size == 3) { // 32 bit IA
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    uint val = tmem16r(pos, cmd.tmem);
    return (val << 16) | (val & 0xffff);
  } else if (tex.format == 4 && tex.size == 0) { // 4 bit I
    uint pos = (tex.addr + t * wd + s / 2) ^ flip;
    return read_i4(tmem4r(pos, s, cmd.tmem));
  } else if (tex.format == 4 && tex.size == 1) { // 8 bit I
    uint pos = (tex.addr + t * wd + s) ^ flip;
    uint i = tmem8r(pos, cmd.tmem);
    return (i << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 4) {                  // 32 bit I
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    uint val = tmem16r(pos, cmd.tmem);
    return (val << 16) | (val & 0xffff);
  }
  return 0xffff00ff;
}

uint read_noise(uint seed) {
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9, seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d, seed = seed ^ (seed >> 15);
  uint i = seed & 0xff;
  return 0xff000000 | (i << 16) | (i << 8) | i;
}

uint visible(uint2 pos, RDPCommand cmd, out int2 dxy) {
  // check pixel within y bounds
  int y1 = max(cmd.yh, cmd.syh);
  int y2 = min(cmd.yl, cmd.syl);
  if (!(y1 / 4 <= pos.y && pos.y <= y2 / 4)) return 0;
  // compute edges at each sub-scanline
  int4 y = pos.y * 4 + int4(0, 1, 2, 3);
  int4 x1 = cmd.xh + cmd.sh * (y - cmd.yh) / 4;
  int4 xm = cmd.xm + cmd.sm * (y - cmd.yh) / 4;
  int4 xl = cmd.xl + cmd.sl * (y - cmd.ym) / 4;
  int4 x2 = y < (int4)cmd.ym ? xm : xl;
  // compute valid sub-scanlines
  bool lft = cmd.type & (T_LMAJOR | T_RECT);
  int4 xa = (lft ? x1 : x2), xb = (lft ? x2 : x1);
  if (cmd.modes[0] & M0_COPY) y2 += 1, xb += 1 << 14;
  bool4 vy = (int4)y1 <= y && y < (int4)y2 && xa < xb;
  dxy = pos * 4, dxy.x -= (x1.x >> 14), dxy.y -= cmd.yh;
  // check x bounds at every other subpixel
  xa = max(xa, (int4)cmd.sxh) >> 14;
  xb = min(xb, (int4)cmd.sxl) >> 14;
  int4 x = pos.x * 4 + int4(0, 1, 0, 1);
  uint cvg = dot(vy && xa <= x + 2 && x + 2 < xb, 1);
  return cvg + dot(vy && xa <= x && x < xb, 1);
}

bool2 compare_z(inout uint zmem, uint cvg, int2 dxy, RDPCommand cmd, out int4 stwz) {
  stwz = cmd.tex + (cmd.tde / 4) * dxy.y;
  stwz = stwz + (cmd.tdx / 4) * dxy.x;
  // read old z, dz from zbuf
  bool skip = (~cmd.type & T_ZBUF) || (~cmd.modes[1] & M1_ZCMP);
  skip = skip || (cmd.modes[0] & M0_COPY);
  if (cvg == 0 || skip) return (bool2)true;
  uint oz = zmem >> 4, odz = zmem & 0xf;
  bool zmax = (oz == 0x3ffff);
  // calculate new z from slopes
  uint nz = (cmd.zprim >> 14) & 0x3fffc;
  if (~cmd.modes[1] & M1_ZSRC) nz = stwz.w >> 14;
  // increase dz at greater floating point depths
  int precision = (packz(zmem) >> 5) & 0x7;
  if (precision < 3) odz = max(odz + 1, 4 - precision);
  // get new dz from prim_z
  uint ndz = firstbithigh(max(cmd.zprim & 0xffff, 1));
  uint dz = 0x8 << max(ndz, odz);
  uint mode = cmd.modes[1] & M1_ZMODE;
  if (mode == ZTRANS || (mode != ZDECAL && cvg > 7)) dz = 0;
  // compare new z with old z
  bool zle = int((nz - dz) << 14) <= int(oz << 14);
  bool zge = int((nz + dz) << 14) >= int(oz << 14);
  if (mode != ZDECAL) zle = zmax || zle;
  else zle = zge && zle && !zmax;
  // write new z/dz if closer
  bool write = cmd.modes[1] & M1_ZWRITE;
  if (zle && write) zmem = (nz << 4) | ndz;
  return bool2(zle, zge);
}

uint sample_tex(int3 stw, uint tile_off, RDPCommand cmd) {
  if (~cmd.type & T_TEX) return 0;
  uint mode = cmd.modes[0]; stw >>= 16;
  RDPTex tex = texes[cmd.tmem * 8 + cmd.tile + tile_off];
  // correct perspective, apply shift
  //if (stw.z > 0) {
  if (mode & M0_PERSP) stw = 32768.0 * stw / (float3)stw.z;
  //} else stw = (int3)0x7fff;
  int2 shl = tex.shift & 0xf, st = stw.xy;
  st = shl > 10 ? st << (16 - shl) : st >> shl;
  // get mask, clamp coords to tex bounds
  uint2 mask = min((tex.shift >> 4) & 0xf, 10);
  bool2 cst = (tex.shift >> 9) || (mask == 0);
  int2 stl = tex.stl << 3, sth = tex.sth << 3;
  st = cst ? clamp(st - stl, 0, sth - stl) : st - stl;
  // read appropriate texels
  int2 stfrac = st & 0x1f; st >>= 5;
  bool up = (stfrac.x + stfrac.y) & 0x20;
  int4 c1 = unpack32(texel(st + (int2)up, mask, tex, cmd));
  int4 c2 = unpack32(texel(st + int2(1, 0), mask, tex, cmd));
  int4 c3 = unpack32(texel(st + int2(0, 1), mask, tex, cmd));
  // 3-tap bilinear filter
  stfrac = up ? 0x20 - stfrac.yx : stfrac.xy;
  int4 mix = stfrac.x * (c2 - c1) + stfrac.y * (c3 - c1);
  return pack32(c1 + (mix + 0x10) / 0x20);
}

uint sample_shade(int2 dxy, RDPCommand cmd) {
  if (~cmd.type & T_SHADE) return 0;
  int4 rgba = cmd.shade + cmd.sde * dxy.y / 4;
  return pack32(max(rgba + cmd.sdx * dxy.x / 4, 0));
}

int4 sext(uint4 val) {
  uint sgn = 1 << 25, mask = (1 << 26) - 1;
  return ((val & mask) ^ sgn) - sgn;
}

uint2 combine(uint tex0, uint tex1, uint shade, uint noise_, uint2 prev, RDPCommand cmd, uint cycle) {
  uint cycle_ = cmd.modes[0] & M0_FILL;
  if (cycle_ == M0_COPY) return tex0;
  if (cycle_ == M0_FILL) return read_rgba16(cmd.fill);
  // setup input options
  uint4 i1 = uint4(prev.x, tex0, tex1, cmd.prim);
  uint4 i2 = uint4(shade, cmd.env, -1, 0);
  uint4 i3 = uint4(tex0, tex1, cmd.prim, shade);
  uint4 ia2 = uint4(shade, cmd.env, -1, noise_);
  uint4 ib2 = uint4(shade, cmd.env, cmd.keyc, 0);
  uint4 ic2 = uint4(shade, cmd.env, cmd.keys, 0);
  // select packed inputs
  uint mux0 = cmd.mux[0], mux1 = cmd.mux[1];
  uint mc, pa, pb, pc, pd;
  if (cycle == 0) {
    mc = (mux0 >> 15) & 0x1f;
    pa = sel((mux0 >> 20) & 0xf, i1, ia2);
    pb = sel((mux1 >> 28) & 0xf, i1, ib2);
    pc = sel(mc, i1, ic2, i3, uint4(cmd.env, 0, cmd.lodf, 0));
    pd = sel((mux1 >> 15) & 0x7, i1, i2);
  } else {
    mc = (mux0 >> 0) & 0x1f;
    pa = sel((mux0 >> 5) & 0xf, i1, ia2);
    pb = sel((mux1 >> 24) & 0xf, i1, ib2);
    pc = sel(mc, i1, ic2, i3, uint4(cmd.env, 0, cmd.lodf, 0));
    pd = sel((mux1 >> 6) & 0x7, i1, i2);
  }
  // unpack input channels
  int4 a = unpack32(pa), b = unpack32(pb);
  int4 c = unpack32(pc), d = unpack32(pd);
  if (cycle != 0 && mc == 0) {
    c |= unpack32(prev.y) << 8;
  }
  if (mc >= 7) c = c.wwww;
  // select alpha inputs
  if (cycle == 0) {
    a.w = unpacka(sel((mux0 >> 12) & 0x7, i1, i2));
    b.w = unpacka(sel((mux1 >> 12) & 0x7, i1, i2));
    d.w = unpacka(sel((mux1 >> 9) & 0x7, i1, i2));
    uint4 tmp1 = uint4(0, tex0, tex1, cmd.prim);
    uint4 tmp2 = uint4(shade, cmd.env, cmd.lodf, 0);
    c.w = unpacka(sel((mux0 >> 9) & 0x7, tmp1, tmp2));
  } else {
    a.w = unpacka(sel((mux1 >> 21) & 0x7, i1, i2));
    b.w = unpacka(sel((mux1 >> 3) & 0x7, i1, i2));
    d.w = unpacka(sel((mux1 >> 0) & 0x7, i1, i2));
    uint4 tmp1 = uint4(0, tex0, tex1, cmd.prim);
    uint4 tmp2 = uint4(shade, cmd.env, cmd.lodf, 0);
    c.w = unpacka(sel((mux1 >> 18) & 0x7, tmp1, tmp2));
  }
  // combine selected inputs
  int4 res = ((a - b) >> 16) * (c >> 16);
  res = d + ((res + 0x80) << 8);
  return uint2(pack32(res), pack32(res >> 8));
}

void mix_alpha_cvg(inout uint alpha, inout uint cvg, RDPCommand cmd) {
  // mix 8-bit alpha and 3-bit coverage
  int mode = cmd.modes[1], cvg8 = cvg << 5;
  alpha >>= 16, alpha = alpha + (alpha + 1) / 256;
  if (mode & M1_ALPHA2CVG) cvg8 = (alpha * cvg + 4) / 8;
  alpha = (mode & M1_CVG2ALPHA) ? cvg8 : alpha + 0;
  alpha = clamp(alpha, 0, 0xff) << 16; cvg = cvg8 >> 5;
}

uint blend(uint pixel, uint color, uint shade, uint alpha, inout uint zmem, uint oz,
    uint cvg, bool far, RDPCommand cmd, uint cycle) {
  int mux = cmd.modes[1], ocvg = pixel >> 29;
  // alpha test if enabled
  mix_alpha_cvg(alpha, cvg, cmd);
  if (cvg == 0) { zmem = oz; return pixel; }
  if (cycle == 0 && (mux & M1_ALPHA) && alpha < unpacka(cmd.blend)) { zmem = oz; return pixel; }
  // select blender inputs
  uint p, a, m, b;
  if (cycle == 0) {
    p = sel(mux >> 30, uint4(color, pixel, cmd.blend, cmd.fog));
    a = sel(mux >> 26, uint4(alpha >> 19, cmd.fog >> 27, shade >> 27, 0x0));
    m = sel(mux >> 22, uint4(color, pixel, cmd.blend, cmd.fog));
    b = sel(mux >> 18, uint4(~a & 0x1f, pixel >> 27, 0x1f, 0x0)) + 1;
  } else {
    p = sel(mux >> 28, uint4(color, pixel, cmd.blend, cmd.fog));
    a = sel(mux >> 24, uint4(alpha >> 19, cmd.fog >> 27, shade >> 27, 0x0));
    m = sel(mux >> 20, uint4(color, pixel, cmd.blend, cmd.fog));
    b = sel(mux >> 16, uint4(~a & 0x1f, pixel >> 27, 0x1f, 0x0)) + 1;
  }
  //if (copy) return (full && alpha ? color : pixel);
  // blend selected inputs
  uint4 p4 = unpack32(p), m4 = unpack32(m);
  uint4 res = (a * p4 + b * m4) / (a + b);
  if (cmd.modes[0] & M0_2CYCLE) return pack32(res);
  // blend coverage
  int c1 = clamp(cvg + ocvg, 0, 7), c2 = (cvg + ocvg) & 7;
  uint output = sel(mux >> 8, uint4(c1, c2, 7, ocvg)) << 29;
  // skip inputs based on flags
  bool copy = cmd.modes[0] & M0_COPY;
  bool on_cvg = mux & M1_ON_CVG, force = mux & M1_BLEND;
  bool full = cvg + ocvg > 7, aa = mux & M1_AA;
  if (on_cvg && !full) return output | (m & 0x1fffffff);
  if (!(force || (!full && aa && far))) return output | (p & 0x1fffffff);
  return (pack32(res) & 0xffffff) | output;
}

uint clamp_color16(uint2 inp) {
  int4 color = unpack32(inp.x);
  color |= unpack32(inp.y) << 8;
  color = (((color - 0x800000) << 7) >> 7) + 0x800000;
  return pack32(clamp(color, 0, 0xff0000));
}
  

[numthreads(8, 8, 1)]
void main(uint3 GlobalID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID) {
  GlobalData global = globals[0];
  if (GlobalID.x >= global.width) return;
  if (GlobalID.y >= 240) return;

  uint pos = GlobalID.y * global.width + GlobalID.x;
  uint pixel = read_rgba16(pix16[pos]);
  uint zmem = unpackz(zbuf[pos]);
  uint noise_ = 0xff888888;

  TileData tile = tiles[GroupID.y * (global.width / 8) + GroupID.x];
  for (uint i = 0; i < 64; ++i) {
    uint bitmask = tile.cmd_idxs[i];
    while (bitmask != 0) {
      uint lsb = firstbitlow(bitmask);
      RDPCommand cmd = cmds[(i << 5) | lsb];
      bitmask &= ~(0x1 << lsb);

      int2 dxy; int4 stwz; uint oz = zmem;
      uint cvg = visible(GlobalID.xy, cmd, dxy);
      bool2 depth = compare_z(zmem, cvg, dxy, cmd, stwz);
      if (cvg == 0 || !depth.x) continue;
      uint tex0 = sample_tex(stwz.xyz, 0, cmd), tex1 = 0;
      if (cmd.modes[0] & M0_2CYCLE) {
        tex1 = sample_tex(stwz.xyz, 1, cmd);
      }
      uint shade = sample_shade(dxy, cmd);
      uint2 color = combine(tex0, tex1, shade, noise_, 0, cmd, 0);
      uint alpha = unpacka(color.y) ? 0 : unpacka(color.x);
      if (cmd.modes[0] & M0_2CYCLE) {
        color = combine(tex1, tex1, shade, noise_, color, cmd, 1);
      }
      //color.x = pack32(unpack32(color.y) ? 0xff : unpack32(color.x));
      if (cmd.modes[0] & M0_2CYCLE) {
        color.x = blend(pixel, color.x, shade, alpha, zmem, oz, cvg, depth.y, cmd, 0);
        cmd.modes[0] &= ~M0_2CYCLE;
        pixel = blend(pixel, color.x, shade, alpha, zmem, oz, cvg, depth.y, cmd, 1);
      } else {
        pixel = blend(pixel, color.x, shade, alpha, zmem, oz, cvg, depth.y, cmd, 0);
      }
    }
  }
  
  zbuf[pos] = (uint16_t)packz(zmem);
  pix16[pos] = write_rgba16(pixel);
}

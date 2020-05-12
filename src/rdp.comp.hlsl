/* === Bitfield Flags === */

#define M0_FILL      0x00300000
#define M0_COPY      0x00200000
#define M0_2CYCLE    0x00100000
#define M0_PERSP     0x00080000
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
  uint env, prim, zprim;
  uint keys, keyc, pad2;
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

[[vk::binding(4)]] ByteAddressBuffer tmem;
[[vk::binding(5)]] RWByteAddressBuffer pixels;
[[vk::binding(6)]] RWByteAddressBuffer zbuf;

/* === Utility Functions === */

uint read_rgba16(uint enc) {
  // bswapped RGBA5551 to ABGR8888
  enc = ((enc << 8) | (enc >> 8)) & 0xffff;
  uint dec = -(enc & 0x1) & (0xff << 24);
  dec |= (enc & 0x3e) << 18;
  dec |= (enc & 0x7c0) << 5;
  return dec | ((enc & 0xf800) >> 8);
}

uint write_rgba16(uint dec) {
  // ABGR8888 to bswapped RGBA5551
  uint enc = (dec & (0xff << 24)) != 0;
  enc |= (dec >> 18) & 0x3e;
  enc |= (dec >> 5) & 0x7c0;
  enc |= (dec << 8) & 0xf800;
  return ((enc << 8) | (enc >> 8)) & 0xffff;
}

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
    case 0: return inputs.x;
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
  uint4 c = min(color >> 16, 0xff);
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
  uint dz =  (enc & 0x3) << 2, mant = (enc >> 2) & 0x7ff;
  uint2 val = globals[0].zdec[enc >> 13];
  uint dec = (mant << val.x) + val.y;
  return ((dec & 0x3ffff) << 4) | dz;
}

uint packz(uint dec) {
  uint dz = (dec >> 2) & 0x3;
  uint2 val = globals[0].zenc[(dec >> 15) & 0x7f];
  uint mant = (dec >> val.x) & 0x7ff;
  uint enc = (val.y << 13) | (mant << 2) | dz;
  return ((enc << 8) | (enc >> 8)) & 0xffff;
}

/* === Pipeline Stages === */

uint read_texel(uint2 st, uint2 mask, RDPTex tex, RDPCommand cmd) {
  bool2 mst = tex.shift & 0x100, en = (mask != 0);
  st ^= -(en & (st >> mask) & mst), st &= (en << mask) - 1;
  uint s = st.x, t = st.y;
  tex.addr = (tex.addr & 0xfff) + (cmd.tmem << 12);
  cmd.tlut = (cmd.tlut & 0xfff) + (cmd.tmem << 12);
  if (tex.format == 0 && tex.size == 2) {        // 16 bit RGBA
    if (t & 0x1) s ^= 0x2;
    uint tex_pos = tex.addr + t * tex.width + s * 2;
    uint texel = tmem.Load(tex_pos);
    return read_rgba16(s & 0x1 ? texel >> 16 : texel & 0xffff);
  } else if (tex.format == 0 && tex.size == 3) { // 32 bit RGBA
    if (t & 0x1) s ^= 0x2;
    uint tex_pos = tex.addr + t * tex.width + s * 2;
    tex_pos = (tex.addr & ~0x7ff) | (tex_pos & 0x7ff);
    uint texel1 = tmem.Load(tex_pos + 0x800), texel2 = tmem.Load(tex_pos);
    texel1 = s & 0x1 ? texel1 & ~0xffff : texel1 << 16;
    texel2 = s & 0x1 ? texel2 >> 16 : texel2 & 0xffff;
    return texel1 | texel2;
  } else if (tex.format == 2 && tex.size == 1) { // 8 bit CI
    if (t & 0x1) s ^= 0x4;
    uint tex_pos = tex.addr + t * tex.width + s;
    uint idx = (tmem.Load(tex_pos) >> ((s & 0x3) * 8)) & 0xff;
    uint texel = tmem.Load(cmd.tlut + idx * 2);
    texel = idx & 0x1 ? texel >> 16 : texel & 0xffff;
    if (cmd.modes[0] & M0_TLUT_IA) {
      uint i = texel & 0xff, a = texel >> 8;
      return (a << 24) | (i << 16) | (i << 8) | i;
    } else return read_rgba16(texel);
  } else if (tex.format == 2 && tex.size == 0) { // 4 bit CI
    if (t & 0x1) s ^= 0x8;
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint idx = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    idx = (s & 0x1 ? idx : idx >> 4) & 0xf;
    uint texel = tmem.Load(cmd.tlut + (tex.pal << 5) + idx * 2);
    texel = idx & 0x1 ? texel >> 16 : texel & 0xffff;
    if (cmd.modes[0] & M0_TLUT_IA) {
      uint i = texel & 0xff, a = texel >> 8;
      return (a << 24) | (i << 16) | (i << 8) | i;
    } else return read_rgba16(texel);
  } else if (tex.format == 3 && tex.size == 2) { // 16 bit IA
    if (t & 0x1) s ^= 0x2;
    uint texel = tmem.Load(tex.addr + t * tex.width + s * 2);
    texel = s & 0x1 ? texel >> 16 : texel & 0xffff;
    uint i = texel & 0xff, a = texel >> 8;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 1) { // 8 bit IA
    if (t & 0x1) s ^= 0x4;
    uint tex_pos = tex.addr + t * tex.width + s;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x3) * 8);
    uint i = texel & 0xf0, a = (texel << 4) & 0xf0;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 0) { // 4 bit IA
    if (t & 0x1) s ^= 0x8;
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    texel = (s & 0x1 ? texel : texel >> 4) & 0xf;
    uint i = (texel & 0xe) << 4, a = (texel & 0x1) * 0xff;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 4 && tex.size == 1) { // 8 bit I
    if (t & 0x1) s ^= 0x4;
    uint tex_pos = tex.addr + t * tex.width + s;
    uint i = (tmem.Load(tex_pos) >> ((s & 0x3) * 8)) & 0xff;
    return (i << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 4 && tex.size == 0) { // 4 bit I
    if (t & 0x1) s ^= 0x8;
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    uint i = (s & 0x1 ? texel << 4 : texel) & 0xf0;
    return (i << 24) | (i << 16) | (i << 8) | i;
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
  int4 x2 = y < int4(cmd.ym) ? xm : xl;
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
  stwz = cmd.tex + cmd.tde * dxy.y / 4;
  stwz = stwz + cmd.tdx * dxy.x / 4;
  // read old z, dz from zbuf
  bool skip = (~cmd.type & T_ZBUF) || (~cmd.modes[1] & M1_ZCMP);
  skip = skip || (cmd.modes[0] & M0_COPY);
  if (cvg == 0 || skip) return bool2(true);
  uint oz = zmem >> 4, odz = zmem & 0xf;
  bool zmax = (oz == 0x3ffff);
  // calculate new z from slopes
  uint nz = (cmd.zprim >> 14) & 0x3fffc;
  if (~cmd.modes[1] & M1_ZSRC) nz = stwz.w >> 14;
  // get new dz from prim_z
  uint ndz = firstbithigh(max(cmd.zprim & 0xffff, 1));
  uint dz = 0x1 << max(ndz, odz);
  //uint dz = 0x1 << ndz;
  uint mode = cmd.modes[1] & M1_ZMODE;
  if (mode == ZTRANS || cvg > 7) dz = 0;
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

uint sample_tex(int3 stw, RDPCommand cmd) {
  if (~cmd.type & T_TEX) return 0;
  uint mode = cmd.modes[0]; stw >>= 16;
  RDPTex tex = texes[cmd.tmem * 8 + cmd.tile];
  // correct perspective, apply shift
  if (mode & M0_PERSP) stw = 32768.0 * stw / stw.z;
  int2 shl = tex.shift & 0xf, st = stw.xy;
  st = shl > 10 ? st << (16 - shl) : st >> shl;
  // get mask, clamp coords to tex bounds
  uint2 mask = min((tex.shift >> 4) & 0xf, 10);
  bool2 cst = (tex.shift >> 9) || (mask == 0);
  int2 stl = tex.stl << 3, sth = tex.sth << 3;
  st = cst ? clamp(st - stl, 0, sth - stl) : st - stl;
  return read_texel(st >> 5, mask, tex, cmd);
}

uint sample_shade(int2 dxy, RDPCommand cmd) {
  if (~cmd.type & T_SHADE) return 0;
  int4 rgba = cmd.shade + cmd.sde * dxy.y / 4;
  return pack32(max(rgba + cmd.sdx * dxy.x / 4, 0));
}

uint combine(uint tex, uint shade, uint noise_, RDPCommand cmd) {
  uint cycle = cmd.modes[0] & M0_FILL;
  if (cycle == M0_COPY) return tex;
  if (cycle == M0_FILL) return read_rgba16(cmd.fill);
  // read mux values
  uint mux0 = cmd.mux[0], mux1 = cmd.mux[1];
  uint mc = (mux0 >> 15) & 0x1f;
  // setup input options
  uint4 i1 = uint4(0, tex, tex, cmd.prim);
  uint4 i2 = uint4(shade, cmd.env, -1, 0);
  uint4 i3 = uint4(tex, tex, cmd.prim, shade);
  uint4 ia2 = uint4(shade, cmd.env, -1, noise_);
  uint4 ib2 = uint4(shade, cmd.env, cmd.keyc, 0);
  uint4 ic2 = uint4(shade, cmd.env, cmd.keys, 0);
  // select packed inputs
  uint pa = sel((mux0 >> 20) & 0xf, i1, ia2);
  uint pb = sel((mux1 >> 28) & 0xf, i1, ib2);
  uint pc = sel(mc, i1, ic2, i3, uint4(cmd.env, 0, 0, 0));
  uint pd = sel((mux1 >> 15) & 0x7, i1, i2);
  // unpack input channels
  uint4 a = unpack32(pa), b = unpack32(pb);
  uint4 c = unpack32(pc), d = unpack32(pd);
  if (mc >= 7) c = c.wwww;
  // select alpha inputs
  a.w = unpacka(sel((mux0 >> 12) & 0x7, i1, i2));
  b.w = unpacka(sel((mux1 >> 12) & 0x7, i1, i2));
  c.w = unpacka(sel((mux0 >> 9) & 0x7, i1, i2));
  d.w = unpacka(sel((mux1 >> 9) & 0x7, i1, i2));
  // combine selected inputs
  return pack32((a - b) * (c >> 16) / 256 + d);
}

uint blend(uint pixel, uint color, inout uint zmem, uint oz, uint cvg, bool far, RDPCommand cmd) {
  // multiply 3-bit coverage and 5-bit alpha
  uint mux = cmd.modes[1], alpha = color >> 27, ocvg = pixel >> 29;
  if (mux & M1_ALPHA2CVG) cvg = (alpha * cvg + 4) >> 5;
  if (mux & M1_CVG2ALPHA) alpha = cvg << 2;
  if (cvg == 0) { zmem = oz; return pixel; }
  if ((mux & M1_ALPHA) && alpha < (cmd.blend >> 27)) { zmem = oz; return pixel; }
  // select blender inputs
  uint p = sel(mux >> 30, uint4(color, pixel, cmd.blend, cmd.fog));
  uint a = sel(mux >> 26, uint4(alpha, cmd.fog >> 27, alpha, 0x0));
  uint m = sel(mux >> 22, uint4(color, pixel, cmd.blend, cmd.fog));
  uint b = sel(mux >> 18, uint4(a ^ 0x1f, pixel >> 29, 0x1f, 0x0)) + 1;
  // skip inputs based on flags
  bool copy = cmd.modes[0] & M0_COPY;
  bool on_cvg = mux & M1_ON_CVG, force = mux & M1_BLEND;
  bool full = cvg + ocvg > 7, aa = mux & M1_AA;
  //if (copy) return (full && alpha ? color : pixel);
  //if (on_cvg && !full) a = 0, b = 0x1f;
  //if (!force && (full || !far || !aa)) a = 0x1f, b = ocvg = 0;
  // blend selected inputs
  int c1 = clamp(cvg + ocvg, 0, 7), c2 = (cvg + ocvg) & 7;
  //if (!force && (full || !far || !aa)) c1 = (cvg + ocvg - 1) & 7;
  uint output = sel(mux >> 8, uint4(c1, c2, 7, ocvg)) << 29;
  uint4 p4 = unpack32(p), m4 = unpack32(m);
  uint4 res = (a * p4 + b * m4) / (a + b);
  return (pack32(res) & 0xffffff) | output;
}

[numthreads(8, 8, 1)]
void main(uint3 GlobalID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID) {
  GlobalData global = globals[0];
  if (GlobalID.x >= global.width) return;
  uint tile_pos = GlobalID.y * global.width + GlobalID.x;
  uint pixel = pixels.Load(tile_pos * global.size), zmem = zbuf.Load(tile_pos * 2);
  zmem = unpackz(tile_pos & 0x1 ? zmem >> 16 : zmem & 0xffff);
  if (global.size == 2)
    pixel = read_rgba16(tile_pos & 0x1 ? pixel >> 16 : pixel & 0xffff);
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
      uint tex = sample_tex(stwz, cmd);
      uint shade = sample_shade(dxy, cmd);
      uint color = combine(tex, shade, noise_, cmd);
      pixel = blend(pixel, color, zmem, oz, cvg, depth.y, cmd);
      //pixel = color;
    }
  }
  
  if (tile_pos & 0x1) {
    zbuf.InterlockedAnd(tile_pos * 2, 0x0000ffff);
    zbuf.InterlockedOr(tile_pos * 2, packz(zmem) << 16);
  } else {
    zbuf.InterlockedAnd(tile_pos * 2, 0xffff0000);
    zbuf.InterlockedOr(tile_pos * 2, packz(zmem));
  }
  if (global.size == 2) {
    if (tile_pos & 0x1) {
      pixels.InterlockedAnd(tile_pos * 2, 0x0000ffff);
      pixels.InterlockedOr(tile_pos * 2, write_rgba16(pixel) << 16);
    } else {
      pixels.InterlockedAnd(tile_pos * 2, 0xffff0000);
      pixels.InterlockedOr(tile_pos * 2, write_rgba16(pixel));
    }
  } else pixels.Store(tile_pos * 4, pixel);
}

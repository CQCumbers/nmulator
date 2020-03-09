struct PerTileData {
  uint n_cmds;
  uint cmd_idxs[31];
};

struct RDPCommand {
  int xyh[2], xym[2], xyl[2];
  int sh, sm, sl;
  int shade[4], sde[4], sdx[4];
  uint zpos, zde, zdx;
  int tex[3], tde[3], tdx[3];
  uint fill, fog, blend;
  uint env, prim, zprim;
  uint bl_mux, cc_mux, tlut;
  uint keys[3];
  uint lft, type, tile;
  uint two_cycle, zsrc, tlut_ia;
};

struct RDPTile {
  uint format, size;
  uint width, addr, pal;
  uint mask[2], shift[2];
};

struct GlobalData {
  uint width, size;
};

[[vk::binding(0)]] StructuredBuffer<RDPCommand> cmds;
[[vk::binding(1)]] StructuredBuffer<PerTileData> tiles;
[[vk::binding(2)]] StructuredBuffer<RDPTile> texes;
[[vk::binding(3)]] ConstantBuffer<GlobalData> global;

[[vk::binding(4)]] ByteAddressBuffer tmem;
[[vk::binding(5)]] RWByteAddressBuffer pixels;
[[vk::binding(6)]] RWByteAddressBuffer zbuf;

uint read_rgba16(uint input) {
  // GBARG25153 to ABGR8888
  uint output = ((input >> 3) & 0x1f) << 3;
  output |= ((input & 0x7) << 2 | ((input >> 14) & 0x3)) << 11;
  output |= ((input >> 9) & 0x1f) << 19;
  return output | (((input >> 8) & 0x1) * 0xff) << 24;
}

uint write_rgba16(uint input) {
  // ABGR8888 to GBARG25153
  uint output = ((input >> 3) & 0x1f) << 3;
  output |= ((input >> 13) & 0x7) | (((input >> 11) & 0x3) << 14);
  output |= ((input >> 19) & 0x1f) << 9;
  return output | (((input >> 31) & 0x1) << 8);
}

uint read_texel(RDPTile tex, RDPCommand cmd, uint s, uint t) {
  s >>= 21, t >>= 21; // convert 16.16 texture coords to texels
  uint ms = s & (1 << tex.mask[0]), mt = t & (1 << tex.mask[1]);
  if (tex.mask[0]) { s &= ((1 << tex.mask[0]) - 1); if (ms) s = ((1 << tex.mask[0]) - 1) - s; }
  if (tex.mask[1]) { t &= ((1 << tex.mask[1]) - 1); if (mt) t = ((1 << tex.mask[1]) - 1) - t; }
  if (tex.shift[0] < 10) { s >>= tex.shift[0]; } else { s <<= (16 - tex.shift[0]); }
  if (tex.shift[1] < 10) { t >>= tex.shift[1]; } else { t <<= (16 - tex.shift[1]); }
  
  if (tex.format == 0 && tex.size == 2) {        // 16 bit RGBA
    if (s * 2 >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s * 2;
    uint texel = tmem.Load(tex_pos);
    return read_rgba16(s & 0x1 ? texel >> 16 : texel & 0xffff);
  } else if (tex.format == 0 && tex.size == 3) { // 32 bit RGBA
    if (s * 2 >= tex.width) return 0;
    uint tex_pos = min(tex.addr + t * tex.width + s * 2, 0x7ff);
    uint texel1 = tmem.Load(tex_pos + 0x800), texel2 = tmem.Load(tex_pos);
    texel1 = s & 0x1 ? texel1 & ~0xffff : texel1 << 16;
    texel2 = s & 0x1 ? texel2 >> 16 : texel2 & 0xffff;
    return texel1 | texel2;
  } else if (tex.format == 2 && tex.size == 1) { // 8 bit CI
    if (s >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s;
    uint idx = (tmem.Load(tex_pos) >> ((s & 0x3) * 8)) & 0xff;
    uint texel = tmem.Load(cmd.tlut + idx * 2);
    texel = idx & 0x1 ? texel >> 16 : texel & 0xffff;
    if (cmd.tlut_ia) {
      uint i = texel & 0xff, a = texel >> 8;
      return (a << 24) | (i << 16) | (i << 8) | i;
    } else return read_rgba16(texel);
  } else if (tex.format == 2 && tex.size == 0) { // 4 bit CI
    if (s / 2 >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint idx = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    idx = (s & 0x1 ? idx : idx >> 4) & 0xf;
    uint texel = tmem.Load(cmd.tlut + (tex.pal << 5) + idx * 2);
    texel = idx & 0x1 ? texel >> 16 : texel & 0xffff;
    if (cmd.tlut_ia) {
      uint i = texel & 0xff, a = texel >> 8;
      return (a << 24) | (i << 16) | (i << 8) | i;
    } else return read_rgba16(texel);
  } else if (tex.format == 3 && tex.size == 2) { // 16 bit IA
    if (s * 2 >= tex.width) return 0;
    uint texel = tmem.Load(tex.addr + t * tex.width + s * 2);
    texel = s & 0x1 ? texel >> 16 : texel & 0xffff;
    uint i = texel & 0xff, a = texel >> 8;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 1) { // 8 bit IA
    if (s >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x3) * 8);
    uint i = texel & 0xf0, a = (texel << 4) & 0xf0;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 0) { // 4 bit IA
    if (s / 2 >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    texel = (s & 0x1 ? texel : texel >> 4) & 0xf;
    uint i = (texel & 0xe) << 4, a = (texel & 0x1) * 0xff;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 4 && tex.size == 1) { // 8 bit I
    if (s >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s;
    uint i = (tmem.Load(tex_pos) >> ((s & 0x3) * 8)) & 0xff;
    return (i << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 4 && tex.size == 0) { // 4 bit I
    if (s / 2 >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    uint i = (s & 0x1 ? texel << 4 : texel) & 0xf0;
    return (i << 24) | (i << 16) | (i << 8) | i;
  }
  return 0xff00ffff; //~0x0;
}

int sadd(int a, int b) {
  return b == 0 ? a : a + max(b, -a);
  //uint val = a + b, sat = (a >> 31) + 0x7fffffff;
  //return ((int)((a ^ b) | ~(b ^ val)) >= 0) ? sat : val;
  //return val | -(int)(val < a);
}

int mul16(int a, int b) {
   // multiply 16.16 ints, with 16.16 result
   int a1 = a >> 16, a2 = a & 0xffff;
   int b1 = b >> 16, b2 = b & 0xffff;
   return (a1 * b1 << 16) + a1 * b2 + a2 * b1;
}

uint sample_color(uint2 pos, RDPCommand cmd) {
  if (cmd.type & 0x2) { // textured
    RDPTile tex = texes[cmd.tile]; uint s, t, w;
    int x = pos.x << 16, dy = (pos.y << 2) - cmd.xyh[1];
    if (cmd.type & 0x8) { // rectangle
      s = cmd.tex[0] + mul16(cmd.tde[0], x - cmd.xyh[0]);
      t = cmd.tex[1] + (cmd.tde[1] * dy >> 2);
    } else { // triangle
      int x1 = cmd.xyh[0] + (cmd.sh * dy >> 2);
      s = cmd.tex[0] + (cmd.tde[0] * dy >> 2) + mul16(cmd.tdx[0], x - x1);
      t = cmd.tex[1] + (cmd.tde[1] * dy >> 2) + mul16(cmd.tdx[1], x - x1);
      w = cmd.tex[2] + (cmd.tde[2] * dy >> 2) + mul16(cmd.tdx[2], x - x1);
    }
    if (cmd.type == 0xb) return read_texel(tex, cmd, t, s);
    else return read_texel(tex, cmd, s, t);
  } else if (cmd.type & 0x4) { // shaded
    // convert to subpixels, calculate major edge
    int x = pos.x << 16, y = pos.y << 2;
    int dy = y - cmd.xyh[1]; uint color = 0x0;
    int x1 = cmd.xyh[0] + (cmd.sh * dy >> 2);
    // interpolate shading along major edge and line
    for (uint i = 0; i < 4; ++i) {
      int chan = sadd(cmd.shade[i], cmd.sde[i] * (dy + 1) >> 2);
      chan = sadd(chan, mul16(cmd.sdx[i], x - x1));
      color |= ((chan >> 16) & 0xff) << (i * 8);
    }
    return color;
  }
  return global.size == 4 ? cmd.fill : read_rgba16(cmd.fill);
}

uint sample_z(uint2 pos, RDPCommand cmd) {
  if (~cmd.type & 0x1) return 0x7fff;
  uint y = pos.y - (cmd.xyh[1] >> 2);
  uint x = pos.x - ((cmd.xyh[0] + cmd.sh * y) >> 16);
  if (cmd.zsrc) return cmd.zprim >> 16;
  uint z = cmd.zpos, z1 = x * cmd.zdx, z2 = y * cmd.zde;
  if (z1 != 0) z += max(z1, -z);
  if (z2 != 0) z += max(z2, -z);
  return z >> 16;
}

uint visible(uint2 pos, RDPCommand cmd) {
  // convert to subpixels, check y bounds
  int x = pos.x << 16, y = pos.y << 2;
  int y1 = cmd.xyh[1] & ~0x3, y2 = cmd.xyl[1] & ~0x3;
  if (!(y1 <= y && y <= y2 + 0x3)) return 0;
  if (cmd.type & 0x8) return cmd.xyh[0] <= x && x <= cmd.xyl[0];
  // calculate x bounds from slopes
  int dy = y + 2 - cmd.xyh[1], x2 = 0;
  int x1 = cmd.xyh[0] + (cmd.sh * dy >> 2);
  if (y < cmd.xym[1]) x2 = cmd.xym[0] + (cmd.sm * dy >> 2);
  else x2 = cmd.xyl[0] + (cmd.sl * (y + 2 - cmd.xym[1]) >> 2);
  x1 &= 0xffff0000, x2 &= 0xffff0000;
  return cmd.lft ? (x1 <= x && x <= x2 + 0xffff) : (x2 <= x && x <= x1 + 0xffff);
}

/*uint combine(RDPCommand cmd) {
  uint a, b, c, d;
  uint ma = (cmd.cc_mux >> 14) & 0x3, mb = (cmd.cc_mux >> 10) & 0x3;
  uint mc = (cmd.cc_mux >> 6) & 0x3, md = (cmd.cc_mux >> 2) & 0x3;
  // select a
  if (ma == 1) a = tex_out;
  else if (ma == 3) a = cmd.prim;
  else if (ma == 4) a = shade_out;
  else if (ma == 5) a = cmd.env;
  else if (ma == 6) a = 0xff;
  else if (ma >= 8) a = 0x0;
  // select b
  if (mb == 1) b = tex_out;
  else if (mb == 3) b = cmd.prim;
  else if (mb == 4) b = shade_out;
  else if (mb == 5) b = cmd.env;
  else if (mb == 6) b = cmd.key_center;
  else if (mb == 8) b = 0x0;
  // select c
  if (mc == 0) c = tex_out;
  else if (mc == 3) c = cmd.prim;
  else if (mc == 4) c = shade_out;
  else if (mc == 5) c = cmd.env;
  else if (mc == 6) c = cmd.key_scale;
  else if (mc == 8) c = tex_out >> 24;
  else if (mc == 10) c = cmd.prim >> 24;
  else if (mc == 11) c = shade_out >> 24;
  else if (mc == 10) c = cmd.prim >> 24;
  // select d
  if (md == 0) d = color;
  else if (md == 3) d = cmd.prim;
  else if (md == 4) d = shaded;
  else if (md == 5) d = cmd.env;
  else if (md == 6) d = 0xff;
  // combine selected sources
  uint output = 0x0;
  for (uint i = 0; i < 4; ++i) {
    uint pc = (p >> (i * 8)) & 0xff;
    uint mc = (m >> (i * 8)) & 0xff;
    uint oc = (a * pc + b * mc) / (a + b);
    output |= (oc & 0xff) << (i * 8);
  }
  return output;
}*/

uint blend(uint pixel, uint color, uint coverage, RDPCommand cmd) {
  uint a, p, b, m;
  uint m1a = (cmd.bl_mux >> 14) & 0x3, m1b = (cmd.bl_mux >> 10) & 0x3;
  uint m2a = (cmd.bl_mux >> 6) & 0x3, m2b = (cmd.bl_mux >> 2) & 0x3;
  // select p 
  if (m1a == 0) p = color;
  else if (m1a == 1) p = pixel;
  else if (m1a == 2) p = cmd.blend;
  else if (m1a == 3) p = cmd.fog;
  // select m
  if (m2a == 0) m = color;
  else if (m2a == 1) m = pixel;
  else if (m2a == 2) m = cmd.blend;
  else if (m2a == 3) m = cmd.fog;
  // select a
  if (m1b == 0) a = (color >> 24) & 0xff;
  else if (m1b == 1) a = (cmd.fog >> 24) & 0xff;
  else if (m1b == 2) a = (color >> 24) & 0xff; // should be shade, from before CC?
  else if (m1b == 3) a = 0x0;
  // select b
  if (m2b == 0) b = 0xff - a;
  else if (m2b == 1) b = (pixel >> 24) & 0xff;
  else if (m2b == 2) b = 0xff;
  else if (m2b == 3) b = 0x0;
  // blend selected colors
  uint output = 0xff000000;
  for (uint i = 0; i < 3; ++i) {
    uint pc = (p >> (i * 8)) & 0xff;
    uint mc = (m >> (i * 8)) & 0xff;
    uint oc = (a * pc + b * mc) / (a + b);
    output |= min(oc, 0xff) << (i * 8);
  }
  return output;
}

[numthreads(8, 8, 1)]
void main(uint3 GlobalID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID) {
  if (GlobalID.x >= global.width) return;
  uint tile_pos = GlobalID.y * global.width + GlobalID.x;

  uint pixel = pixels.Load(tile_pos * global.size);
  if (global.size == 2) pixel = read_rgba16(tile_pos & 0x1 ? pixel >> 16 : pixel);
  uint zval = 0x7fff;
  //uint zval = zbuf.Load(tile_pos * global.size);
  //zval = (tile_pos & 0x1 ? zval >> 16 : zval) & 0x7fff;

  PerTileData tile = tiles[GroupID.y * (global.width / 8) + GroupID.x];
  for (uint i = 0; i < tile.n_cmds; ++i) {
    RDPCommand cmd = cmds[tile.cmd_idxs[i]];
    uint coverage = visible(GlobalID.xy, cmd);
    uint color = sample_color(GlobalID.xy, cmd);
    uint z = sample_z(GlobalID.xy, cmd);
    if (coverage == 0 || z > zval) continue;
    pixel = blend(pixel, color, coverage, cmd);
    if (cmd.type & 0x1) zval = z;
  }

  if (tile_pos & 0x1) {
    if (global.size == 2) {
      pixels.InterlockedAnd(tile_pos * global.size, 0x0000ffff);
      pixels.InterlockedOr(tile_pos * global.size, write_rgba16(pixel) << 16);
    } else pixels.Store(tile_pos * global.size, pixel);
    //zbuf.InterlockedAnd(tile_pos * global.size, 0x0000ffff);
    //zbuf.InterlockedOr(tile_pos * global.size, (zval & 0xffff) << 16);
  } else {
    if (global.size == 2) {
      pixels.InterlockedAnd(tile_pos * global.size, 0xffff0000);
      pixels.InterlockedOr(tile_pos * global.size, write_rgba16(pixel));
    } else pixels.Store(tile_pos * global.size, pixel);
    //zbuf.InterlockedAnd(tile_pos * global.size, 0xffff0000);
    //zbuf.InterlockedOr(tile_pos * global.size, zval & 0xffff);
  }
}

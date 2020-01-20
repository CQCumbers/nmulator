struct PerTileData {
  uint n_cmds;
  uint cmd_idxs[31];
};

struct RDPCommand {
  uint xyh[2], xym[2], xyl[2];
  uint sh, sm, sl;
  uint shade[4], sde[4], sdx[4];
  uint zpos, zde, zdx;
  uint tex[2], tde[2], tdx[2];
  uint fill, fog, blend;
  uint env, prim, zprim;
  uint bl_mux, cc_mux, keys[3];
  uint lft, type, tile, two_cycle;
};

struct RDPTile {
  uint format, size;
  uint width, addr;
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

uint read_texel(RDPTile tex, uint s, uint t) {
  uint ms = s & (1 << tex.mask[0]), mt = t & (1 << tex.mask[1]);
  if (tex.mask[0]) { s &= ((1 << tex.mask[0]) - 1); if (ms) s = ((1 << tex.mask[0]) - 1) - s; }
  if (tex.mask[1]) { t &= ((1 << tex.mask[1]) - 1); if (mt) t = ((1 << tex.mask[1]) - 1) - t; }
  if (tex.shift[0] < 10) { s >>= tex.shift[0]; } else { s <<= (16 - tex.shift[0]); }
  if (tex.shift[1] < 10) { t >>= tex.shift[1]; } else { t <<= (16 - tex.shift[1]); }
  
  if (tex.format == 0 && tex.size == 2) { // 16 bit RGBA
    uint tex_pos = tex.addr + t * tex.width + s * 2;
    uint texel = tmem.Load(tex_pos);
    return read_rgba16(tex_pos & 0x2 ? texel >> 16 : texel & 0xffff);
  } else if (tex.format == 0 && tex.size == 3) { // 32 bit RGBA
    uint texel = tmem.Load(tex.addr + t * tex.width + s * 4);
    //uint texel2 = tmem.Load(tex.addr + s * 4 + 2);// + t * tex.width + s * 2);
    return texel; //((texel2 & 0xffff) << 16) | (texel2 & 0xffff);
  } else if (tex.format == 3 && tex.size == 2) { // 16 bit IA
    uint texel = tmem.Load(tex.addr + t * tex.width + s * 2);
    uint i = (texel >> 8) & 0xff, a = texel & 0xff;
    return (i << 24) | (i << 16) | (i << 8) | a;
  }
  return ~0x0;
}

uint sample_color(uint2 pos, RDPCommand cmd) {
  if (cmd.type & 0x2) { // textured
    RDPTile tex = texes[cmd.tile]; uint x, y, s, t;
    if (cmd.type & 0x8) { // rectangle
      s = pos.x - (cmd.xyh[0] >> 2); //s = (s * cmd.tdx[0]) >> 12;
      t = pos.y - (cmd.xyh[1] >> 2); //t = (t * cmd.tde[1]) >> 10;
    } else { // triangle
      uint x1 = cmd.xyh[0] + cmd.sh * (pos.y - (cmd.xyh[1] >> 2));
      x = pos.x - (x1 >> 16); y = pos.y - (cmd.xyh[1] >> 2);
      s = (x * cmd.tdx[0] + y * cmd.tde[0] + cmd.tex[0]) >> 21;
      t = (x * cmd.tdx[1] + y * cmd.tde[1] + cmd.tex[1]) >> 21;
    }
    return cmd.type == 0xb ? read_texel(tex, t, s) : read_texel(tex, s, t);
  } else if (cmd.type & 0x4) { // shaded
    // convert to subpixel, calc major edge
    uint x = pos.x << 16, y = pos.y << 2, color = 0x0;
    uint x1 = cmd.xyh[0] + cmd.sh * ((y - cmd.xyh[1]) >> 2);
    // add d(s)de and d(s)dx to inital shade, convert to RGBA8888
    for (uint i = 0; i < 4; ++i) {
      uint channel = cmd.shade[i] + cmd.sde[i];
      channel += cmd.sde[i] * ((y - cmd.xyh[1]) >> 2);
      uint c2 = cmd.sdx[i] * ((x - x1) >> 16);
      if (c2 != 0) channel += max(c2, -channel);
      color |= ((channel >> 16) & 0xff) << (i * 8);
    }
    return color;
  }
  return global.size == 4 ? cmd.fill : read_rgba16(cmd.fill);
}

uint sample_z(uint2 pos, RDPCommand cmd) {
  if (~cmd.type & 0x1) return 0;
  uint x1 = cmd.xyh[0] + cmd.sh * (pos.y - (cmd.xyh[1] >> 2));
  uint x = pos.x - (x1 >> 16), y = pos.y - (cmd.xyh[1] >> 2);
  uint output = cmd.zpos, o1 = x * cmd.zdx, o2 = y * cmd.zde;
  if (o1 != 0) output -= max(o1, -output);
  if (o2 != 0) output -= max(o2, -output);
  return output >> 16;
}

uint visible(uint2 pos, RDPCommand cmd) {
  // convert to subpixels, check y bounds
  uint x = pos.x << 16, y = pos.y << 2;
  if (cmd.xyh[1] > y || y > cmd.xyl[1]) return 0;
  if (cmd.type & 0x8) // rectangles
    return (cmd.xyh[0] <= (pos.x << 2) && (pos.x << 2) <= cmd.xyl[0]);
  // calculate x bounds from slopes
  uint x1 = cmd.xyh[0] + cmd.sh * ((y - cmd.xyh[1]) >> 2), x2;
  if (y < cmd.xym[1]) x2 = cmd.xym[0] + cmd.sm * ((y - cmd.xyh[1]) >> 2);
  else x2 = cmd.xyl[0] + cmd.sl * ((y - cmd.xym[1]) >> 2);
  return ((cmd.lft && x1 <= x && x <= x2) ||
         (!cmd.lft && x2 <= x && x <= x1));
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
  uint output = 0x0;
  for (uint i = 0; i < 4; ++i) {
    uint pc = (p >> (i * 8)) & 0xff;
    uint mc = (m >> (i * 8)) & 0xff;
    uint oc = (a * pc + b * mc) / (a + b);
    output |= (oc & 0xff) << (i * 8);
  }
  return output;
}

[numthreads(8, 8, 1)]
void main(uint3 GlobalID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID) {
  if (GlobalID.x >= global.width) return;
  uint tile_pos = GlobalID.y * global.width + GlobalID.x;
  uint pixel = pixels.Load(tile_pos * global.size);
  if (global.size == 2) pixel = read_rgba16(tile_pos & 0x1 ? pixel >> 16 : pixel);
  //uint zval = zbuf.Load(tile_pos * global.size);
  PerTileData tile = tiles[GroupID.y * (global.width / 8) + GroupID.x];
  for (uint i = 0; i < tile.n_cmds; ++i) {
    RDPCommand cmd = cmds[tile.cmd_idxs[i]];
    uint coverage = visible(GlobalID.xy, cmd);
    uint color = sample_color(GlobalID.xy, cmd);
    //uint z = sample_z(GlobalID.xy, cmd); coverage *= z < zval;
    if (coverage == 0) continue;
    pixel = blend(pixel, color, coverage, cmd);
    //if (cmd.type & 0x4) zval = z;
  }
  if (global.size == 4) pixels.Store(tile_pos * global.size, pixel);
  else if (tile_pos & 0x1) {
    pixels.InterlockedAnd(tile_pos * global.size, 0x0000ffff);
    pixels.InterlockedOr(tile_pos * global.size, write_rgba16(pixel) << 16);
    //zbuf.InterlockedAnd(tile_pos * global.size, 0x0000ffff);
    //zbuf.InterlockedOr(tile_pos * global.size, (zval & 0xffff) << 16);
  } else {
    pixels.InterlockedAnd(tile_pos * global.size, 0xffff0000);
    pixels.InterlockedOr(tile_pos * global.size, write_rgba16(pixel));
    //zbuf.InterlockedAnd(tile_pos * global.size, 0xffff0000);
    //zbuf.InterlockedOr(tile_pos * global.size, zval & 0xffff);
  }
}

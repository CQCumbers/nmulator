[[vk::binding(0)]] ByteAddressBuffer rdram;

float4 read_rgba16(uint enc) {
  // bswapped RGBA5551 to ABGR8888
  enc = ((enc << 8) | (enc >> 8)) & 0xffff;
  float4 dec;
  dec.a = float(enc & 0x1);
  dec.b = float((enc & 0x3e) >> 1) / 32;
  dec.g = float((enc & 0x7c0) >> 6) / 32;
  dec.r = float((enc & 0xf800) >> 11) / 32;
  return dec;
}

float4 main(float4 pos : SV_POSITION, float4 col : COLOR) : SV_TARGET {
  uint tile_pos = (pos.x - 0.5) * 320 + (pos.y - 0.5);
  uint pixel = rdram.Load(tile_pos * 2);
  return read_rgba16(pixel);
}

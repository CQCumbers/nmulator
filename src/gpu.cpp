uint32_t read, write, alloc;

extern struct RDPCmd;

uint8_t *alloc_bytes(uint8_t *buf, ) { // allocate space in buffer for next cmd
  if (alloc - write > MAX_BYTES) // submit commands
  while (alloc - read > MAX_BYTES) {
    vkWaitFences(readers[0].fence);
    memcpy(rdram + readers[0].img_addr, pixels[readers[0].idx]);
    read = readers[0].idx, readers.pop_front();
  } // don’t use memory being read by GPU
  return &buf[(alloc++) & MAX_BYTES];
}

// same for acquire_tmem
// may be hard to copy ringbuf to gpu if we use C++ features

// on render()
waitFence(region.fence);
Shader.cmd_start = write; // if binning on CPU, cmd_start and cmd_end not needed
Shader.cmd_end = alloc, write = alloc;
readers.push_back({.idx = write});
Read <= write <= alloc, alloc must not exceed read by > MAX_CMDS


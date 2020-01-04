#ifndef RDP_H
#define RDP_H

#include <vulkan/vulkan.h>
#include <vector>
#include "rdp.spv.array"

typedef struct PerTileData {
  uint32_t n_cmds;
  uint32_t cmd_idxs[31];
} tile_t;

typedef struct RDPCommand {
  uint32_t xyh[2], xym[2], xyl[2];
  uint32_t sh, sm, sl;
  uint32_t shade[4], sde[4], sdx[4];
  uint32_t zpos, zde, zdx;
  uint32_t tex[2], tde[2], tdx[2];
  uint32_t fill, fog, blend;
  uint32_t env, prim, zprim;
  uint32_t bl_mux, cc_mux, keys[3];
  uint32_t lft, type, tile;
} cmd_t;

typedef struct RDPTile {
  uint32_t format, size;
  uint32_t width, addr;
  uint32_t mask[2], shift[2];
} tex_t;

typedef struct GlobalData {
  uint32_t width, size;
} global_t;

namespace Vulkan {
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queue_idx = 0;
  VkCommandBuffer commands = VK_NULL_HANDLE;

  /* === Descriptor Memory Access === */

  uint8_t *mapped_mem;
  const uint32_t group_size = 8, max_cmds = 64;
  uint32_t gwidth = 320 / group_size, gheight = 240 / group_size;
  uint32_t n_cmds = 0;

  const VkDeviceSize cmds_size = max_cmds * sizeof(cmd_t);
  cmd_t *cmds_ptr() { return reinterpret_cast<cmd_t*>(mapped_mem); }
  VkBuffer cmds = VK_NULL_HANDLE;

  const VkDeviceSize tiles_offset = cmds_size;
  const VkDeviceSize tiles_size = (320 / group_size) * (240 / group_size) * sizeof(tile_t);
  tile_t *tiles_ptr() { return reinterpret_cast<tile_t*>(mapped_mem + tiles_offset); }
  VkBuffer tiles = VK_NULL_HANDLE;

  const VkDeviceSize texes_offset = tiles_offset + tiles_size;
  const VkDeviceSize texes_size = 8 * sizeof(tex_t);
  tex_t *texes_ptr() { return reinterpret_cast<tex_t*>(mapped_mem + texes_offset); }
  VkBuffer texes = VK_NULL_HANDLE;

  const VkDeviceSize globals_offset = texes_offset + texes_size;
  const VkDeviceSize globals_size = sizeof(global_t);
  global_t *globals_ptr() { return reinterpret_cast<global_t*>(mapped_mem + globals_offset); }
  VkBuffer globals = VK_NULL_HANDLE;

  const VkDeviceSize tmem_offset = globals_offset + globals_size;
  const VkDeviceSize tmem_size = 0x1000;
  uint8_t *tmem_ptr() { return mapped_mem + tmem_offset; }
  VkBuffer tmem = VK_NULL_HANDLE;

  const VkDeviceSize pixels_offset = tmem_offset + tmem_size;
  const VkDeviceSize pixels_size = 320 * 240 * sizeof(uint32_t);
  uint8_t *pixels_ptr() { return mapped_mem + pixels_offset; }
  VkBuffer pixels = VK_NULL_HANDLE;

  const VkDeviceSize zbuf_offset = pixels_offset + pixels_size;
  const VkDeviceSize zbuf_size = 320 * 240 * sizeof(uint16_t);
  uint8_t *zbuf_ptr() { return mapped_mem + zbuf_offset; }
  VkBuffer zbuf = VK_NULL_HANDLE;
  const VkDeviceSize total_size = zbuf_offset + zbuf_size;

  /* === Vulkan Initialization == */

  void init_device(const VkInstance &instance, VkPhysicalDevice *gpu) {
    // check all vulkan physical devices
    uint32_t n_gpus = 0;
    vkEnumeratePhysicalDevices(instance, &n_gpus, 0);
    std::vector<VkPhysicalDevice> gpus(n_gpus);
    vkEnumeratePhysicalDevices(instance, &n_gpus, gpus.data());
    for (VkPhysicalDevice gpu_ : gpus) {
      // check queue families on each device
      uint32_t n_queues = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &n_queues, 0);
      std::vector<VkQueueFamilyProperties> queues(n_queues);
      vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &n_queues, &queues[0]);
      // if queue family supports compute, init virtual device
      for (uint32_t i = 0; i < n_queues; ++i) {
        if ((VK_QUEUE_COMPUTE_BIT & queues[i].queueFlags) == 0) continue;
        const float priorities = 1.0;
        const VkDeviceQueueCreateInfo queue_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = (queue_idx = i), .queueCount = 1,
          .pQueuePriorities = &priorities
        };
        const VkDeviceCreateInfo device_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
          .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info
        };
        vkCreateDevice((*gpu = gpu_), &device_info, 0, &device); return;
      }
    }
    printf("No compute queue found\n"); exit(1);
  }

  void init_pipeline(const uint32_t *code, uint32_t code_size,
      VkDescriptorSetLayout *desc_layout, VkPipelineLayout *layout, VkPipeline *pipeline) {
    // load shader code into module
    const VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code_size, .pCode = code
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    vkCreateShaderModule(device, &shader_info, 0, &shader);
    // describe needed descriptor set layout for shader
    const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}
    };
    const VkDescriptorSetLayoutCreateInfo desc_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 7, .pBindings = bindings
    };
    vkCreateDescriptorSetLayout(device, &desc_layout_info, 0, desc_layout);
    // create compute pipeline with shader and descriptor set layout
    const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = desc_layout
    };
    vkCreatePipelineLayout(device, &layout_info, 0, layout);
    const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shader, .pName = "main"
    };
    const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info, .layout = *layout
    };
    vkCreateComputePipelines(device, 0, 1, &pipeline_info, 0, pipeline);
  }

  void init_buffers(const VkPhysicalDevice &gpu, VkDeviceMemory *memory) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(gpu, &props);
    // find host visible, coherent memory of at least 'size' bytes
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      const VkMemoryType memoryType = props.memoryTypes[i];
      if (!(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & memoryType.propertyFlags) ||
          !(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & memoryType.propertyFlags) ||
          !(total_size < props.memoryHeaps[memoryType.heapIndex].size)) continue;
      const VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .memoryTypeIndex = i, .allocationSize = total_size
      };
      vkAllocateMemory(device, &allocate_info, 0, memory);
    }
    // exit if memory not allocated
    // create cmds buffer from allocated device memory
    VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = cmds_size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1, .pQueueFamilyIndices = &queue_idx
    };
    vkCreateBuffer(device, &buffer_info, 0, &cmds);
    vkBindBufferMemory(device, cmds, *memory, 0);
    // create tiles buffer
    buffer_info.size = tiles_size;
    vkCreateBuffer(device, &buffer_info, 0, &tiles);
    vkBindBufferMemory(device, tiles, *memory, tiles_offset);
    // create texes buffer
    buffer_info.size = texes_size;
    vkCreateBuffer(device, &buffer_info, 0, &texes);
    vkBindBufferMemory(device, texes, *memory, texes_offset);
    // create globals buffer
    buffer_info.size = globals_size;
    vkCreateBuffer(device, &buffer_info, 0, &globals);
    vkBindBufferMemory(device, globals, *memory, globals_offset);
    // create tmem buffer
    buffer_info.size = tmem_size;
    vkCreateBuffer(device, &buffer_info, 0, &tmem);
    vkBindBufferMemory(device, tmem, *memory, tmem_offset);
    // create pixels buffer
    buffer_info.size = pixels_size;
    vkCreateBuffer(device, &buffer_info, 0, &pixels);
    vkBindBufferMemory(device, pixels, *memory, pixels_offset);
    // create zbuf buffer
    buffer_info.size = zbuf_size;
    vkCreateBuffer(device, &buffer_info, 0, &zbuf);
    vkBindBufferMemory(device, zbuf, *memory, zbuf_offset);
  }

  void init_descriptors(const VkDescriptorSetLayout &layout, VkDescriptorSet *descriptors) {
    // allocate descriptor set from descriptor pool
    const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 7
    };
    const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size
    };
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(device, &pool_info, 0, &pool);
    const VkDescriptorSetAllocateInfo descriptors_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout
    };
    vkAllocateDescriptorSets(device, &descriptors_info, descriptors);
    // bind buffers to descriptor set
    const VkDescriptorBufferInfo buffer_info[] = {
      { .buffer = cmds, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = tiles, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = texes, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = globals, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = tmem, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = pixels, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = zbuf, .offset = 0, .range = VK_WHOLE_SIZE }
    };
    VkWriteDescriptorSet write_descriptors[7];
    for (uint8_t i = 0; i < 7; ++i) {
      write_descriptors[i] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = *descriptors, .dstBinding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .pBufferInfo = &buffer_info[i]
      };
    }
    vkUpdateDescriptorSets(device, 7, write_descriptors, 0, 0);
  }
  
  void record_commands(const VkPipelineLayout &layout, const VkPipeline &pipeline,
      const VkDescriptorSet &descriptors) {
    // allocate command buffer from command pool
    const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_idx
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &pool_info, 0, &pool);
    const VkCommandBufferAllocateInfo commands_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .commandBufferCount = 1,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    };
    vkAllocateCommandBuffers(device, &commands_info, &commands);
    // record commands - bind descriptor set to set layout, start pipeline
    const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    vkBeginCommandBuffer(commands, &begin_info);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
      layout, 0, 1, &descriptors, 0, 0);
    vkCmdDispatch(commands, gwidth, gheight, 1);
    vkEndCommandBuffer(commands);
  }

  void init() {
    // setup Vulkan instance, logical device
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    const VkApplicationInfo app_info = {
      .pApplicationName = "nmulator RDP",
      .apiVersion = VK_API_VERSION_1_0
    };
    const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
    };
    vkCreateInstance(&instance_info, 0, &instance);
    init_device(instance, &gpu);

    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet descriptors = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;

    // use structured bindings instead of output params?
    const uint32_t *code = reinterpret_cast<const uint32_t*>(rdp_spv);
    init_pipeline(code, rdp_spv_len, &desc_layout, &layout, &pipeline);
    init_buffers(gpu, &memory);
    init_descriptors(desc_layout, &descriptors);
    record_commands(layout, pipeline, descriptors);
    
    // map memory so buffers can filled by cpu
    vkMapMemory(device, memory, 0, total_size, 0, reinterpret_cast<void**>(&mapped_mem));
    memset(tiles_ptr(), 0, tiles_size);
  }

  /* === Runtime Methods === */

  void run_commands() {
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_idx, 0, &queue);
    const VkSubmitInfo submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &commands
    };
    vkQueueSubmit(queue, 1, &submitInfo, 0);
    vkQueueWaitIdle(queue);
  }

  void add_rdp_cmd(cmd_t cmd) {
    uint32_t yh = (cmd.xyh[1] >> 2) / group_size;
    if (yh >> 13) yh = 0;
    uint32_t yl = (cmd.xyl[1] >> 2) / group_size;
    for (uint32_t i = yh; i <= yl; ++i) {
      for (uint32_t j = 0; j < gwidth; ++j) {
        tile_t *tile = tiles_ptr() + i * gwidth + j;
        if (tile->n_cmds >= 31) { printf("max cmds reached\n"); break; }
        tile->cmd_idxs[tile->n_cmds++] = n_cmds;
      }
    }
    cmds_ptr()[n_cmds++] = cmd;
  }

  void render(uint8_t *pixels, uint8_t *zbuf, uint32_t len) {
    if (!pixels || n_cmds == 0) return;
    memcpy(pixels_ptr(), pixels, len);
    //if (zbuf) memcpy(zbuf_ptr(), zbuf, len);
    run_commands(); n_cmds = 0; memset(tiles_ptr(), 0, tiles_size);
    memcpy(pixels, pixels_ptr(), len);
    //if (zbuf) memcpy(zbuf, zbuf_ptr(), len);
  }
}

namespace RSP {
  extern uint64_t reg_array[0x89];
  extern const uint8_t dev_cop0;
  template <typename T, bool all>
  int64_t read(uint32_t addr);
}

namespace R4300 {
  extern uint32_t mi_irqs;
  extern uint8_t *pages[0x100];
  template <typename T>
  int64_t read(uint32_t addr);
}

namespace RDP {
  uint64_t *rsp_cop0 = RSP::reg_array + RSP::dev_cop0;
  uint64_t &pc_start = rsp_cop0[8], &pc_end = rsp_cop0[9];
  uint64_t &pc = rsp_cop0[10], &status = rsp_cop0[11];

  uint32_t img_size = 0x0, img_width = 0, height = 240;
  uint8_t *img_addr = nullptr, *zbuf_addr = nullptr;
  uint32_t tex_size = 0x0, tex_width = 0, tex_addr = 0x0;

  uint32_t fill = 0x0, fog = 0x0, blend = 0x0;
  uint32_t env = 0x0, prim = 0x0, zprim = 0x0;
  uint32_t bl_mux = 0x0, cc_mux = 0x0, keys[3] = {0};

  uint8_t opcode(uint64_t addr) {
    uint32_t out = 0;
    if (status & 0x1) out = RSP::read<uint32_t>(addr);
    else out = R4300::read<uint32_t>(addr);
    return (out >> 24) & 0x3f;
  }

  std::vector<uint32_t> fetch(uint64_t &addr, uint8_t len) {
    std::vector<uint32_t> out(len);
    for (uint8_t i = 0; i < len; ++i, addr += 4) {
      if (status & 0x1) out[i] = RSP::read<uint32_t>(addr);
      else out[i] = R4300::read<uint32_t>(addr);
    }
    return out;
  }

  void render() {
    Vulkan::render(img_addr, zbuf_addr, img_width * height * img_size);
  }

  /* === Instruction Translations === */

  void set_image() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    img_size = 1 << (((instr[0] >> 19) & 0x3) - 1);
    img_width = (instr[0] & 0x3ff) + 1;
    img_addr = R4300::pages[0] + (instr[1] & 0x3ffffff);

    global_t *globals = Vulkan::globals_ptr();
    globals->width = img_width, globals->size = img_size;
    Vulkan::gwidth = img_width / Vulkan::group_size;
  }

  void set_scissor() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    uint32_t yh = instr[0] & 0xfff, yl = instr[1] & 0xfff;
    height = (yl >> 2) - (yh >> 2);
    Vulkan::gheight = height / Vulkan::group_size;
  }

  void set_other_modes() {
    bl_mux = fetch(pc, 2)[1] >> 16;
  }

  void set_fill() {
    fill = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_fog() {
    fog = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_blend() {
    blend = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_combine() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    cc_mux = ((instr[0] >> 31) & 0x7fff) << 16;
    cc_mux |= (instr[1] >> 9) & 0x1ff;
  }

  void set_env() {
    env = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_prim() {
    prim = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_zprim() {
    zprim = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_zbuf() {
    zbuf_addr = R4300::pages[0] + (fetch(pc, 2)[1] & 0x3ffffff);
  }

  void set_key_r() {
    keys[0] = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_key_gb() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    keys[2] = (instr[1] & 0xffff) | ((instr[0] & 0xfff) << 16);
    keys[1] = (instr[1] >> 16) | ((instr[0] & 0xfff000) >> 4);
  }

  void set_texture() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    tex_size = 1 << (((instr[0] >> 19) & 0x3) - 1);
    tex_width = (instr[0] & 0x3ff) + 1;
    tex_addr = instr[1] & 0x3ffffff;
  }

  void set_tile() {
    std::vector<uint32_t> instr = fetch(pc, 2); render();
    Vulkan::texes_ptr()[(instr[1] >> 24) & 0x7] = {
      .format = (instr[0] >> 21) & 0x7, .size = (instr[0] >> 19) & 0x3,
      .width = ((instr[0] >> 9) & 0xff) << 3, .addr = (instr[0] & 0x1ff) << 3,
      .mask = { (instr[1] >> 4) & 0xf, (instr[1] >> 14) & 0xf },
      .shift = { instr[1] & 0xf, (instr[1] >> 10) & 0xf }
    };
  }

  void load_tile() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    uint32_t sh = (instr[1] >> 12) & 0xfff, th = instr[1] & 0xfff;
    uint32_t sl = (instr[0] >> 12) & 0xfff, tl = instr[0] & 0xfff;
    th >>= 2, tl >>= 2, sh >>= 2, sl >>= 2;
    tex_t tex = Vulkan::texes_ptr()[(instr[1] >> 24) & 0x7];
    uint8_t *mem = Vulkan::tmem_ptr() + tex.addr;

    uint32_t offset = (tl * tex_width + sl) * tex_size;
    uint32_t width = (sh - sl + 1) * tex_size;
    uint32_t ram = tex_addr + offset;
    for (uint32_t i = 0; i <= th - tl; ++i) {
      memcpy(mem, R4300::pages[0] + ram, width);
      mem += tex.width, ram += tex_width * tex_size;
    }
  }

  void load_block() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    uint32_t sh = (instr[1] >> 12) & 0xfff, dxt = instr[1] & 0xfff;
    uint32_t sl = (instr[0] >> 12) & 0xfff, tl = instr[0] & 0xfff;
    tl >>= 2, sh >>= 2, sl >>= 2;
    tex_t tex = Vulkan::texes_ptr()[(instr[1] >> 24) & 0x7];
    uint8_t *mem = Vulkan::tmem_ptr() + tex.addr;

    uint32_t offset = (tl * tex_width + sl) * tex_size;
    uint32_t len = (sh - sl + 1) * tex_size;
    uint32_t width = (dxt == 0 ? len : 0x4000 / dxt);
    for (uint32_t i = 0, ram = tex_addr + offset; i < len; i += width) {
      memcpy(mem, R4300::pages[0] + ram, width);
      mem += tex.width, ram += width;
    }
  }

  void shade_triangle(cmd_t &cmd) {
    std::vector<uint32_t> instr = fetch(pc, 16);
    cmd.shade[0] = (instr[0] & 0xffff0000) | (instr[4] >> 16);
    cmd.shade[1] = (instr[0] << 16) | (instr[4] & 0xffff);
    cmd.shade[2] = (instr[1] & 0xffff0000) | (instr[5] >> 16);
    cmd.shade[3] = (instr[1] << 16) | (instr[5] & 0xffff);
    cmd.sde[0] = (instr[8] & 0xffff0000) | (instr[12] >> 16);
    cmd.sde[1] = (instr[8] << 16) | (instr[12] & 0xffff);
    cmd.sde[2] = (instr[9] & 0xffff0000) | (instr[13] >> 16);
    cmd.sde[3] = (instr[9] << 16) | (instr[13] & 0xffff);
    cmd.sdx[0] = (instr[2] & 0xffff0000) | (instr[6] >> 16);
    cmd.sdx[1] = (instr[2] << 16) | (instr[6] & 0xffff);
    cmd.sdx[2] = (instr[3] & 0xffff0000) | (instr[7] >> 16);
    cmd.sdx[3] = (instr[3] << 16) | (instr[7] & 0xffff);
  }

  void tex_triangle(cmd_t &cmd) {
    std::vector<uint32_t> instr = fetch(pc, 16);
    cmd.tex[0] = (instr[0] & 0xffff0000) | (instr[4] >> 16);
    cmd.tex[1] = (instr[0] << 16) | (instr[4] & 0xffff);
    cmd.tde[0] = (instr[8] & 0xffff0000) | (instr[12] >> 16);
    cmd.tde[1] = (instr[8] << 16) | (instr[12] & 0xffff);
    cmd.tdx[0] = (instr[2] & 0xffff0000) | (instr[6] >> 16);
    cmd.tdx[1] = (instr[2] << 16) | (instr[6] & 0xffff);
  }

  void zbuf_triangle(cmd_t &cmd) {
    std::vector<uint32_t> instr = fetch(pc, 4);
    cmd.zpos = instr[0], cmd.zde = instr[2];
    cmd.zdx = instr[1];
  }

  template <uint8_t type>
  void triangle() {
    printf("Triangle of type %x\n", type);
    std::vector<uint32_t> instr = fetch(pc, 8);
    cmd_t cmd = {
      .xyh = { instr[4], instr[1] & 0x3fff },
      .xym = { instr[6], (instr[1] >> 16) & 0x3fff },
      .xyl = { instr[2], instr[0] & 0x3fff },
      .sh = instr[5], .sm = instr[7], .sl = instr[3],
      .fill = fill, .fog = fog, .blend = blend,
      .env = env, .prim = prim, .zprim = zprim,
      .bl_mux = bl_mux, .cc_mux = cc_mux,
      .keys = { keys[0], keys[1], keys[2] },
      .lft = (instr[0] >> 23) & 0x1, .type = type,
    };
    if (type & 0x4) shade_triangle(cmd);
    if (type & 0x2) tex_triangle(cmd);
    if (type & 0x1) zbuf_triangle(cmd);
    Vulkan::add_rdp_cmd(cmd);
  }

  template <bool flip>
  void tex_rectangle(cmd_t &cmd) {
    std::vector<uint32_t> instr = fetch(pc, 2);
    cmd.tex[0] = (flip ? instr[0] & 0xffff : instr[0] >> 16);
    cmd.tex[1] = (flip ? instr[0] >> 16 : instr[0] & 0xffff);
    cmd.tde[0] = instr[1] & 0xffff, cmd.tde[1] = instr[1] & 0xffff;
    cmd.tdx[0] = instr[1] >> 16, cmd.tdx[1] = instr[1] >> 16;
  }

  template <uint8_t type>
  void rectangle() {
    printf("Rectangle of type %x\n", type);
    std::vector<uint32_t> instr = fetch(pc, 2);
    cmd_t cmd = {
      .xyh = { (instr[1] >> 12) & 0xfff, instr[1] & 0xfff },
      .xyl = { (instr[0] >> 12) & 0xfff, instr[0] & 0xfff },
      .fill = fill, .fog = fog, .blend = blend,
      .env = env, .prim = prim, .zprim = zprim,
      .bl_mux = bl_mux, .cc_mux = cc_mux,
      .keys = { keys[0], keys[1], keys[2] },
      .type = type, .tile = (instr[1] >> 24) & 0x7,
    };
    if (type == 0xa) tex_rectangle<false>(cmd);
    if (type == 0xb) tex_rectangle<true>(cmd);
    Vulkan::add_rdp_cmd(cmd);
  }

  void invalid() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    printf("[RDP] Unimplemented instruction %x%x\n", instr[0], instr[1]);
    exit(1);
  }

  void update(uint32_t cycles) {
    // interpret config instructions 
    pc = pc_start;
    while (pc < pc_end) {
      switch (opcode(pc)) {
        case 0x00: pc += 8; break;
        case 0x08: triangle<0x0>(); break;
        case 0x09: triangle<0x1>(); break;
        case 0x0a: triangle<0x2>(); break;
        case 0x0b: triangle<0x3>(); break;
        case 0x0c: triangle<0x4>(); break;
        case 0x0d: triangle<0x5>(); break;
        case 0x0e: triangle<0x6>(); break;
        case 0x0f: triangle<0x7>(); break;
        case 0x24: rectangle<0xa>(); break;
        case 0x25: rectangle<0xb>(); break;
        case 0x2a: set_key_gb(); break;
        case 0x2b: set_key_r(); break;
        case 0x2c: printf("[RDP] SET CONVERT\n"); pc += 8; break;
        case 0x2d: set_scissor(); break;
        case 0x2e: set_zprim(); break;
        case 0x2f: set_other_modes(); break;
        case 0x30: load_tile(); break;  // LOAD TLUT
        case 0x32: printf("[RDP] SET TILE SIZE\n"); pc += 8; break;
        case 0x33: load_block(); break;
        case 0x34: load_tile(); break;
        case 0x35: set_tile(); break;
        case 0x36: rectangle<0x8>(); break;
        case 0x37: set_fill(); break;
        case 0x38: set_fog(); break;
        case 0x39: set_blend(); break;
        case 0x3a: set_prim(); break;
        case 0x3b: set_env(); break;
        case 0x3c: set_combine(); break;
        case 0x3d: set_texture(); break;
        case 0x3e: set_zbuf(); break;
        case 0x3f: set_image(); break;
        case 0x26: case 0x27: case 0x28: case 0x29:
          printf("[RDP] SYNC\n"); pc += 8; break;
        default: invalid(); break;
      }
    }
    // rasterize on GPU according to config
    status |= 0x80, R4300::mi_irqs |= 0x20; render();
  }
}

#endif

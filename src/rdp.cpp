#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL_vulkan.h>
#include <glad/vulkan.h>
#include "nmulator.h"
#include "comp.spv"

/* === Shader structs === */

// all coeffs are 16.16 fixed point
// except yh/yl/ym, which is 12.2

#pragma pack(push, 1)
struct TileData {
  uint32_t cmd_idxs[64];
};

struct ConfData {
  uint32_t width, size;
  uint32_t n_cmds, fmt;
};

struct RDPState {
  int32_t sxh, sxl, syh, syl;
  uint32_t modes[2], mux[2];
  uint32_t tlut, tmem, texes;
  uint32_t fill, fog, blend;
  uint32_t env, prim, lodf;
  uint32_t zprim, keys, keyc;
};

struct RDPCommand {
  uint32_t type, tile, pad;
  int32_t xh, xm, xl;
  int32_t yh, ym, yl;
  int32_t sh, sm, sl;

  int32_t shade[4], sde[4], sdx[4];
  int32_t tex[4], tde[4], tdx[4];
  RDPState state;
};

struct RDPTex {
  uint32_t format, size;
  uint32_t width, addr;
  int32_t sth[2], stl[2], shift[2];
  uint32_t pal, pad;
};
#pragma pack(pop)

struct RenderInfo {
  uint8_t *cbufl, *cbufh;
  uint8_t *zbufl, *zbufh;
  uint32_t cbuf_len;
  uint32_t zbuf_len;
  bool pending;
};

/* === Vulkan setup === */

namespace RDP {
  void render(bool sync);
}

static const uint32_t gsize = 8;
static uint32_t gwidth = 80, gheight = 60;

namespace Vulkan {
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_idx = 0;

  constexpr VkDeviceSize align(VkDeviceSize offset) {
    return (offset + 0x1f) & ~0x1f;
  }

  /* === Descriptor memory access === */

  void *vmems[2];
  uint8_t *mapped_mem = nullptr;
  const uint32_t max_cmds = 2048, max_copies = 4096;
  uint32_t n_cmds = 0, n_tmems = 0, mem_idx = 1;

  const VkDeviceSize cmds_offset = 0;
  const VkDeviceSize cmds_size = max_cmds * sizeof(RDPCommand);
  RDPCommand *cmds_ptr() { return (RDPCommand*)(mapped_mem); }

  const VkDeviceSize tiles_offset = align(cmds_offset + cmds_size);
  const VkDeviceSize tiles_size = gwidth * gheight * sizeof(TileData);
  TileData *tiles_ptr() { return (TileData*)(mapped_mem + tiles_offset); }

  const VkDeviceSize texes_offset = align(tiles_offset + tiles_size);
  const VkDeviceSize texes_size = max_copies * sizeof(RDPTex) * 8;
  RDPTex *texes_ptr() { return (RDPTex*)(mapped_mem + texes_offset) + n_tmems * 8; }

  const VkDeviceSize conf_offset = align(texes_offset + texes_size);
  const VkDeviceSize conf_size = sizeof(ConfData);
  ConfData *conf_ptr() { return (ConfData*)(mapped_mem + conf_offset); }

  const VkDeviceSize tmem_offset = align(conf_offset + conf_size);
  const VkDeviceSize tmem_size = max_copies * 0x1000;
  uint8_t *tmem_ptr() { return mapped_mem + tmem_offset + (n_tmems << 12); }

  const VkDeviceSize cbufl_offset = align(tmem_offset + tmem_size);
  const VkDeviceSize cbufl_size = 640 * 480 * sizeof(uint32_t);
  uint8_t *cbufl_ptr() { return mapped_mem + cbufl_offset; }

  const VkDeviceSize cbufh_offset = align(cbufl_offset + cbufl_size);
  const VkDeviceSize cbufh_size = 640 * 480 * sizeof(uint32_t);
  uint8_t *cbufh_ptr() { return mapped_mem + cbufh_offset; }

  const VkDeviceSize zbufl_offset = align(cbufh_offset + cbufh_size);
  const VkDeviceSize zbufl_size = 640 * 480 * sizeof(uint16_t);
  uint8_t *zbufl_ptr() { return mapped_mem + zbufl_offset; }

  const VkDeviceSize zbufh_offset = align(zbufl_offset + zbufl_size);
  const VkDeviceSize zbufh_size = 640 * 480 * sizeof(uint16_t);
  uint8_t *zbufh_ptr() { return mapped_mem + zbufh_offset; }

  const VkDeviceSize total_size = zbufh_offset + zbufh_size;
  VkCommandBuffer comp_cmds[2];
  VkFence fences[2];
  VkBuffer buffers[2];
  RenderInfo renders[2];

  /* === Vulkan initialization == */

  void init_instance(VkInstance *instance) {
    // create vulkan instance
    const char *layers[] = { "VK_LAYER_KHRONOS_validation" };
    const VkApplicationInfo app_info = {
      .pApplicationName = "nmulator RDP",
      .apiVersion = VK_API_VERSION_1_0
    };
    const char *exts[] = { "VK_KHR_get_physical_device_properties2" };
    const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
      .enabledLayerCount = 1, .ppEnabledLayerNames = layers,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts
    };
    vkCreateInstance(&instance_info, 0, instance);
  }

  void init_device(const VkInstance &instance, VkPhysicalDevice *gpu) {
    // check all vulkan physical devices
    uint32_t n_gpus = 0;
    VkPhysicalDevice gpus[16];
    vkEnumeratePhysicalDevices(instance, &n_gpus, 0);
    vkEnumeratePhysicalDevices(instance, &n_gpus, gpus);
    for (VkPhysicalDevice gpu_ : gpus) {
      // check queue families on each device
      uint32_t n_queues = 0;
      VkQueueFamilyProperties queues[16];
      vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &n_queues, 0);
      vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &n_queues, queues);
      // if queue family supports compute, init virtual device
      for (uint32_t i = 0; i < n_queues; ++i) {
        if (~queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) continue;
        const float priority = 1.0;
        const VkDeviceQueueCreateInfo queue_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = (queue_idx = i),
          .queueCount = 1, .pQueuePriorities = &priority
        };
        const char *exts[] = {
          "VK_KHR_storage_buffer_storage_class",
          "VK_KHR_16bit_storage"
        };
        const VkDeviceCreateInfo device_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
          .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
          .enabledExtensionCount = 2, .ppEnabledExtensionNames = exts
        };
        vkCreateDevice((*gpu = gpu_), &device_info, 0, &device);
        vkGetDeviceQueue(device, queue_idx, 0, &queue);
        return;
      }
    }
    printf("No compatible GPU found\n"), exit(1);
  }

  void init_compute(VkDescriptorSetLayout *desc_layout,
      VkPipelineLayout *layout, VkPipeline *pipeline) {
    // load shader code into module
    const VkShaderModuleCreateInfo comp_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g_main), .pCode = (uint32_t*)g_main,
    };
    VkShaderModule comp;
    vkCreateShaderModule(device, &comp_info, 0, &comp);
    // describe needed descriptor set layout for shader
    const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}
    };
    const VkDescriptorSetLayoutCreateInfo desc_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 9, .pBindings = bindings
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
      .module = comp, .pName = "main"
    };
    const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info, .layout = *layout
    };
    vkCreateComputePipelines(device, 0, 1, &pipeline_info, 0, pipeline);
    vkDestroyShaderModule(device, comp, nullptr);
  }

  void init_buffers(const VkPhysicalDevice &gpu, VkDeviceMemory *memory) {
    // create shared buffer to hold descriptors
    const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = total_size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1, .pQueueFamilyIndices = &queue_idx
    };
    vkCreateBuffer(device, &buffer_info, 0, &buffers[0]);
    vkCreateBuffer(device, &buffer_info, 0, &buffers[1]);
    // get memory requirements for buffer
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffers[0], &requirements);
    VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size
    };
    // find host visible, coherent memory meeting requirements
    const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(gpu, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      const VkMemoryType memory_type = props.memoryTypes[i];
      if ((memory_type.propertyFlags & flags) != flags) continue;
      allocate_info.memoryTypeIndex = i;
      vkAllocateMemory(device, &allocate_info, 0, &memory[0]);
      vkAllocateMemory(device, &allocate_info, 0, &memory[1]);
      break;
    }
    vkBindBufferMemory(device, buffers[0], memory[0], 0);
    vkBindBufferMemory(device, buffers[1], memory[1], 0);
  }

  void init_compute_desc(const VkDescriptorSetLayout &layout, VkDescriptorSet *desc) {
    // allocate descriptor set from descriptor pool
    const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 * 9
    };
    const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &pool_size
    };
    VkDescriptorPool pool;
    vkCreateDescriptorPool(device, &pool_info, 0, &pool);
    const VkDescriptorSetAllocateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout
    };
    vkAllocateDescriptorSets(device, &desc_info, &desc[0]);
    vkAllocateDescriptorSets(device, &desc_info, &desc[1]);
    // bind buffers to descriptor set
    for (uint32_t i = 0; i < 2; ++i) {
      const VkDescriptorBufferInfo buffer_info[] = {
        { .buffer = buffers[i], .offset = cmds_offset, .range = cmds_size },
        { .buffer = buffers[i], .offset = tiles_offset, .range = tiles_size },
        { .buffer = buffers[i], .offset = texes_offset, .range = texes_size },
        { .buffer = buffers[i], .offset = conf_offset, .range = conf_size },
        { .buffer = buffers[i], .offset = tmem_offset, .range = tmem_size },
        { .buffer = buffers[i], .offset = cbufl_offset, .range = cbufl_size },
        { .buffer = buffers[i], .offset = cbufh_offset, .range = cbufh_size },
        { .buffer = buffers[i], .offset = zbufl_offset, .range = zbufl_size },
        { .buffer = buffers[i], .offset = zbufh_offset, .range = zbufh_size }
      };
      const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = desc[i], .dstBinding = 0, .descriptorCount = 9,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = buffer_info
      };
      vkUpdateDescriptorSets(device, 1, &write, 0, 0);
    };
  }
  
  void record_compute(const VkPipelineLayout &layout,
      const VkPipeline &pipeline, const VkDescriptorSet *desc) {
    // allocate command buffer from command pool
    const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_idx
    };
    VkCommandPool pool;
    vkCreateCommandPool(device, &pool_info, 0, &pool);
    const VkCommandBufferAllocateInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 2
    };
    vkAllocateCommandBuffers(device, &cmd_info, comp_cmds);
    const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    // record commands - bind descriptor set and pipeline
    for (uint32_t i = 0; i < 2; ++i) {
      vkBeginCommandBuffer(comp_cmds[i], &begin_info);
      vkCmdBindPipeline(comp_cmds[i], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(comp_cmds[i], VK_PIPELINE_BIND_POINT_COMPUTE,
        layout, 0, 1, &desc[i], 0, 0);
      vkCmdDispatch(comp_cmds[i], gwidth, gheight, 1);
      VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT
      };
      vkCmdPipelineBarrier(comp_cmds[i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, 0, 0, 0);
      vkEndCommandBuffer(comp_cmds[i]);
    }
    // create fence for each command buffer
    VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    vkCreateFence(device, &fence_info, 0, &fences[0]);
    vkCreateFence(device, &fence_info, 0, &fences[1]);
  }
  
  void init() {
    // load vulkan functions
    SDL_Vulkan_LoadLibrary(0);
    void *ptr = SDL_Vulkan_GetVkGetInstanceProcAddr();
    GLADuserptrloadfunc fp = (GLADuserptrloadfunc)ptr;
    gladLoadVulkanUserPtr(0, fp, 0);

    // configure GPU and memory
    VkInstance instance;
    VkPhysicalDevice gpu;
    VkDeviceMemory memory[2];

    init_instance(&instance);
    gladLoadVulkanUserPtr(0, fp, instance);
    init_device(instance, &gpu);
    gladLoadVulkanUserPtr(gpu, fp, instance);
    init_buffers(gpu, memory);

    // setup compute pipeline work
    VkPipelineLayout layout;
    VkDescriptorSetLayout desc_layout;
    VkPipeline pipeline;
    VkDescriptorSet desc[2];

    init_compute(&desc_layout, &layout, &pipeline);
    init_compute_desc(desc_layout, desc);
    record_compute(layout, pipeline, desc);
    
    // map memory so buffers can filled by cpu
    vkMapMemory(device, memory[0], 0, total_size, 0, &vmems[0]);
    vkMapMemory(device, memory[1], 0, total_size, 0, &vmems[1]);
    mapped_mem = (uint8_t*)vmems[mem_idx];
    memset(tiles_ptr(), 0, tiles_size);
  }

  /* === Runtime methods === */

  void add_tmem_copy(RDPState &state) {
    uint8_t *last_tmem = tmem_ptr();
    RDPTex *last_texes = texes_ptr();
    state.tmem = ++n_tmems;
    memcpy(tmem_ptr(), last_tmem, 0x1000);
    memcpy(texes_ptr(), last_texes, sizeof(RDPTex) * 8);
    if (n_tmems >= max_copies - 1) RDP::render(0);
  }

  void add_rdp_cmd(RDPCommand cmd) {
    uint32_t yh = (cmd.yh < 0 ? 0 : cmd.yh / (gsize * 4));
    uint32_t yl = (cmd.yl < 0 ? 0 : cmd.yl / (gsize * 4));
    for (uint32_t i = yh; i <= yl && i < gheight; ++i) {
      for (uint32_t j = 0; j < gwidth; ++j) {
        TileData *tile = tiles_ptr() + i * gwidth + j;
        tile->cmd_idxs[n_cmds >> 5] |= 0x1 << (n_cmds & 0x1f);
      }
    }
    cmds_ptr()[n_cmds++] = cmd;
    if (n_cmds >= max_cmds) RDP::render(0);
  }

  void flush_compute(uint8_t idx) {
    const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &comp_cmds[idx],
    };
    vkQueueSubmit(queue, 1, &submit_info, fences[idx]);
    renders[idx].pending = true;
  }

  bool wait_compute(uint8_t idx) {
    if (!renders[idx].pending) return true;
    vkWaitForFences(device, 1, &fences[idx], VK_TRUE, -1);
    vkResetFences(device, 1, &fences[idx]);
    return renders[idx].pending = false;
  }

  void sync() {
    mapped_mem = (uint8_t*)vmems[mem_idx ^= 1];
    if (!wait_compute(mem_idx)) {
      RenderInfo *r = &renders[mem_idx];
      memcpy(r->zbufl, zbufl_ptr(), r->zbuf_len);
      memcpy(r->zbufh, zbufh_ptr(), r->zbuf_len);
      memcpy(r->cbufl, cbufl_ptr(), r->cbuf_len);
      memcpy(r->cbufh, cbufh_ptr(), r->cbuf_len);
    }
    mapped_mem = (uint8_t*)vmems[mem_idx ^= 1];
  }

  void render(RenderInfo *r) {
    if (r->cbuf_len > cbufl_size) r->cbuf_len = cbufl_size;
    if (r->zbuf_len > zbufl_size) r->zbuf_len = zbufl_size;
    memcpy(&renders[mem_idx], r, sizeof(RenderInfo));

    uint8_t *last_tmem = tmem_ptr();
    RDPTex *last_texes = texes_ptr();
    sync(); // drain other buffer

    // upload current buffer to GPU
    if (n_cmds > 0) {
      conf_ptr()->n_cmds = n_cmds;
      memcpy(zbufl_ptr(), r->zbufl, r->zbuf_len);
      memcpy(zbufh_ptr(), r->zbufh, r->zbuf_len);
      memcpy(cbufl_ptr(), r->cbufl, r->cbuf_len);
      memcpy(cbufh_ptr(), r->cbufh, r->cbuf_len);
      flush_compute(mem_idx);
    }

    // change buffers and reset data
    n_cmds = 0, n_tmems = 0;
    mapped_mem = (uint8_t*)vmems[mem_idx ^= 1];
    memset(tiles_ptr(), 0, tiles_size);
    memcpy(tmem_ptr(), last_tmem, 0x1000);
    memcpy(texes_ptr(), last_texes, sizeof(RDPTex) * 8);
  }
}

/* === RDP Interface === */

static uint32_t img_size, img_width, height;
static uint32_t cbuf_addr, zbuf_addr, zwrite;
static uint32_t tex_nibs, tex_width, tex_addr;
static RDPState state;

namespace RDP {
  uint64_t &pc_start = RSP::cop0[8], &pc_end = RSP::cop0[9];
  uint64_t &pc = RSP::cop0[10], &status = RSP::cop0[11];
}

/* === Helper functions === */

static int32_t sext(uint32_t val, uint32_t bits=32) {
  if (bits >= 32) return val;
  uint32_t mask = (1 << bits) - 1, sign = 1 << (bits - 1);
  return ((val & mask) ^ sign) - sign;
}

static int32_t zext(uint32_t val, uint32_t bits=32) {
  if (bits >= 32) return val;
  return val & ((1 << bits) - 1);
}

void RDP::render(bool sync) {
  RenderInfo render_info = {
    .cbufl = R4300::ram + cbuf_addr, .cbufh = R4300::hram + cbuf_addr,
    .zbufl = R4300::ram + zbuf_addr, .zbufh = R4300::hram + zbuf_addr,
    .cbuf_len = img_width * height * img_size,
    .zbuf_len = img_width * height * zwrite * 2,
  };
  Vulkan::render(&render_info), zwrite = 0;
  //if (sync) Vulkan::sync();
}

/* === Instruction translation === */

static void set_color_image(uint32_t *instr) {
  uint32_t addr = instr[1] & R4300::mask;
  uint32_t width = (instr[0] & 0x3ff) + 1;
  if (addr == cbuf_addr && width == img_width) return;
  RDP::render(0), img_width = width, cbuf_addr = addr;
  img_size = 1 << (((instr[0] >> 19) & 0x3) - 1);

  ConfData *conf = Vulkan::conf_ptr();
  conf->width = img_width, conf->size = img_size;
  conf->fmt = (instr[0] >> 19) & 0x1;
  gwidth = (img_width + gsize - 1) / gsize;
}

static void set_depth_image(uint32_t *instr) {
  if ((instr[1] & R4300::mask) == zbuf_addr) return;
  RDP::render(0), zbuf_addr = instr[1] & R4300::mask;
}

static void set_scissor(uint32_t *instr) {
  state.sxh = zext(instr[0] >> 12, 12), state.syh = zext(instr[0], 12);
  state.sxl = zext(instr[1] >> 12, 12), state.syl = zext(instr[1], 12);
  height = (state.syl + 3) / 4, gheight = (height + gsize - 1) / gsize;
}

static void set_other_modes(uint32_t *instr) {
  memcpy(state.modes, instr, 8);
  zwrite |= (state.modes[1] >> 5) & 1;
}

static void set_combine(uint32_t *instr) {
  memcpy(state.mux, instr, 8);
}

static void set_fill(uint32_t *instr) {
  state.fill = bswap32(instr[1]);
}

static void set_fog(uint32_t *instr) {
  state.fog = bswap32(instr[1]);
}

static void set_blend(uint32_t *instr) {
  state.blend = bswap32(instr[1]);
}

static void set_env(uint32_t *instr) {
  state.env = bswap32(instr[1]);
}

static void set_prim(uint32_t *instr) {
  state.lodf = (instr[0] & 0xff) << 24;
  state.prim = bswap32(instr[1]);
}

static void set_prim_depth(uint32_t *instr) {
  state.zprim = instr[1];
}

static void set_key_gb(uint32_t *instr) {
  state.keys &= 0xff, state.keyc &= 0xff;
  state.keys |= (instr[1] >>  8) & 0x00ff00;
  state.keys |= (instr[1] <<  8) & 0xff0000;
  state.keyc |= (instr[1] >> 16) & 0x00ff00;
  state.keyc |= (instr[1] <<  4) & 0xff0000;
}

static void set_key_r(uint32_t *instr) {
  state.keys &= 0xffff00, state.keys |= (instr[1] >> 0) & 0xff;
  state.keyc &= 0xffff00, state.keyc |= (instr[1] >> 8) & 0xff;
}

static void set_convert(uint32_t*) {
  /*uint64_t cmd = ((uint64_t)instr[0] << 32) | instr[1];
  state.convert[0] = 2 * sext(cmd >> 0, 9) + 1;
  state.convert[1] = 2 * sext(cmd >> 9, 9) + 1;
  state.convert[2] = 2 * sext(cmd >> 18, 9) + 1;
  state.convert[3] = 2 * sext(cmd >> 27, 9) + 1;
  state.convert[4] = sext(cmd >> 36, 9);
  state.convert[5] = sext(cmd >> 45, 9);*/
}

/* === TMEM handling === */

static void set_texture(uint32_t *instr) {
  tex_nibs = 0x1 << ((instr[0] >> 19) & 0x3);
  tex_width = (instr[0] & 0x3ff) + 1;
  tex_addr = instr[1] & R4300::mask;
}

static void set_tile(uint32_t *instr) {
  Vulkan::add_tmem_copy(state);
  uint8_t tex_idx = (instr[1] >> 24) & 0x7;
  RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
  tex.format = (instr[0] >> 21) & 0x7, tex.size = (instr[0] >> 19) & 0x3;
  tex.width = ((instr[0] >> 9) & 0x1ff) << 3;
  tex.addr = (instr[0] & 0x1ff) << 3, tex.pal = (instr[1] >> 20) & 0xf;
  tex.shift[0] = instr[1] & 0x3ff, tex.shift[1] = (instr[1] >> 10) & 0x3ff;
}

static void set_tile_size(uint32_t *instr) {
  Vulkan::add_tmem_copy(state);
  uint8_t tex_idx = (instr[1] >> 24) & 0x7;
  RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
  tex.sth[0] = (instr[1] >> 12) & 0xfff, tex.sth[1] = instr[1] & 0xfff;
  tex.stl[0] = (instr[0] >> 12) & 0xfff, tex.stl[1] = instr[0] & 0xfff;
}

static uint32_t taddr(uint32_t base, uint32_t addr, uint32_t row, uint32_t tn) {
  // swap every other 16-bit word on odd rows
  int32_t swap = (row & 0x1) << 2;
  if (tn != 8) return ((base + addr) & 0xffe) ^ swap;
  // handle rgba32 split into hi/lo tmem
  uint32_t offs = (base + addr / 2) & 0x7fe;
  return ((addr & 0x2) << 10 | offs) ^ swap;
}

static void load_tile(uint32_t *instr) {
  Vulkan::add_tmem_copy(state);
  uint8_t tex_idx = (instr[1] >> 24) & 0x7;
  RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
  int32_t sh = (tex.sth[0] = (instr[1] >> 12) & 0xfff) / 4;
  int32_t th = (tex.sth[1] = (instr[1] >>  0) & 0xfff) / 4;
  int32_t sl = (tex.stl[0] = (instr[0] >> 12) & 0xfff) / 4;
  int32_t tl = (tex.stl[1] = (instr[0] >>  0) & 0xfff) / 4;

  int32_t offset = (tl * tex_width + sl) * (tex_nibs / 2);
  int32_t len = (sh - sl + 1) * (tex_nibs / 2);
  uint8_t *ram = R4300::ram + tex_addr + offset;
  uint8_t *mem = Vulkan::tmem_ptr(), tn = tex_nibs;

  for (int32_t i = 0; i <= th - tl; ++i) {
    int32_t off = tex.addr + tex.width * i;
    for (int32_t j = 0; j & 0x7 || j < len; j += 2)
      *(uint16_t*)(mem + taddr(off, j, i, tn)) = *(uint16_t*)(ram + j);
    ram += tex_width * (tex_nibs / 2);
  }
}

static void load_block(uint32_t *instr) {
  Vulkan::add_tmem_copy(state);
  uint8_t tex_idx = (instr[1] >> 24) & 0x7;
  RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
  int32_t sh = tex.sth[0] = (instr[1] >> 12) & 0xfff;
  int32_t th = tex.sth[1] = (instr[1] >>  0) & 0xfff;
  int32_t sl = tex.stl[0] = (instr[0] >> 12) & 0xfff;
  int32_t tl = tex.stl[1] = (instr[0] >>  0) & 0xfff;

  int32_t offset = (tl * tex_width + sl) * (tex_nibs / 2);
  int32_t len = (sh - sl + 1) * (tex_nibs / 2);
  uint8_t *ram = R4300::ram + tex_addr + offset;
  uint8_t *mem = Vulkan::tmem_ptr(), tn = tex_nibs;

  for (int32_t j = 0, t = 0; j & 0x7 || j < len; j += 2) {
    int32_t i = t >> 11, off = tex.addr + tex.width * i;
    *(uint16_t*)(mem + taddr(off, j, i, tn)) = *(uint16_t*)(ram + j);
    if ((j & 0x6) == 0x6) t += th;
  }
}

static void load_tlut(uint32_t *instr) {
  Vulkan::add_tmem_copy(state);
  uint8_t tex_idx = (instr[1] >> 24) & 0x7;
  RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
  int32_t sh = (tex.sth[0] = (instr[1] >> 12) & 0xfff) / 4;
  int32_t th = (tex.sth[1] = (instr[1] >>  0) & 0xfff) / 4;
  int32_t sl = (tex.stl[0] = (instr[0] >> 12) & 0xfff) / 4;
  int32_t tl = (tex.stl[1] = (instr[0] >>  0) & 0xfff) / 4;

  int32_t offset = (tl * tex_width + sl) * (tex_nibs / 2);
  int32_t len = (sh - sl + 1) * (tex_nibs / 2) * 4;
  uint8_t *ram = R4300::ram + tex_addr + offset;
  uint8_t *mem = Vulkan::tmem_ptr(), tn = tex_nibs;

  for (int32_t i = 0; i <= th - tl; ++i) {
    int32_t off = tex.addr + tex.width * i;
    for (int32_t j = 0; j & 0x7 || j < len; j += 2)
      *(uint16_t*)(mem + taddr(off, j, i, tn)) = ((uint16_t*)ram)[j / 8];
    ram += tex_width * (tex_nibs / 2);
  }
}

/* === Geometry commands === */

static void shade_triangle(RDPCommand &cmd, uint32_t *instr) {
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

static void tex_triangle(RDPCommand &cmd, uint32_t *instr) {
  cmd.tex[0] = (instr[0] & 0xffff0000) | (instr[4] >> 16);
  cmd.tex[1] = (instr[0] << 16) | (instr[4] & 0xffff);
  cmd.tex[2] = (instr[1] & 0xffff0000) | (instr[5] >> 16);
  cmd.tde[0] = (instr[8] & 0xffff0000) | (instr[12] >> 16);
  cmd.tde[1] = (instr[8] << 16) | (instr[12] & 0xffff);
  cmd.tde[2] = (instr[9] & 0xffff0000) | (instr[13] >> 16);
  cmd.tdx[0] = (instr[2] & 0xffff0000) | (instr[6] >> 16);
  cmd.tdx[1] = (instr[2] << 16) | (instr[6] & 0xffff);
  cmd.tdx[2] = (instr[3] & 0xffff0000) | (instr[7] >> 16);
  if (state.modes[0] & 0x200000) cmd.tdx[0] = 0x200000;
}

static void zbuf_triangle(RDPCommand &cmd, uint32_t *instr) {
  cmd.tex[3] = instr[0], cmd.tde[3] = instr[2];
  cmd.tdx[3] = instr[1];
}

template <uint8_t type>
static void triangle(uint32_t *instr) {
  uint32_t t = type | ((instr[0] >> 19) & 0x10);
  RDPCommand cmd = {
    .type = t, .tile = (instr[0] >> 16) & 0x7,
    .xh = sext(instr[4] >>  1, 27), .xm = sext(instr[6] >> 1, 27),
    .xl = sext(instr[2] >>  1, 27), .yh = sext(instr[1] >> 0, 14),
    .ym = sext(instr[1] >> 16, 14), .yl = sext(instr[0] >> 0, 14),
    .sh = sext(instr[5] >>  3, 27), .sm = sext(instr[7] >> 3, 27),
    .sl = sext(instr[3] >>  3, 27),
  };
  cmd.state = state, instr += 8;
  if (type & 0x4) shade_triangle(cmd, instr), instr += 16;
  if (type & 0x2) tex_triangle(cmd, instr), instr += 16;
  if (type & 0x1) zbuf_triangle(cmd, instr), instr += 4;
  Vulkan::add_rdp_cmd(cmd);
}

template <bool flip>
static void tex_rectangle(RDPCommand &cmd, uint32_t *instr) {
  (flip ? cmd.tex[1] : cmd.tex[0]) = (instr[0] >> 16) << 16;
  (flip ? cmd.tex[0] : cmd.tex[1]) = (instr[0] & 0xffff) << 16;
  (flip ? cmd.tdx[0] : cmd.tde[1]) = (instr[1] & 0xffff) << 11;
  (flip ? cmd.tde[1] : cmd.tdx[0]) = (instr[1] >> 16) << 11;
  if (state.modes[0] & 0x200000) cmd.tdx[0] = 0x200000;
  cmd.tde[0] = 0, cmd.tdx[1] = 0;
}

template <uint8_t type>
static void rectangle(uint32_t *instr) {
  RDPCommand cmd = {
    .type = type, .tile = (instr[1] >> 24) & 0x7,
    .xh = zext(instr[1] >> 12, 12) << 13,
    .xl = zext(instr[0] >> 12, 12) << 13,
    .yh = zext(instr[1] >>  0, 12),
    .yl = zext(instr[0] >>  0, 12),
  };
  cmd.state = state, instr += 2;
  if (state.modes[0] & 0x200000) cmd.yl |= 3;
  if (type == 0xa) tex_rectangle<false>(cmd, instr);
  if (type == 0xb) tex_rectangle<true>(cmd, instr);
  Vulkan::add_rdp_cmd(cmd);
}

/* === Command decoding === */

static void invalid(uint32_t *instr) {
  const char *msg = "[RDP] Invalid Command %08x %08x\n";
  printf(msg, instr[0], instr[1]);
}

void RDP::update() {
  static uint32_t instr[44], len;
  const uint32_t instr_lengths[] = {
    2, 2, 2, 2, 2, 2, 2, 2, 8, 12, 24, 28, 24, 28, 40, 44,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,
    2, 2, 2, 2, 4, 4, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,
  };
  pc &= ~0x7, pc_end &= ~0x7, status |= 0x80;
  uint8_t *src = (status & 0x1 ? RSP::mem : R4300::ram);
  uint32_t mask = (status & 0x1 ? RSP::mask : R4300::mask);

  // read commands into buffer
  for (; pc < pc_end; pc += 8) {
    instr[len++] = read32(src + (pc & mask) + 0);
    instr[len++] = read32(src + (pc & mask) + 4);
    uint8_t opcode = (instr[0] >> 24) & 0x3f;
    if (len < instr_lengths[opcode]) continue;

    // call handler when complete
    switch ((len = 0), opcode) {
      case 0x08: triangle<0x0>(instr); break;
      case 0x09: triangle<0x1>(instr); break;
      case 0x0a: triangle<0x2>(instr); break;
      case 0x0b: triangle<0x3>(instr); break;
      case 0x0c: triangle<0x4>(instr); break;
      case 0x0d: triangle<0x5>(instr); break;
      case 0x0e: triangle<0x6>(instr); break;
      case 0x0f: triangle<0x7>(instr); break;
      case 0x24: rectangle<0xa>(instr); break;
      case 0x25: rectangle<0xb>(instr); break;
      case 0x2a: set_key_gb(instr); break;
      case 0x2b: set_key_r(instr); break;
      case 0x2c: set_convert(instr); break;
      case 0x2d: set_scissor(instr); break;
      case 0x2e: set_prim_depth(instr); break;
      case 0x2f: set_other_modes(instr); break;
      case 0x30: load_tlut(instr); break;
      case 0x32: set_tile_size(instr); break;
      case 0x33: load_block(instr); break;
      case 0x34: load_tile(instr); break;
      case 0x35: set_tile(instr); break;
      case 0x36: rectangle<0x8>(instr); break;
      case 0x37: set_fill(instr); break;
      case 0x38: set_fog(instr); break;
      case 0x39: set_blend(instr); break;
      case 0x3a: set_prim(instr); break;
      case 0x3b: set_env(instr); break;
      case 0x3c: set_combine(instr); break;
      case 0x3d: set_texture(instr); break;
      case 0x3e: set_depth_image(instr); break;
      case 0x3f: set_color_image(instr); break;
      case 0x29: R4300::set_irqs(0x20); render(1); break;
      case 0x00: case 0x26: case 0x27: case 0x28: break;
      default: invalid(instr); break;
    }
  }
}

void RDP::init() {
  Vulkan::init();
}

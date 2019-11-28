#ifndef RDP_H
#define RDP_H

#include <vulkan/vulkan.h>
#include <vector>

typedef struct PerTileData {
  uint32_t n_cmds;
  uint32_t cmd_idxs[31];
} tile_t;

typedef struct RDPCommand {
  uint32_t xyh[2];
  uint32_t xym[2];
  uint32_t xyl[2];
  uint32_t sh, sm, sl;
  uint32_t fill, lft, type;
  uint32_t shade[4];
  uint32_t de[4], dx[4];
} cmd_t;

namespace Vulkan {
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queue_idx = 0;
  VkCommandBuffer commands = VK_NULL_HANDLE;
  uint8_t *mapped_mem;

  const uint32_t group_size = 8, max_cmds = 32;
  const VkDeviceSize cmds_size = max_cmds * sizeof(cmd_t);
  const VkDeviceSize tiles_size = (320 / group_size) * (240 / group_size) * sizeof(tile_t);
  const VkDeviceSize pixels_size = 320 * 240 * sizeof(uint32_t);
  const VkDeviceSize mem_size = cmds_size + tiles_size + pixels_size;

  VkBuffer cmds = VK_NULL_HANDLE;
  VkBuffer tiles = VK_NULL_HANDLE;
  VkBuffer pixels = VK_NULL_HANDLE;
  uint32_t n_cmds = 0;

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
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}
    };
    const VkDescriptorSetLayoutCreateInfo desc_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3, .pBindings = bindings
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
          !(mem_size < props.memoryHeaps[memoryType.heapIndex].size)) continue;
      const VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .memoryTypeIndex = i, .allocationSize = mem_size
      };
      vkAllocateMemory(device, &allocate_info, 0, memory);
    }
    // exit if memory not allocated
    // create buffers from allocated device memory
    VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = cmds_size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1, .pQueueFamilyIndices = &queue_idx
    };
    vkCreateBuffer(device, &buffer_info, 0, &cmds);
    vkBindBufferMemory(device, cmds, *memory, 0);
    // tile data buffer
    buffer_info.size = tiles_size;
    vkCreateBuffer(device, &buffer_info, 0, &tiles);
    vkBindBufferMemory(device, tiles, *memory, cmds_size);
    // pixel buffer
    buffer_info.size = pixels_size;
    vkCreateBuffer(device, &buffer_info, 0, &pixels);
    vkBindBufferMemory(device, pixels, *memory, cmds_size + tiles_size);
  }

  void init_descriptors(const VkDescriptorSetLayout &layout, VkDescriptorSet *descriptors) {
    // allocate descriptor set from descriptor pool
    const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3
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
      { .buffer = pixels, .offset = 0, .range = VK_WHOLE_SIZE }
    };
    const VkWriteDescriptorSet write_descriptors[] = {
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = *descriptors, .dstBinding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .pBufferInfo = &buffer_info[0]
      }, {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = *descriptors, .dstBinding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .pBufferInfo = &buffer_info[1]
      }, {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = *descriptors, .dstBinding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .pBufferInfo = &buffer_info[2]
      }
    };
    vkUpdateDescriptorSets(device, 3, write_descriptors, 0, 0);
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
    vkCmdDispatch(commands, 320 / group_size, 240 / group_size, 1);
    vkEndCommandBuffer(commands);
  }

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

  void init(FILE *file) {
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

    // read shader file
    fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint32_t *code = new uint32_t[fsize >> 2];
    fread(code, 1, fsize, file);

    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet descriptors = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;

    // use structured bindings instead of output params?
    init_pipeline(code, fsize, &desc_layout, &layout, &pipeline);
    init_buffers(gpu, &memory);
    init_descriptors(desc_layout, &descriptors);
    record_commands(layout, pipeline, descriptors);
    
    // fill cmds and tiles with display list data
    vkMapMemory(device, memory, 0, mem_size, 0, reinterpret_cast<void**>(&mapped_mem));
  }

  void add_rdp_cmd(cmd_t cmd) {
    cmd_t *host_cmds = reinterpret_cast<cmd_t*>(mapped_mem);
    host_cmds[n_cmds++] = cmd;
  }

  uint32_t *render() {
    if (n_cmds == 0) return nullptr;
    tile_t *host_tiles = reinterpret_cast<tile_t*>(mapped_mem + cmds_size);
    for (uint32_t i = 0; i < tiles_size / sizeof(tile_t); ++i)
      host_tiles[i] = { .n_cmds = n_cmds, .cmd_idxs = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 } };
    printf("num commands: %x\n", n_cmds);
    run_commands(), n_cmds = 0;
    return reinterpret_cast<uint32_t*>(mapped_mem + cmds_size + tiles_size);
  }
}

namespace RSP {
  template <typename T>
  int64_t read(uint32_t addr);
}

namespace R4300 {
  template <typename T>
  int64_t read(uint32_t addr);
}

namespace RDP {
  uint32_t color = 0x0;
  uint32_t pc_start = 0x0, pc_end = 0x0;
  uint32_t pc = 0x0, status = 0x0;

  uint64_t fetch(uint32_t addr) {
    return status & 0x1 ? RSP::read<uint64_t>(addr) : R4300::read<uint64_t>(addr);
  }

  void set_fill_color() {
    color = fetch(pc) & 0xffffffff; pc += 8;
  }

  void fill_triangle() {
    uint64_t instr[4] = {
      fetch(pc), fetch(pc + 8), fetch(pc + 16), fetch(pc + 24)
    };
    Vulkan::add_rdp_cmd({
      .xyh = { uint32_t(instr[2] >> 32), uint32_t(instr[0] & 0x3fff) },
      .xym = { uint32_t(instr[3] >> 32), uint32_t((instr[0] >> 16) & 0x3fff) },
      .xyl = { uint32_t(instr[1] >> 32), uint32_t((instr[0] >> 32) & 0x3fff) },
      .sh = uint32_t(instr[2]), .sm = uint32_t(instr[3]), .sl = uint32_t(instr[1]),
      .lft = uint32_t((instr[0] >> 55) & 0x1), .fill = color, .type = 0
    });
    pc += 32;
  }

  void shade_triangle() {
    uint64_t instr[12] = {
      fetch(pc), fetch(pc + 8), fetch(pc + 16), fetch(pc + 24),
      fetch(pc + 32), fetch(pc + 40), fetch(pc + 48), fetch(pc + 56),
      fetch(pc + 64), fetch(pc + 72), fetch(pc + 80), fetch(pc + 88)
    };
    Vulkan::add_rdp_cmd({
      .xyh = { uint32_t(instr[2] >> 32), uint32_t(instr[0] & 0x3fff) },
      .xym = { uint32_t(instr[3] >> 32), uint32_t((instr[0] >> 16) & 0x3fff) },
      .xyl = { uint32_t(instr[1] >> 32), uint32_t((instr[0] >> 32) & 0x3fff) },
      .sh = uint32_t(instr[2]), .sm = uint32_t(instr[3]), .sl = uint32_t(instr[1]),
      .lft = uint32_t((instr[0] >> 55) & 0x1), .fill = color, .type = 2,
      .shade = {
        uint32_t(instr[4] >> 48) << 16 | uint32_t(instr[6] >> 48),
        uint32_t((instr[4] >> 32) & 0xffff) << 16 | uint32_t((instr[6] >> 32) & 0xffff),
        uint32_t((instr[4] >> 16) & 0xffff) << 16 | uint32_t((instr[6] >> 16) & 0xffff),
        uint32_t(instr[4] & 0xffff) << 16 | uint32_t(instr[6] & 0xffff),
      },
      .de = {
        uint32_t(instr[8] >> 48) << 16 | uint32_t(instr[10] >> 48),
        uint32_t((instr[8] >> 32) & 0xffff) << 16 | uint32_t((instr[10] >> 32) & 0xffff),
        uint32_t((instr[8] >> 16) & 0xffff) << 16 | uint32_t((instr[10] >> 16) & 0xffff),
        uint32_t(instr[8] & 0xffff) << 16 | uint32_t(instr[10] & 0xffff),
      },
      .dx = {
        uint32_t(instr[5] >> 48) << 16 | uint32_t(instr[7] >> 48),
        uint32_t((instr[5] >> 32) & 0xffff) << 16 | uint32_t((instr[7] >> 32) & 0xffff),
        uint32_t((instr[5] >> 16) & 0xffff) << 16 | uint32_t((instr[7] >> 16) & 0xffff),
        uint32_t(instr[5] & 0xffff) << 16 | uint32_t(instr[7] & 0xffff),
      }
    });
    pc += 96;
  }

  void fill_rectangle() {
    uint64_t instr = fetch(pc);
    Vulkan::add_rdp_cmd({
      .xyh = { uint32_t((instr >> 12) & 0xfff), uint32_t(instr & 0xfff) },
      .xyl = { uint32_t((instr >> 44) & 0xfff), uint32_t((instr >> 32) & 0x3ff) },
      .fill = color, .type = 1
    });
    pc += 8;
  }

  uint32_t *update(uint32_t cycles) {
    // interpret config instructions 
    pc = pc_start;
    while (pc != pc_end) {
      uint64_t instr = fetch(pc);
      switch (instr >> 56) {
        case 0x08: fill_triangle(); break;
        case 0x0c: shade_triangle(); break;
        case 0x36: fill_rectangle(); break;
        case 0x37: set_fill_color(); break;
        default: printf("RDP instruction %llx\n", instr); pc += 8; break;
      }
    }
    // rasterize on GPU according to config
    return Vulkan::render();
  }
}

#endif

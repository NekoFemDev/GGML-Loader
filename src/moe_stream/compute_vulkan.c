/*
 * compute_vulkan.c — batched Vulkan compute backend with VRAM weight residency.
 *
 * Submission model
 * ----------------
 *   begin_batch()  reset + begin one command buffer, slot = 0
 *   matmul(...)    upload x, resolve/cache W, record bind+push+dispatch
 *   end_batch()    vkEndCommandBuffer, one vkQueueSubmit, one fence wait,
 *                  then copy every recorded y back to host
 *
 * The caller must not read a batched output before end_batch(). The model
 * groups independent matmuls (q/k/v, 8x gate/up, 8x down) between the two
 * calls, so a 48-layer decode goes from ~1392 submits to ~240.
 *
 * Weight residency
 * ----------------
 * The model assigns every tensor a stable 64-bit key (bit 63 = resident,
 * bit 62 = expert, mixed with a model id). The streamer reuses host pointers,
 * so a pointer-keyed cache would be wrong; keys survive eviction from the host
 * LRU. A malloc'd VRAM mirror is kept under a byte budget with LRU eviction.
 * Entries touched during a batch are pinned so a 16-dispatch gate/up batch
 * cannot evict its own weights.
 */
#ifdef MOE_ENABLE_VULKAN

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "compute_backend.h"
#include "dequant.h"
#include "memtrack.h"
#include "vulkan_shaders.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#define VK_MAX_BATCH  32
#define VK_X_MAX      4096     /* floats per slot */
#define VK_Y_MAX      4096
#define VK_XSTRIDE    (VK_X_MAX * 4)
#define VK_YSTRIDE    (VK_Y_MAX * 4)
#define VK_CACHE_MAX  256
#define VK_BXSTRIDE   (1024 * 1024)   /* per-slot batch staging */
#define VK_BYSTRIDE   (1024 * 1024)
#define VK_DEFAULT_BUDGET (1536ull * 1024 * 1024)

typedef struct {
    uint64_t       key;
    VkBuffer       buf;
    VkDeviceMemory mem;
    void          *map;
    size_t         size;
    uint64_t       used;
    int            pinned;
    uint64_t       hash;
} wcache_t;

static uint64_t fnv64(const uint8_t *p, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

typedef struct {
    int              state;      /* 0 uninit, 1 ready, -1 failed */

    VkInstance       inst;
    VkPhysicalDevice phys;
    VkDevice         dev;
    VkQueue          queue;
    uint32_t         qfam;

    VkCommandPool    pool;
    VkCommandBuffer  cmd;
    VkFence          fence;

    VkDescriptorSetLayout dsl;
    VkDescriptorPool      dpool;
    VkDescriptorSet       dset[VK_MAX_BATCH + 1];  /* last = synchronous */
    VkPipelineLayout      playout;
    VkPipeline            pipe_f32;
    VkPipeline            pipe_q;
    VkPipeline            pipe_gemm;
    VkPipeline            pipe_coal;
    VkPipeline            pipe_noshared;

    /* per-slot staging for batched dispatches */
    VkBuffer       xbuf;
    VkDeviceMemory xmem;
    void          *xmap;
    VkBuffer       ybuf;
    VkDeviceMemory ymem;
    void          *ymap;

    /* prefill batch staging: one 1 MiB slice per slot */
    VkBuffer       bxbuf, bybuf;
    VkDeviceMemory bxmem, bymem;
    void          *bxmap, *bymap;

    /* synchronous staging (lm_head and other oversized outputs) */
    VkBuffer       sbuf[3];
    VkDeviceMemory smem[3];
    size_t         scap[3];
    void          *smap[3];

    int      in_batch;
    int      slot;
    uint64_t clock;
    struct {
        float *y;
        size_t bytes;
        size_t src;
        int    from_batch;
    } pending[VK_MAX_BATCH];
    int npending;

    wcache_t cache[VK_CACHE_MAX];
    int      ncache;
    size_t   cache_bytes;
    size_t   budget;

    char name[256];
} vk_state_t;

static vk_state_t g_vk;
static int g_coalesced = -1;   /* MOE_VK_COALESCED */
static int g_noshared  = -1;   /* MOE_VK_NOSHARED  */

/* One host-visible VkBuffer shared by the expert LRU and the GPU. The streamer
 * pread()s directly into this mapping, so there is no host copy and no upload. */
static struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    uint8_t       *base;
    size_t         size;
    size_t         slot;
    uint32_t       nslots;
    uint32_t       used;
    uint8_t       *bits;
} g_arena;

/* ---- helpers ------------------------------------------------------------ */

static void fallback_matmul(uint32_t type, const uint8_t *w, uint64_t ne0,
                            uint64_t ne1, const float *x, float *y)
{
    compute_backend_scalar()->matmul(NULL, type, w, 0, ne0, ne1, x, y);
}

static uint32_t find_mem_type(VkPhysicalDevice phys, uint32_t bits,
                              VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

static int make_buffer(vk_state_t *s, VkDeviceSize size, VkBuffer *buf,
                       VkDeviceMemory *mem, void **map)
{
    VkBufferCreateInfo bci = {
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(s->dev, &bci, NULL, buf) != VK_SUCCESS) {
        return -1;
    }

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(s->dev, *buf, &mr);
    uint32_t mt = find_mem_type(s->phys, mr.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) {
        return -1;
    }

    VkMemoryAllocateInfo mai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = mr.size,
        .memoryTypeIndex = mt,
    };
    if (vkAllocateMemory(s->dev, &mai, NULL, mem) != VK_SUCCESS) {
        return -1;
    }
    if (vkBindBufferMemory(s->dev, *buf, *mem, 0) != VK_SUCCESS) {
        return -1;
    }
    if (map && vkMapMemory(s->dev, *mem, 0, VK_WHOLE_SIZE, 0, map) != VK_SUCCESS) {
        return -1;
    }
    return 0;
}

static void destroy_buffer(vk_state_t *s, VkBuffer *buf, VkDeviceMemory *mem)
{
    if (*buf) {
        vkDestroyBuffer(s->dev, *buf, NULL);
        *buf = VK_NULL_HANDLE;
    }
    if (*mem) {
        vkFreeMemory(s->dev, *mem, NULL);
        *mem = VK_NULL_HANDLE;
    }
}

/* ---- VRAM weight cache -------------------------------------------------- */

static void cache_drop(vk_state_t *s, int i)
{
    s->cache_bytes -= s->cache[i].size;
    moe_mem_release(s->cache[i].size);
    if (s->cache[i].map) {
        vkUnmapMemory(s->dev, s->cache[i].mem);
    }
    destroy_buffer(s, &s->cache[i].buf, &s->cache[i].mem);
    s->cache[i] = s->cache[--s->ncache];
}

/* Find an unpinned entry for eviction. Returns -1 if all are in flight. */
static int cache_victim(vk_state_t *s)
{
    int      victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < s->ncache; i++) {
        if (!s->cache[i].pinned && s->cache[i].used < oldest) {
            oldest = s->cache[i].used;
            victim = i;
        }
    }
    return victim;
}

static VkBuffer cache_get(vk_state_t *s, uint64_t key, const uint8_t *w,
                          size_t bytes)
{
    for (int i = 0; i < s->ncache; i++) {
        if (s->cache[i].key == key && s->cache[i].size == bytes) {
            static int vfy = -1;
            if (vfy < 0) vfy = getenv("MOE_VK_VERIFY") != NULL;
            if (vfy && fnv64(w, bytes) != s->cache[i].hash) {
                fprintf(stderr, "[cache] STALE key=%016llx size=%zu\n",
                        (unsigned long long)key, bytes);
            }
            s->cache[i].used   = ++s->clock;
            s->cache[i].pinned = 1;
            return s->cache[i].buf;
        }
    }

    if (bytes > s->budget) {
        return VK_NULL_HANDLE;
    }

    while (s->cache_bytes + bytes > s->budget) {
        int victim = cache_victim(s);
        if (victim < 0) {
            return VK_NULL_HANDLE;   /* everything in flight this batch */
        }
        cache_drop(s, victim);
    }

    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void          *map = NULL;
    if (make_buffer(s, bytes, &buf, &mem, &map) != 0) {
        return VK_NULL_HANDLE;
    }
    if (!moe_mem_reserve(bytes)) {
        vkUnmapMemory(s->dev, mem);
        destroy_buffer(s, &buf, &mem);
        return VK_NULL_HANDLE;
    }
    memcpy(map, w, bytes);
    {
        static int vfy = -1;
        if (vfy < 0) vfy = getenv("MOE_VK_VERIFY") != NULL;
        if (vfy) {
            /* recompute after insert below; store via e->hash */
        }
    }

    if (s->ncache == VK_CACHE_MAX) {
        int victim = cache_victim(s);
        if (victim < 0) {
            vkUnmapMemory(s->dev, mem);
            destroy_buffer(s, &buf, &mem);
            moe_mem_release(bytes);
            return VK_NULL_HANDLE;
        }
        cache_drop(s, victim);
    }
    wcache_t *e = &s->cache[s->ncache++];
    e->key    = key;
    e->buf    = buf;
    e->mem    = mem;
    e->map    = map;
    e->size   = bytes;
    e->used   = ++s->clock;
    e->pinned = 1;
    e->hash   = 0;   /* filled only under MOE_VK_VERIFY */
    s->cache_bytes += bytes;
    return buf;
}

/* ---- init --------------------------------------------------------------- */

static VkPipeline build_pipeline(vk_state_t *s, const uint32_t *code, unsigned len)
{
    VkShaderModuleCreateInfo smci = {
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)len * sizeof(uint32_t),
        .pCode    = code,
    };
    VkShaderModule sm = VK_NULL_HANDLE;
    if (vkCreateShaderModule(s->dev, &smci, NULL, &sm) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm,
            .pName  = "main",
        },
        .layout = s->playout,
    };

    VkPipeline p = VK_NULL_HANDLE;
    VkResult   r = vkCreateComputePipelines(s->dev, VK_NULL_HANDLE, 1, &cpci,
                                            NULL, &p);
    vkDestroyShaderModule(s->dev, sm, NULL);
    return r == VK_SUCCESS ? p : VK_NULL_HANDLE;
}

static int vk_init(vk_state_t *s)
{
    if (s->state != 0) {
        return s->state > 0 ? 0 : -1;
    }
    s->state = -1;

#if !MOE_VULKAN_HAVE_SHADERS
    fprintf(stderr, "[vulkan] no embedded SPIR-V (run shaders/build.sh)\n");
    return -1;
#endif

    s->budget = VK_DEFAULT_BUDGET;
    const char *env = getenv("MOE_VK_BUDGET_MB");
    if (env) {
        long mb = strtol(env, NULL, 10);
        if (mb > 0) {
            s->budget = (size_t)mb * 1024 * 1024;
        }
    }

    VkApplicationInfo app = {
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = "moe-vulkan",
        .applicationVersion = 1,
        .apiVersion         = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo ici = {
        .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    if (vkCreateInstance(&ici, NULL, &s->inst) != VK_SUCCESS) {
        return -1;
    }

    uint32_t ndev = 0;
    if (vkEnumeratePhysicalDevices(s->inst, &ndev, NULL) != VK_SUCCESS || !ndev) {
        return -1;
    }
    VkPhysicalDevice *devs = (VkPhysicalDevice *)malloc(ndev * sizeof(*devs));
    if (!devs) {
        return -1;
    }
    vkEnumeratePhysicalDevices(s->inst, &ndev, devs);

    for (uint32_t i = 0; i < ndev && s->phys == VK_NULL_HANDLE; i++) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, NULL);
        VkQueueFamilyProperties *qf =
            (VkQueueFamilyProperties *)malloc(nq * sizeof(*qf));
        if (!qf) {
            continue;
        }
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, qf);
        for (uint32_t q = 0; q < nq; q++) {
            if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                s->phys = devs[i];
                s->qfam = q;
                break;
            }
        }
        free(qf);
    }
    free(devs);
    if (s->phys == VK_NULL_HANDLE) {
        return -1;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(s->phys, &props);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = s->qfam,
        .queueCount       = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos    = &qci,
    };
    if (vkCreateDevice(s->phys, &dci, NULL, &s->dev) != VK_SUCCESS) {
        return -1;
    }
    vkGetDeviceQueue(s->dev, s->qfam, 0, &s->queue);

    VkCommandPoolCreateInfo pci = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = s->qfam,
    };
    if (vkCreateCommandPool(s->dev, &pci, NULL, &s->pool) != VK_SUCCESS) {
        return -1;
    }
    VkCommandBufferAllocateInfo cbi = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = s->pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(s->dev, &cbi, &s->cmd) != VK_SUCCESS) {
        return -1;
    }

    VkFenceCreateInfo fci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    if (vkCreateFence(s->dev, &fci, NULL, &s->fence) != VK_SUCCESS) {
        return -1;
    }

    VkDescriptorSetLayoutBinding bind[3];
    for (int i = 0; i < 3; i++) {
        bind[i] = (VkDescriptorSetLayoutBinding){
            .binding         = (uint32_t)i,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo lci = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings    = bind,
    };
    if (vkCreateDescriptorSetLayout(s->dev, &lci, NULL, &s->dsl) != VK_SUCCESS) {
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 16,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &s->dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pcr,
    };
    if (vkCreatePipelineLayout(s->dev, &plci, NULL, &s->playout) != VK_SUCCESS) {
        return -1;
    }

    s->pipe_f32  = build_pipeline(s, matmul_f32_spv, matmul_f32_spv_len);
    s->pipe_q    = build_pipeline(s, matmul_quant_spv, matmul_quant_spv_len);
    s->pipe_gemm = build_pipeline(s, matmul_gemm_spv, matmul_gemm_spv_len);
    s->pipe_coal = build_pipeline(s, matmul_coalesced_spv, matmul_coalesced_spv_len);
    s->pipe_noshared = build_pipeline(s, matmul_noshared_spv, matmul_noshared_spv_len);
    if (!s->pipe_f32 || !s->pipe_q || !s->pipe_gemm || !s->pipe_coal ||
        !s->pipe_noshared) {
        return -1;
    }
    if (g_coalesced < 0) {
        g_coalesced = getenv("MOE_VK_COALESCED") != NULL;
        g_noshared = getenv("MOE_VK_NOSHARED") != NULL;
        fprintf(stderr, "[vk] GEMV variant: %s\n",
                g_noshared ? "no-shared-x" : (g_coalesced ? "coalesced" : "row"));
    }

    VkDescriptorPoolSize dps = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3 * (VK_MAX_BATCH + 1),
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = VK_MAX_BATCH + 1,
        .poolSizeCount = 1,
        .pPoolSizes    = &dps,
    };
    if (vkCreateDescriptorPool(s->dev, &dpci, NULL, &s->dpool) != VK_SUCCESS) {
        return -1;
    }
    VkDescriptorSetAllocateInfo dsai = {
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = s->dpool,
        .descriptorSetCount = VK_MAX_BATCH + 1,
        .pSetLayouts        = NULL,
    };
    VkDescriptorSetLayout layouts[VK_MAX_BATCH + 1];
    for (int i = 0; i < VK_MAX_BATCH + 1; i++) {
        layouts[i] = s->dsl;
    }
    dsai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(s->dev, &dsai, s->dset) != VK_SUCCESS) {
        return -1;
    }

    if (make_buffer(s, (VkDeviceSize)VK_MAX_BATCH * VK_XSTRIDE, &s->xbuf,
                    &s->xmem, &s->xmap) != 0 ||
        make_buffer(s, (VkDeviceSize)VK_MAX_BATCH * VK_YSTRIDE, &s->ybuf,
                    &s->ymem, &s->ymap) != 0 ||
        make_buffer(s, (VkDeviceSize)VK_MAX_BATCH * VK_BXSTRIDE, &s->bxbuf,
                    &s->bxmem, &s->bxmap) != 0 ||
        make_buffer(s, (VkDeviceSize)VK_MAX_BATCH * VK_BYSTRIDE, &s->bybuf,
                    &s->bymem, &s->bymap) != 0) {
        return -1;
    }

    if (getenv("MOE_VK_MEMINFO")) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(s->phys, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            fprintf(stderr, "[vk] memtype %u flags=0x%02x heap=%u\n", i,
                    mp.memoryTypes[i].propertyFlags,
                    mp.memoryTypes[i].heapIndex);
        }
        for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
            fprintf(stderr, "[vk] heap %u size=%llu MiB flags=0x%02x\n", i,
                    (unsigned long long)(mp.memoryHeaps[i].size / (1024 * 1024)),
                    mp.memoryHeaps[i].flags);
        }
    }

    s->state = 1;
    snprintf(s->name, sizeof(s->name), "%s", props.deviceName);
    fprintf(stderr, "[vulkan] %s, budget %zu MiB, batch %d\n",
            s->name, s->budget / (1024 * 1024), VK_MAX_BATCH);
    return 0;
}

/* ---- shared expert arena ------------------------------------------------ */

int compute_vulkan_arena_init(size_t bytes, size_t slot_size)
{
    if (g_arena.buf) {
        return 0;
    }
    if (slot_size == 0 || bytes < slot_size) {
        return -1;
    }
    if (vk_init(&g_vk) != 0) {
        return -1;
    }

    const uint32_t n  = (uint32_t)(bytes / slot_size);
    const size_t   sz = (size_t)n * slot_size;
    if (n == 0) {
        return -1;
    }
    if (!moe_mem_reserve(sz)) {
        fprintf(stderr, "[vk] arena refused by memory budget\n");
        return -1;
    }
    if (make_buffer(&g_vk, sz, &g_arena.buf, &g_arena.mem,
                    (void **)&g_arena.base) != 0) {
        moe_mem_release(sz);
        return -1;
    }
    g_arena.bits = (uint8_t *)calloc((n + 7u) / 8u, 1);
    if (!g_arena.bits) {
        vkUnmapMemory(g_vk.dev, g_arena.mem);
        destroy_buffer(&g_vk, &g_arena.buf, &g_arena.mem);
        moe_mem_release(sz);
        return -1;
    }
    g_arena.size   = sz;
    g_arena.slot   = slot_size;
    g_arena.nslots = n;
    g_arena.used   = 0;

    fprintf(stderr,
            "[vk] expert arena: %zu MiB, %u slots x %zu B (host+GPU one mapping)\n",
            sz / (1024 * 1024), n, slot_size);
    return 0;
}

void *compute_vulkan_arena_alloc(void *user, size_t size)
{
    (void)user;
    if (!g_arena.buf || size > g_arena.slot) {
        return NULL;
    }
    for (uint32_t i = 0; i < g_arena.nslots; i++) {
        const uint32_t byte = i >> 3u;
        const uint8_t  mask = (uint8_t)(1u << (i & 7u));
        if (!(g_arena.bits[byte] & mask)) {
            g_arena.bits[byte] |= mask;
            g_arena.used++;
            return g_arena.base + (size_t)i * g_arena.slot;
        }
    }
    return NULL;
}

void compute_vulkan_arena_free(void *user, void *ptr, size_t size)
{
    (void)user;
    (void)size;
    if (!g_arena.base || !ptr) {
        return;
    }
    const size_t off = (size_t)((uint8_t *)ptr - g_arena.base);
    const uint32_t i = (uint32_t)(off / g_arena.slot);
    if (off % g_arena.slot != 0 || i >= g_arena.nslots) {
        return;
    }
    g_arena.bits[i >> 3u] &= (uint8_t)~(1u << (i & 7u));
    if (g_arena.used > 0) {
        g_arena.used--;
    }
}

void compute_vulkan_cache_clear(void)
{
    while (g_vk.ncache > 0) {
        cache_drop(&g_vk, 0);
    }
}

void compute_vulkan_arena_shutdown(void)
{
    if (!g_arena.buf) {
        return;
    }
    vkUnmapMemory(g_vk.dev, g_arena.mem);
    destroy_buffer(&g_vk, &g_arena.buf, &g_arena.mem);
    free(g_arena.bits);
    moe_mem_release(g_arena.size);
    memset(&g_arena, 0, sizeof(g_arena));
}

static int arena_bind(const uint8_t *w, VkBuffer *buf, VkDeviceSize *off)
{
    if (!g_arena.base || !w) {
        return 0;
    }
    if (w >= g_arena.base && w < g_arena.base + g_arena.size) {
        *buf = g_arena.buf;
        *off = (VkDeviceSize)(w - g_arena.base);
        return 1;
    }
    return 0;
}

/* ---- batching ----------------------------------------------------------- */

static int g_no_batch = -1;

static void vk_begin_batch(void *self)
{
    vk_state_t *s = (vk_state_t *)self;
    if (g_no_batch < 0) {
        g_no_batch = getenv("MOE_VK_NO_BATCH") != NULL;
    }
    if (g_no_batch) {
        return;
    }
    if (!s || s->in_batch || vk_init(s) != 0) {
        return;
    }

    vkResetCommandBuffer(s->cmd, 0);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(s->cmd, &bi) != VK_SUCCESS) {
        s->state = -1;
        return;
    }
    s->in_batch = 1;
    s->slot     = 0;
    s->npending = 0;
    for (int i = 0; i < s->ncache; i++) {
        s->cache[i].pinned = 0;
    }
}

static int vk_end_batch(void *self)
{
    vk_state_t *s = (vk_state_t *)self;
    if (!s || !s->in_batch) {
        return 0;
    }

    if (vkEndCommandBuffer(s->cmd) != VK_SUCCESS) {
        s->in_batch = 0;
        s->state    = -1;
        return -1;
    }

    vkResetFences(s->dev, 1, &s->fence);
    VkSubmitInfo si = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &s->cmd,
    };
    if (vkQueueSubmit(s->queue, 1, &si, s->fence) != VK_SUCCESS ||
        vkWaitForFences(s->dev, 1, &s->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        s->in_batch = 0;
        s->state    = -1;
        return -1;
    }

    for (int i = 0; i < s->npending; i++) {
        const uint8_t *src = s->pending[i].from_batch
            ? (const uint8_t *)s->bymap + s->pending[i].src
            : (const uint8_t *)s->ymap + s->pending[i].src;
        memcpy(s->pending[i].y, src, s->pending[i].bytes);
    }

    s->in_batch = 0;
    s->slot     = 0;
    s->npending = 0;
    return 0;
}

/* Upload x for a slot and point descriptor set `d` at (wbuf, x, y). */
static int bind_slot(vk_state_t *s, int slot, VkBuffer wbuf, VkDeviceSize woff,
                     const float *x, uint64_t ne0,
                     const uint8_t *yhost, uint64_t ne1)
{
    if (ne0 > VK_X_MAX || ne1 > VK_Y_MAX) {
        return -1;
    }

    memcpy((uint8_t *)s->xmap + (size_t)slot * VK_XSTRIDE, x,
           (size_t)ne0 * sizeof(float));

    VkDescriptorBufferInfo info[3] = {
        { .buffer = wbuf, .offset = woff, .range = VK_WHOLE_SIZE },
        { .buffer = s->xbuf, .offset = (VkDeviceSize)slot * VK_XSTRIDE,
          .range = (VkDeviceSize)ne0 * sizeof(float) },
        { .buffer = s->ybuf, .offset = (VkDeviceSize)slot * VK_YSTRIDE,
          .range = (VkDeviceSize)ne1 * sizeof(float) },
    };
    VkWriteDescriptorSet wr[3];
    for (int i = 0; i < 3; i++) {
        wr[i] = (VkWriteDescriptorSet){
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = s->dset[slot],
            .dstBinding      = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo     = &info[i],
        };
    }
    vkUpdateDescriptorSets(s->dev, 3, wr, 0, NULL);

    s->pending[slot].y          = (float *)yhost;
    s->pending[slot].bytes      = (size_t)ne1 * sizeof(float);
    s->pending[slot].src        = (size_t)slot * VK_YSTRIDE;
    s->pending[slot].from_batch = 0;
    return 0;
}

static VkPipeline pick_gemv_pipe(vk_state_t *s, uint32_t type)
{
    if (type != GGML_T_F32) {
        if (g_noshared > 0 && s->pipe_noshared) {
            return s->pipe_noshared;
        }
        if (g_coalesced > 0 && s->pipe_coal) {
            return s->pipe_coal;
        }
    }
    return (type == GGML_T_F32) ? s->pipe_f32 : s->pipe_q;
}

static int record_dispatch(vk_state_t *s, int slot,
                           uint32_t type, uint64_t ne0, uint64_t ne1,
                           uint32_t x_off, uint32_t y_off)
{
    struct {
        uint32_t ne0, ne1, type, x_off, y_off, pad;
    } pc = { (uint32_t)ne0, (uint32_t)ne1, type, x_off, y_off, 0 };

    VkPipeline pipe = pick_gemv_pipe(s, type);

    vkCmdBindPipeline(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            s->playout, 0, 1, &s->dset[slot], 0, NULL);
    vkCmdPushConstants(s->cmd, s->playout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(s->cmd, (uint32_t)((ne1 + 63) / 64), 1, 1);
    return 0;
}

/* ---- synchronous path (outputs larger than a batch slot) ---------------- */

static int ensure_sbuf(vk_state_t *s, int i, size_t need)
{
    if (s->sbuf[i] && s->scap[i] >= need) {
        return 0;
    }
    if (s->smap[i]) {
        vkUnmapMemory(s->dev, s->smem[i]);
        s->smap[i] = NULL;
    }
    destroy_buffer(s, &s->sbuf[i], &s->smem[i]);
    size_t sz = need < (1u << 20) ? (1u << 20) : need;
    if (make_buffer(s, sz, &s->sbuf[i], &s->smem[i], &s->smap[i]) != 0) {
        return -1;
    }
    s->scap[i] = sz;
    return 0;
}

static void sync_matmul(vk_state_t *s, uint32_t type, const uint8_t *w,
                        uint64_t wkey, uint64_t ne0, uint64_t ne1,
                        const float *x, float *y)
{
    const size_t wbytes = (size_t)ggml_type_row_bytes(type, ne0) * (size_t)ne1;
    const size_t xbytes = (size_t)ne0 * sizeof(float);
    const size_t ybytes = (size_t)ne1 * sizeof(float);

    if (ensure_sbuf(s, 1, xbytes) != 0 ||
        ensure_sbuf(s, 2, ybytes) != 0) {
        fallback_matmul(type, w, ne0, ne1, x, y);
        return;
    }

    /* lm_head is 255 MiB: upload it once through the keyed cache instead of
     * copying it into the staging buffer on every token. */
    VkBuffer     wbuf = VK_NULL_HANDLE;
    VkDeviceSize woff = 0;
    if (wkey != 0) {
        wbuf = cache_get(s, wkey, w, wbytes);
    }
    if (wbuf == VK_NULL_HANDLE) {
        if (ensure_sbuf(s, 0, wbytes) != 0) {
            fallback_matmul(type, w, ne0, ne1, x, y);
            return;
        }
        memcpy(s->smap[0], w, wbytes);
        wbuf = s->sbuf[0];
    }

    memcpy(s->smap[1], x, xbytes);

    VkDescriptorBufferInfo info[3] = {
        { wbuf, woff, VK_WHOLE_SIZE },
        { s->sbuf[1], 0, VK_WHOLE_SIZE },
        { s->sbuf[2], 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet wr[3];
    for (int i = 0; i < 3; i++) {
        wr[i] = (VkWriteDescriptorSet){
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = s->dset[VK_MAX_BATCH],
            .dstBinding      = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo     = &info[i],
        };
    }
    vkUpdateDescriptorSets(s->dev, 3, wr, 0, NULL);

    struct {
        uint32_t ne0, ne1, type, x_off, y_off, pad;
    } pc = { (uint32_t)ne0, (uint32_t)ne1, type, 0, 0, 0 };

    vkResetFences(s->dev, 1, &s->fence);
    vkResetCommandBuffer(s->cmd, 0);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(s->cmd, &bi);
    vkCmdBindPipeline(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pick_gemv_pipe(s, type));
    vkCmdBindDescriptorSets(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            s->playout, 0, 1, &s->dset[VK_MAX_BATCH], 0, NULL);
    vkCmdPushConstants(s->cmd, s->playout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(s->cmd, (uint32_t)((ne1 + 63) / 64), 1, 1);
    vkEndCommandBuffer(s->cmd);

    VkSubmitInfo si = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &s->cmd,
    };
    if (vkQueueSubmit(s->queue, 1, &si, s->fence) != VK_SUCCESS ||
        vkWaitForFences(s->dev, 1, &s->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        s->state = -1;
        fallback_matmul(type, w, ne0, ne1, x, y);
        return;
    }
    memcpy(y, s->smap[2], ybytes);
}

/* ---- vtable matmul ------------------------------------------------------ */

static void vk_matmul(void *self, uint32_t type, const uint8_t *w, uint64_t wkey,
                      uint64_t ne0, uint64_t ne1, const float *x, float *y)
{
    vk_state_t *s = (vk_state_t *)self;

    if (!s || vk_init(s) != 0) {
        fallback_matmul(type, w, ne0, ne1, x, y);
        return;
    }

    if (type != GGML_T_F32 && type != GGML_T_Q4_K &&
        type != GGML_T_Q5_K && type != GGML_T_Q6_K) {
        fallback_matmul(type, w, ne0, ne1, x, y);
        return;
    }
    if (ne0 % 256 != 0 && type != GGML_T_F32) {
        fallback_matmul(type, w, ne0, ne1, x, y);
        return;
    }
    if (ne0 > VK_X_MAX || ne1 > VK_Y_MAX) {
        /* lm_head: too big for a slot; do it synchronously. If a batch is
         * open it must be flushed first so ordering is preserved. */
        if (s->in_batch) {
            vk_end_batch(s);
        }
        sync_matmul(s, type, w, wkey, ne0, ne1, x, y);
        return;
    }
    if (wkey == 0 || g_no_batch > 0) {
        if (s->in_batch) {
            vk_end_batch(s);
        }
        sync_matmul(s, type, w, wkey, ne0, ne1, x, y);
        return;
    }

    const size_t wbytes = (size_t)ggml_type_row_bytes(type, ne0) * (size_t)ne1;

    VkBuffer     wbuf = VK_NULL_HANDLE;
    VkDeviceSize woff = 0;
    const int in_arena = arena_bind(w, &wbuf, &woff);

    if (!in_arena) {
        wbuf = cache_get(s, wkey, w, wbytes);
    }
    if (wbuf == VK_NULL_HANDLE) {
        /* budget exhausted with every entry pinned: flush, retry once */
        if (s->in_batch) {
            vk_end_batch(s);
        }
        wbuf = cache_get(s, wkey, w, wbytes);
        if (wbuf == VK_NULL_HANDLE) {
            fallback_matmul(type, w, ne0, ne1, x, y);
            return;
        }
    }

    int was_batching = s->in_batch;
    if (!was_batching) {
        vk_begin_batch(s);
        if (!s->in_batch) {
            fallback_matmul(type, w, ne0, ne1, x, y);
            return;
        }
    }
    if (s->slot >= VK_MAX_BATCH || s->npending >= VK_MAX_BATCH) {
        vk_end_batch(s);
        vk_begin_batch(s);
        if (!s->in_batch) {
            fallback_matmul(type, w, ne0, ne1, x, y);
            return;
        }
    }

    const int slot = s->slot;
    if (bind_slot(s, slot, wbuf, woff, x, ne0, (const uint8_t *)y, ne1) != 0) {
        vk_end_batch(s);
        fallback_matmul(type, w, ne0, ne1, x, y);
        return;
    }
    record_dispatch(s, slot, type, ne0, ne1, 0, 0);
    s->slot++;
    s->npending++;

    if (!was_batching) {
        vk_end_batch(s);
    }
}


/* ---- batched prefill GEMM ----------------------------------------------- */

static void fallback_batch(uint32_t type, const uint8_t *w, uint64_t ne0,
                           uint64_t ne1, const float *x, const uint32_t *idx,
                           int n, float *y)
{
    compute_backend_scalar()->matmul_batch(NULL, type, w, 0, ne0, ne1, x, idx, n, y);
}

static void vk_matmul_batch(void *self, uint32_t type, const uint8_t *w,
                            uint64_t wkey, uint64_t ne0, uint64_t ne1,
                            const float *x, const uint32_t *idx, int n_tokens,
                            float *y)
{
    vk_state_t *s = (vk_state_t *)self;

    if (!s || n_tokens <= 0 || vk_init(s) != 0 || wkey == 0 ||
        (type != GGML_T_F32 && type != GGML_T_Q4_K &&
         type != GGML_T_Q5_K && type != GGML_T_Q6_K) ||
        (type != GGML_T_F32 && ne0 % 256 != 0) ||
        ne0 > VK_X_MAX ||
        (uint64_t)n_tokens * ne0 * 4 > VK_BXSTRIDE ||
        (uint64_t)n_tokens * ne1 * 4 > VK_BYSTRIDE) {
        fallback_batch(type, w, ne0, ne1, x, idx, n_tokens, y);
        return;
    }

    const size_t wbytes = (size_t)ggml_type_row_bytes(type, ne0) * (size_t)ne1;
    VkBuffer     wbuf = VK_NULL_HANDLE;
    VkDeviceSize woff = 0;
    const int in_arena = arena_bind(w, &wbuf, &woff);

    if (!in_arena) {
        wbuf = cache_get(s, wkey, w, wbytes);
    }
    if (wbuf == VK_NULL_HANDLE) {
        if (s->in_batch) {
            vk_end_batch(s);
        }
        wbuf = cache_get(s, wkey, w, wbytes);
        if (wbuf == VK_NULL_HANDLE) {
            fallback_batch(type, w, ne0, ne1, x, idx, n_tokens, y);
            return;
        }
    }

    int was_batching = s->in_batch;
    if (!was_batching) {
        vk_begin_batch(s);
        if (!s->in_batch) {
            fallback_batch(type, w, ne0, ne1, x, idx, n_tokens, y);
            return;
        }
    }
    if (s->slot >= VK_MAX_BATCH || s->npending >= VK_MAX_BATCH) {
        vk_end_batch(s);
        vk_begin_batch(s);
        if (!s->in_batch) {
            fallback_batch(type, w, ne0, ne1, x, idx, n_tokens, y);
            return;
        }
    }

    const int slot = s->slot;
    uint8_t *xb = (uint8_t *)s->bxmap + (size_t)slot * VK_BXSTRIDE;

    for (int t = 0; t < n_tokens; t++) {
        const uint32_t row = idx ? idx[t] : (uint32_t)t;
        memcpy(xb + (size_t)t * ne0 * sizeof(float),
               x + (size_t)row * ne0, (size_t)ne0 * sizeof(float));
    }

    VkDescriptorBufferInfo info[3] = {
        { .buffer = wbuf, .offset = woff, .range = VK_WHOLE_SIZE },
        { .buffer = s->bxbuf, .offset = (VkDeviceSize)slot * VK_BXSTRIDE,
          .range = (VkDeviceSize)n_tokens * ne0 * sizeof(float) },
        { .buffer = s->bybuf, .offset = (VkDeviceSize)slot * VK_BYSTRIDE,
          .range = (VkDeviceSize)n_tokens * ne1 * sizeof(float) },
    };
    VkWriteDescriptorSet wr[3];
    for (int i = 0; i < 3; i++) {
        wr[i] = (VkWriteDescriptorSet){
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = s->dset[slot],
            .dstBinding      = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo     = &info[i],
        };
    }
    vkUpdateDescriptorSets(s->dev, 3, wr, 0, NULL);

    static int g_use_gemm = -1;
    if (g_use_gemm < 0) g_use_gemm = getenv("MOE_VK_GEMM") != NULL;
    if (g_use_gemm && n_tokens > 1 && type != GGML_T_F32) {
        struct {
            uint32_t ne0, ne1, type, n_tokens, x_off, y_off;
        } pc = { (uint32_t)ne0, (uint32_t)ne1, type, (uint32_t)n_tokens, 0, 0 };

        vkCmdBindPipeline(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipe_gemm);
        vkCmdBindDescriptorSets(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                s->playout, 0, 1, &s->dset[slot], 0, NULL);
        vkCmdPushConstants(s->cmd, s->playout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(s->cmd, (uint32_t)((ne1 + 63) / 64), 1, 1);
    } else {
        for (int t = 0; t < n_tokens; t++) {
            record_dispatch(s, slot, type, ne0, ne1,
                            (uint32_t)t * (uint32_t)ne0,
                            (uint32_t)t * (uint32_t)ne1);
        }
    }

    s->pending[slot].y          = y;
    s->pending[slot].bytes      = (size_t)n_tokens * ne1 * sizeof(float);
    s->pending[slot].src        = (size_t)slot * VK_BYSTRIDE;
    s->pending[slot].from_batch = 1;

    s->slot++;
    s->npending++;

    if (!was_batching) {
        vk_end_batch(s);
    }
}

/* ---- vtable ------------------------------------------------------------- */

int compute_vulkan_is_ready(void) { return g_vk.state == 1; }
const char *compute_vulkan_device_name(void) { return g_vk.name[0] ? g_vk.name : "none"; }

const compute_backend_t *compute_backend_vulkan_impl(void)
{
    static compute_backend_t be;
    static int built = 0;

    if (!built) {
        const compute_backend_t *sc = compute_backend_scalar();
        be.name        = "vulkan";
        be.self        = &g_vk;
        be.matmul      = vk_matmul;
        be.matmul_batch = vk_matmul_batch;
        be.rmsnorm     = sc->rmsnorm;
        be.rope        = sc->rope;
        be.softmax     = sc->softmax;
        be.swiglu      = sc->swiglu;
        be.begin_batch = vk_begin_batch;
        be.end_batch   = vk_end_batch;
        be.sync        = NULL;
        built          = 1;
    }
    return &be;
}

#endif /* MOE_ENABLE_VULKAN */

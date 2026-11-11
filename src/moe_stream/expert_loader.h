/*
 * expert_loader.h — position-independent expert weight reads from a GGUF shard.
 *
 * Design notes:
 *   - The portable path (load_expert_to_buffer) is a plain buffered pread loop.
 *     No fseek: pread is atomic w.r.t. the file offset, so N worker threads can
 *     share one fd with zero locking.
 *   - The O_DIRECT path exists because a byte-budgeted userspace LRU plus the
 *     kernel page cache would otherwise hold two copies of every hot expert.
 *     On this model the tensor bases are only 32-byte aligned (observed
 *     base % 4096 == 160), while the slice length is a multiple of 4096, so
 *     O_DIRECT necessarily goes through an aligned bounce buffer.
 */
#ifndef MOE_EXPERT_LOADER_H
#define MOE_EXPERT_LOADER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#define EXPERT_IO_ALIGN 4096u

typedef struct {
    int      fd;
    int      odirect;     /* 1 if the fd was opened with O_DIRECT */
    size_t   align;       /* logical block size used for the bounce window */
    uint64_t file_size;
} expert_io_t;

/* Open a shard. If try_odirect is non-zero, fall back cleanly to buffered I/O
 * when the filesystem rejects O_DIRECT (common on exFAT/sdcardfs). */
int  expert_io_open(expert_io_t *io, const char *path, int try_odirect);
void expert_io_close(expert_io_t *io);

/* Requested signature: buffered, thread-safe, no seek state.
 * Returns 0 on success or -errno. Reads exactly `size` bytes unless the file
 * ends early, which is reported as -EIO with *got set to the short count. */
int load_expert_to_buffer(FILE *f, uint64_t offset, size_t size, uint8_t *buffer);

/* Same contract on a raw fd. */
int pread_exact(int fd, uint64_t offset, void *buf, size_t size, size_t *got);

/* Full read path used by the streaming layer: picks direct vs buffered,
 * handles the alignment window, copies into dst. */
int expert_io_read(expert_io_t *io, uint64_t offset, size_t size, void *dst);

/* Page-cache hints. expert_io_dontneed is the cheap way to stop buffered reads
 * from double-caching the model: we already own an LRU. */
void expert_io_willneed(int fd, uint64_t offset, size_t size);
void expert_io_dontneed(int fd, uint64_t offset, size_t size);
int  expert_io_advise_random(int fd);

#endif /* MOE_EXPERT_LOADER_H */

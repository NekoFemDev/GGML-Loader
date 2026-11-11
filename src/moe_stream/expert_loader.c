#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _FILE_OFFSET_BITS 64

#include "expert_loader.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

static int read_loop(int fd, uint64_t offset, void *buf, size_t size, size_t *got)
{
    uint8_t *p = (uint8_t *)buf;
    size_t done = 0;

    if (offset > (uint64_t)INT64_MAX) {
        errno = EINVAL;
        return -EINVAL;
    }

    while (done < size) {
        ssize_t n = pread(fd, p + done, size - done, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (got) {
                *got = done;
            }
            return -errno;
        }
        if (n == 0) {
            break; /* EOF */
        }
        done += (size_t)n;
    }

    if (got) {
        *got = done;
    }
    return done == size ? 0 : -EIO;
}

int pread_exact(int fd, uint64_t offset, void *buf, size_t size, size_t *got)
{
    return read_loop(fd, offset, buf, size, got);
}

int load_expert_to_buffer(FILE *f, uint64_t offset, size_t size, uint8_t *buffer)
{
    if (!f || !buffer) {
        return -EINVAL;
    }
    if (size == 0) {
        return 0;
    }

    /* fileno() is the whole point: it keeps the FILE* API without letting libc
     * buffer or seek on our behalf. */
    int fd = fileno(f);
    if (fd < 0) {
        return -EINVAL;
    }

    size_t got = 0;
    int rc = read_loop(fd, offset, buffer, size, &got);
    if (rc != 0) {
        return rc;
    }
    return 0;
}

/* O_DIRECT needs a block-aligned bounce window. It MUST be per-thread: the
 * cache-fill pool calls expert_io_read() from N workers sharing one fd. A
 * shared buffer means one worker free()s it while another is memcpy'ing out of
 * it -- a use-after-free that only shows up as a crash under load. */
struct tls_bounce {
    uint8_t *p;
    size_t   cap;
};

static pthread_key_t  g_bounce_key;
static pthread_once_t g_bounce_once = PTHREAD_ONCE_INIT;

static void tls_bounce_dtor(void *arg)
{
    struct tls_bounce *b = (struct tls_bounce *)arg;
    if (b) {
        free(b->p);
        free(b);
    }
}

static void tls_bounce_init(void)
{
    (void)pthread_key_create(&g_bounce_key, tls_bounce_dtor);
}

static int ensure_bounce(expert_io_t *io, size_t need, uint8_t **out)
{
    pthread_once(&g_bounce_once, tls_bounce_init);

    struct tls_bounce *b = (struct tls_bounce *)pthread_getspecific(g_bounce_key);
    if (!b) {
        b = (struct tls_bounce *)calloc(1, sizeof(*b));
        if (!b) {
            return -ENOMEM;
        }
        if (pthread_setspecific(g_bounce_key, b) != 0) {
            free(b);
            return -ENOMEM;
        }
    }

    if (!b->p || b->cap < need) {
        size_t cap = need;
        if (cap < (1u << 20)) {
            cap = 1u << 20;
        }
        void *p = NULL;
        if (posix_memalign(&p, io->align, cap) != 0) {
            return -ENOMEM;
        }
        free(b->p);
        b->p   = (uint8_t *)p;
        b->cap = cap;
    }

    *out = b->p;
    return 0;
}

int expert_io_open(expert_io_t *io, const char *path, int try_odirect)
{
    if (!io || !path) {
        return -EINVAL;
    }
    memset(io, 0, sizeof(*io));
    io->fd    = -1;
    io->align = EXPERT_IO_ALIGN;

    int flags = O_RDONLY | O_CLOEXEC;
    int fd    = -1;

#if defined(O_DIRECT) && (O_DIRECT != 0)
    if (try_odirect) {
        fd = open(path, flags | O_DIRECT);
        if (fd >= 0) {
            io->odirect = 1;
        }
    }
#else
    (void)try_odirect;
#endif
    if (fd < 0) {
        fd = open(path, flags);
        if (fd < 0) {
            return -errno;
        }
        io->odirect = 0;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        int err = -errno;
        close(fd);
        return err;
    }

    io->fd        = fd;
    io->file_size = (uint64_t)st.st_size;

    expert_io_advise_random(fd);
    return 0;
}

void expert_io_close(expert_io_t *io)
{
    if (!io) {
        return;
    }
    if (io->fd >= 0) {
        close(io->fd);
        io->fd = -1;
    }
}

int expert_io_read(expert_io_t *io, uint64_t offset, size_t size, void *dst)
{
    if (!io || io->fd < 0 || !dst) {
        return -EINVAL;
    }
    if (size == 0) {
        return 0;
    }
    if (offset + size < offset || offset + size > io->file_size) {
        return -EINVAL;
    }

    if (!io->odirect) {
        size_t got = 0;
        int rc = read_loop(io->fd, offset, dst, size, &got);
        return rc;
    }

    /* O_DIRECT contract: file offset, buffer address and byte count must all be
     * multiples of the logical block size. Round the window outwards, read into
     * the aligned bounce buffer, then copy the requested interior out. */
    const size_t   a     = io->align;
    const uint64_t start = offset & ~(uint64_t)(a - 1);
    uint64_t       end   = (offset + size + a - 1) & ~(uint64_t)(a - 1);
    if (end > io->file_size) {
        end = (io->file_size + a - 1) & ~(uint64_t)(a - 1);
    }
    if (end < offset + size) {
        return -EIO;
    }

    size_t span = (size_t)(end - start);
    uint8_t *bounce = NULL;
    int rc = ensure_bounce(io, span, &bounce);
    if (rc != 0) {
        return rc;
    }

    size_t got = 0;
    rc = read_loop(io->fd, start, bounce, span, &got);
    if (rc != 0) {
        /* A short tail read is expected only at EOF; anything else is real. */
        if (got < (size_t)(offset - start) + size) {
            return rc;
        }
    }

    memcpy(dst, bounce + (size_t)(offset - start), size);
    return 0;
}

void expert_io_willneed(int fd, uint64_t offset, size_t size)
{
#ifdef POSIX_FADV_WILLNEED
    if (fd >= 0 && size) {
        (void)posix_fadvise(fd, (off_t)offset, (off_t)size, POSIX_FADV_WILLNEED);
    }
#else
    (void)fd; (void)offset; (void)size;
#endif
}

void expert_io_dontneed(int fd, uint64_t offset, size_t size)
{
#ifdef POSIX_FADV_DONTNEED
    if (fd >= 0 && size) {
        (void)posix_fadvise(fd, (off_t)offset, (off_t)size, POSIX_FADV_DONTNEED);
    }
#else
    (void)fd; (void)offset; (void)size;
#endif
}

int expert_io_advise_random(int fd)
{
#ifdef POSIX_FADV_RANDOM
    if (fd >= 0) {
        return posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM) == 0 ? 0 : -errno;
    }
#else
    (void)fd;
#endif
    return 0;
}

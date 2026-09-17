/*
 * vision.c - the 54x42 stage that stands in for the VL53L9CX.
 *
 *   ESP32 byte stream -> frame parser -> threshold -> open/close -> Zhang-Suen
 *                                             |          |            |
 *                                          coverage    mask       skeleton   -> CDC 1
 *
 * Everything here is tunable at runtime over CDC 1, because this is the board
 * that gets reflashed by dropping a UF2 onto a drive, and the ESP32 is not.
 */

#include <stdio.h>
#include <string.h>

#include "pico/stdlib.h"
#include "tusb.h"

#include "vision.h"

#define CDC_VISION 1

#define VN (VISION_W * VISION_H)

#define MAGIC0 0xA5
#define MAGIC1 0x5A
#define HDR_LEN 12
#define MAX_PAYLOAD (160 * 120)     /* the 160x120 preview is the largest frame */

enum {
    TYPE_COVERAGE = 1,
    TYPE_MASK     = 2,
    TYPE_SKELETON = 3,
    TYPE_PREVIEW  = 4,
    TYPE_RAW54    = 6,
    TYPE_GRAPH    = 7,   /* joint list + edge list, not an image */
};

// ---------------------------------------------------------------- CRC

/* CRC-16/CCITT-FALSE. Same table as the sketch and, on the host side, the same
 * polynomial binascii.crc_hqx(data, 0xFFFF) computes. */
static const uint16_t crc_tab[16] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF
};

static uint16_t crc16(const uint8_t *p, uint32_t n) {
    uint16_t c = 0xFFFF;
    while (n--) {
        c = (uint16_t)((c << 4) ^ crc_tab[((c >> 12) ^ (*p >> 4)) & 0x0F]);
        c = (uint16_t)((c << 4) ^ crc_tab[((c >> 12) ^ (*p & 0x0F)) & 0x0F]);
        p++;
    }
    return c;
}

// ---------------------------------------------------------------- tunables

static uint8_t cfg_threshold = 40;
/* 0, matching the board this was tuned on. An opening severs any connection a
 * single cell wide, and at 54x42 a neck is about that - the head detaches into
 * its own blob and the largest-component step then throws it away. Turn it back
 * on with 'o' if the mask is speckled. */
static uint8_t cfg_open      = 0;
static uint8_t cfg_close     = 0;   /* dilations then erosions - see fill_holes */
static bool    cfg_fill      = true;
static bool    cfg_skeleton  = true;
static bool    cfg_coverage  = true;
/* Raw brightness needs the other polarity: a person darker than the wall is
 * "below the level", not above it. Against a difference image it stays off. */
static bool    cfg_invert    = false;
static bool    cfg_largest   = true;   /* keep only the biggest blob */
static uint8_t cfg_effective = 0;   /* the level actually used, for reporting */

// ---------------------------------------------------------------- buffers

static uint8_t in_buf[HDR_LEN + MAX_PAYLOAD];
static uint8_t out_buf[HDR_LEN + MAX_PAYLOAD];
static uint32_t out_len, out_pos;

static uint8_t g_mask[VN], g_skel[VN], g_tmp[VN];
static uint16_t g_stack[VN];
static uint8_t out_seq;

static uint32_t stat_frames, stat_bad_crc, stat_dropped;

// ---------------------------------------------------------------- output

static void out_reset(void) { out_len = out_pos = 0; }

bool vision_out_idle(void) { return out_len == 0; }

static void emit(uint8_t type, uint16_t w, uint16_t h, const uint8_t *d, uint16_t len) {
    if (out_len + HDR_LEN + len > sizeof out_buf) return;

    uint8_t *p = out_buf + out_len;
    uint16_t crc = crc16(d, len);
    p[0] = MAGIC0;          p[1] = MAGIC1;
    p[2] = type;            p[3] = out_seq++;
    p[4] = (uint8_t)w;      p[5] = (uint8_t)(w >> 8);
    p[6] = (uint8_t)h;      p[7] = (uint8_t)(h >> 8);
    p[8] = (uint8_t)len;    p[9] = (uint8_t)(len >> 8);
    p[10] = (uint8_t)crc;   p[11] = (uint8_t)(crc >> 8);
    memcpy(p + HDR_LEN, d, len);
    out_len += HDR_LEN + len;
}

static void out_drain(void) {
    while (out_pos < out_len) {
        uint32_t room = tud_cdc_n_write_available(CDC_VISION);
        if (!room) break;
        uint32_t n = out_len - out_pos;
        if (n > room) n = room;
        tud_cdc_n_write(CDC_VISION, out_buf + out_pos, n);
        out_pos += n;
    }
    tud_cdc_n_write_flush(CDC_VISION);
    if (out_pos >= out_len) out_reset();
}

/* 통째로 들어갈 자리가 날 때까지 들고 있는 한 줄.
 *
 * out_buf 를 거치지 않고 CDC FIFO 에 직접 쓰기 때문에, 프레임이 나가는 중에 쓰면
 * 그 한가운데 텍스트가 끼어 호스트 쪽 CRC 가 깨진다. 그리고 FIFO 에 자리가 모자라면
 * tud_cdc_n_write 는 들어가는 만큼만 쓰고 나머지를 버린다 - 반환값을 안 보면
 * 상태 줄이 중간에서 잘린 채 나간다(실제로 skel=1 에서 끊겼다).
 *
 * 그래서 자르지도, 끼워 넣지도 않는다. 자리가 날 때까지 미룬다. */
static char say_buf[256];
static bool say_pending;

static void say(const char *s) {
    snprintf(say_buf, sizeof say_buf, "%s", s);
    say_pending = true;
}

static void say_flush(void) {
    if (!say_pending || out_len || !tud_cdc_n_connected(CDC_VISION)) return;
    uint32_t n = (uint32_t)strlen(say_buf);
    if (tud_cdc_n_write_available(CDC_VISION) < n) return;   /* 다음 기회에 */
    tud_cdc_n_write(CDC_VISION, say_buf, n);
    tud_cdc_n_write_flush(CDC_VISION);
    say_pending = false;
}

// ---------------------------------------------------------------- morphology

/*
 * 3x3, 8-connected, with the border replicated rather than treated as empty.
 * A silhouette that runs off the edge of the frame is the normal case here -
 * somebody standing close - and zero-padding would eat a pixel off it on every
 * erosion, which then shows up as a skeleton that stops short of the edge.
 */
static uint8_t fetch(const uint8_t *m, int x, int y) {
    if (x < 0) x = 0; else if (x >= VISION_W) x = VISION_W - 1;
    if (y < 0) y = 0; else if (y >= VISION_H) y = VISION_H - 1;
    return m[y * VISION_W + x];
}

static void erode(const uint8_t *src, uint8_t *dst) {
    for (int y = 0; y < VISION_H; y++) {
        for (int x = 0; x < VISION_W; x++) {
            uint8_t v = 255;
            for (int dy = -1; dy <= 1 && v; dy++)
                for (int dx = -1; dx <= 1 && v; dx++)
                    if (!fetch(src, x + dx, y + dy)) v = 0;
            dst[y * VISION_W + x] = v;
        }
    }
}

static void dilate(const uint8_t *src, uint8_t *dst) {
    for (int y = 0; y < VISION_H; y++) {
        for (int x = 0; x < VISION_W; x++) {
            uint8_t v = 0;
            for (int dy = -1; dy <= 1 && !v; dy++)
                for (int dx = -1; dx <= 1 && !v; dx++)
                    if (fetch(src, x + dx, y + dy)) v = 255;
            dst[y * VISION_W + x] = v;
        }
    }
}

/* Each pass writes through g_tmp and copies back, so callers always find the
 * result in m. At 2268 pixels the copy costs nothing worth avoiding. */
static void morph(uint8_t *m, int n, bool erode_first) {
    for (int i = 0; i < n; i++) {
        erode_first ? erode(m, g_tmp) : dilate(m, g_tmp);
        memcpy(m, g_tmp, VN);
    }
    for (int i = 0; i < n; i++) {
        erode_first ? dilate(m, g_tmp) : erode(m, g_tmp);
        memcpy(m, g_tmp, VN);
    }
}

/*
 * Fill interior holes: flood the background inwards from the border, then
 * promote whatever background it never reached.
 *
 * This is here instead of a morphological closing because at 54x42 a person's
 * legs are one cell apart, and a single dilation bridges that gap - the two
 * legs fuse and the skeleton loses both limbs, which is far worse than the
 * holes closing was meant to fix. Hole filling repairs a hole of any size and
 * cannot join things that were separate, because it only ever touches
 * background that the outside could not reach.
 *
 * The flood is 4-connected on purpose: a diagonal chain of foreground then
 * counts as a wall, so a thin silhouette does not leak.
 */
static void fill_holes(uint8_t *m) {
    static const int dx[4] = {1, -1, 0, 0};
    static const int dy[4] = {0, 0, 1, -1};
    uint32_t sp = 0;

    memset(g_tmp, 0, VN);

    for (int x = 0; x < VISION_W; x++) {
        int a = x, b = (VISION_H - 1) * VISION_W + x;
        if (!m[a] && !g_tmp[a]) { g_tmp[a] = 1; g_stack[sp++] = (uint16_t)a; }
        if (!m[b] && !g_tmp[b]) { g_tmp[b] = 1; g_stack[sp++] = (uint16_t)b; }
    }
    for (int y = 0; y < VISION_H; y++) {
        int a = y * VISION_W, b = a + VISION_W - 1;
        if (!m[a] && !g_tmp[a]) { g_tmp[a] = 1; g_stack[sp++] = (uint16_t)a; }
        if (!m[b] && !g_tmp[b]) { g_tmp[b] = 1; g_stack[sp++] = (uint16_t)b; }
    }

    /* Every cell is marked before it is pushed, so the stack cannot exceed VN. */
    while (sp) {
        uint16_t p = g_stack[--sp];
        int px = p % VISION_W, py = p / VISION_W;
        for (int k = 0; k < 4; k++) {
            int nx = px + dx[k], ny = py + dy[k];
            if (nx < 0 || ny < 0 || nx >= VISION_W || ny >= VISION_H) continue;
            int q = ny * VISION_W + nx;
            if (m[q] || g_tmp[q]) continue;
            g_tmp[q] = 1;
            g_stack[sp++] = (uint16_t)q;
        }
    }

    for (int i = 0; i < VN; i++)
        if (!m[i] && !g_tmp[i]) m[i] = 255;
}

/*
 * Keep only the biggest blob.
 *
 * A difference image carries the background's texture wherever the subject
 * covers it - the edge of a shelf behind someone reads as a strong difference
 * along that edge, because the shirt in front of it is nothing like the shelf.
 * No threshold separates that from the person, since some of those edges are
 * stronger than the person's own middle.
 *
 * What does separate them is shape: the subject is one large connected region
 * and the leftovers are thin fragments. Taking the largest component drops
 * every fragment at once, whatever its brightness, and for one person in frame
 * that is exactly the right answer.
 */
static void keep_largest(uint8_t *m) {
    memset(g_tmp, 0, VN);
    uint8_t label = 0, best = 0;
    uint32_t best_n = 0;

    for (int s = 0; s < VN; s++) {
        if (!m[s] || g_tmp[s]) continue;
        if (label == 255) break;
        label++;

        uint32_t n = 0, sp = 0;
        g_tmp[s] = label;
        g_stack[sp++] = (uint16_t)s;
        while (sp) {
            uint16_t p = g_stack[--sp];
            n++;
            int px = p % VISION_W, py = p / VISION_W;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = px + dx, ny = py + dy;
                    if (nx < 0 || ny < 0 || nx >= VISION_W || ny >= VISION_H) continue;
                    int q = ny * VISION_W + nx;
                    if (!m[q] || g_tmp[q]) continue;
                    g_tmp[q] = label;
                    g_stack[sp++] = (uint16_t)q;
                }
        }
        if (n > best_n) { best_n = n; best = label; }
    }

    if (!best) return;
    for (int i = 0; i < VN; i++)
        if (g_tmp[i] != best) m[i] = 0;
}

// ---------------------------------------------------------------- thinning

static uint8_t nb(const uint8_t *m, int x, int y) {
    if (x < 0 || y < 0 || x >= VISION_W || y >= VISION_H) return 0;
    return m[y * VISION_W + x] ? 1 : 0;
}

/*
 * One Zhang-Suen sub-iteration. P2..P9 run clockwise from north; B is how many
 * neighbours are set and A how many 0->1 transitions there are going round.
 * Outside the frame counts as empty here (unlike the morphology above), which
 * is what lets a blob touching the edge thin all the way to it.
 */
static bool thin_pass(uint8_t *m, int step) {
    bool changed = false;
    memset(g_tmp, 0, VN);

    for (int y = 0; y < VISION_H; y++) {
        for (int x = 0; x < VISION_W; x++) {
            if (!m[y * VISION_W + x]) continue;

            uint8_t p[10];
            p[2] = nb(m, x,     y - 1);
            p[3] = nb(m, x + 1, y - 1);
            p[4] = nb(m, x + 1, y);
            p[5] = nb(m, x + 1, y + 1);
            p[6] = nb(m, x,     y + 1);
            p[7] = nb(m, x - 1, y + 1);
            p[8] = nb(m, x - 1, y);
            p[9] = nb(m, x - 1, y - 1);

            int b = p[2] + p[3] + p[4] + p[5] + p[6] + p[7] + p[8] + p[9];
            if (b < 2 || b > 6) continue;

            int a = 0;
            for (int k = 2; k <= 9; k++) {
                uint8_t next = (k == 9) ? p[2] : p[k + 1];
                if (!p[k] && next) a++;
            }
            if (a != 1) continue;

            if (step == 0) {
                if (p[2] && p[4] && p[6]) continue;
                if (p[4] && p[6] && p[8]) continue;
            } else {
                if (p[2] && p[4] && p[8]) continue;
                if (p[2] && p[6] && p[8]) continue;
            }

            g_tmp[y * VISION_W + x] = 1;
            changed = true;
        }
    }

    if (changed)
        for (int i = 0; i < VN; i++)
            if (g_tmp[i]) m[i] = 0;

    return changed;
}

static void thin(uint8_t *m) {
    /* Bounded: a 54x42 blob converges in well under ten rounds, and the cap
     * keeps a pathological input from eating the frame budget. */
    for (int i = 0; i < 24; i++) {
        bool c1 = thin_pass(m, 0);
        bool c2 = thin_pass(m, 1);
        if (!c1 && !c2) break;
    }
}

// ---------------------------------------------------------------- skeleton graph

/*
 * Turn the thinned skeleton into joints and straight edges.
 *
 * On a skeleton every pixel has exactly two neighbours unless it is an end
 * (one) or a branch (three or more), so those are the interesting points: ends
 * are heads, hands and feet, branches are shoulders and hips. Cluster them,
 * walk the two-neighbour runs between them, and what comes out is a stick
 * figure - points joined by lines - instead of a pixel trail.
 *
 * Thinning leaves branches as small blobs of two or three adjacent pixels
 * rather than one, so adjacent non-path pixels are merged into a single joint
 * at their centroid. Without that, one shoulder shows up as three.
 */

#define MAX_JOINTS 40
#define MAX_EDGES  48

static uint8_t g_nb[VN];        /* neighbour count on the skeleton */
static uint8_t g_jid[VN];       /* joint index + 1; 0 means path or empty */

typedef struct { uint8_t x, y, kind, deg; } joint_t;   /* kind: 0 end, 1 branch */
static joint_t g_joint[MAX_JOINTS];
static uint8_t g_njoint;
static uint8_t g_edge[MAX_EDGES][3];    /* a, b, run length */
static uint8_t g_nedge;

static uint8_t cfg_spur  = 3;   /* drop stubs shorter than this */
static bool    cfg_graph = true;

static void add_edge(uint8_t a, uint8_t b, uint8_t len) {
    if (a == b || !a || !b || g_nedge >= MAX_EDGES) return;
    if (a > b) { uint8_t t = a; a = b; b = t; }
    for (uint8_t i = 0; i < g_nedge; i++)
        if (g_edge[i][0] == a && g_edge[i][1] == b) return;
    g_edge[g_nedge][0] = a;
    g_edge[g_nedge][1] = b;
    g_edge[g_nedge][2] = len;
    g_nedge++;
}

static void graph_build(const uint8_t *sk) {
    g_njoint = g_nedge = 0;
    memset(g_jid, 0, VN);

    for (int y = 0; y < VISION_H; y++) {
        for (int x = 0; x < VISION_W; x++) {
            int i = y * VISION_W + x;
            if (!sk[i]) { g_nb[i] = 0; continue; }
            int c = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    if ((dx || dy) && nb(sk, x + dx, y + dy)) c++;
            g_nb[i] = (uint8_t)c;
        }
    }

    /* Merge touching ends and branches into one joint each. */
    for (int s = 0; s < VN; s++) {
        if (!sk[s] || g_nb[s] == 2 || g_jid[s]) continue;
        if (g_njoint >= MAX_JOINTS) break;

        uint8_t id = (uint8_t)(++g_njoint);
        uint32_t sx = 0, sy = 0, n = 0, sp = 0, maxnb = 0;
        g_jid[s] = id;
        g_stack[sp++] = (uint16_t)s;

        while (sp) {
            uint16_t p = g_stack[--sp];
            int px = p % VISION_W, py = p / VISION_W;
            sx += (uint32_t)px; sy += (uint32_t)py; n++;
            if (g_nb[p] > maxnb) maxnb = g_nb[p];
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = px + dx, ny = py + dy;
                    if (nx < 0 || ny < 0 || nx >= VISION_W || ny >= VISION_H) continue;
                    int q = ny * VISION_W + nx;
                    if (!sk[q] || g_nb[q] == 2 || g_jid[q]) continue;
                    g_jid[q] = id;
                    g_stack[sp++] = (uint16_t)q;
                }
        }
        g_joint[id - 1].x = (uint8_t)(sx / n);
        g_joint[id - 1].y = (uint8_t)(sy / n);
        g_joint[id - 1].kind = maxnb >= 3 ? 1 : 0;
        g_joint[id - 1].deg = 0;
    }

    /* Walk every run out of every joint. Path pixels have exactly two
     * neighbours, so each step has one way forward and the walk cannot fork. */
    memset(g_tmp, 0, VN);
    for (int s = 0; s < VN; s++) {
        if (!g_jid[s]) continue;
        int sx = s % VISION_W, sy = s / VISION_W;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (!dx && !dy) continue;
                int nx = sx + dx, ny = sy + dy;
                if (nx < 0 || ny < 0 || nx >= VISION_W || ny >= VISION_H) continue;
                int q = ny * VISION_W + nx;
                if (!sk[q]) continue;

                if (g_jid[q]) { add_edge(g_jid[s], g_jid[q], 1); continue; }
                if (g_tmp[q]) continue;

                int prev = s, cur = q, len = 1;
                g_tmp[cur] = 1;
                for (;;) {
                    int cx = cur % VISION_W, cy = cur / VISION_W, nxt = -1;
                    for (int ey = -1; ey <= 1 && nxt < 0; ey++)
                        for (int ex = -1; ex <= 1; ex++) {
                            if (!ex && !ey) continue;
                            int ax = cx + ex, ay = cy + ey;
                            if (ax < 0 || ay < 0 || ax >= VISION_W || ay >= VISION_H) continue;
                            int r = ay * VISION_W + ax;
                            if (r == prev || !sk[r]) continue;
                            if (!g_jid[r] && g_tmp[r]) continue;
                            nxt = r;
                            break;
                        }
                    if (nxt < 0) break;                 /* dead end */
                    if (g_jid[nxt]) { add_edge(g_jid[s], g_jid[nxt], (uint8_t)len); break; }
                    g_tmp[nxt] = 1;
                    prev = cur;
                    cur = nxt;
                    if (++len > 250) break;
                }
            }
        }
    }

    /* Prune stubs: a short run ending at a dead end is mask noise, not a limb.
     * Left in, every ragged edge of the silhouette sprouts its own twig. */
    for (uint8_t i = 0; i < g_nedge; i++) {
        g_joint[g_edge[i][0] - 1].deg++;
        g_joint[g_edge[i][1] - 1].deg++;
    }
    if (cfg_spur) {
        uint8_t kept = 0;
        for (uint8_t i = 0; i < g_nedge; i++) {
            joint_t *a = &g_joint[g_edge[i][0] - 1];
            joint_t *b = &g_joint[g_edge[i][1] - 1];
            bool stub = (a->deg == 1 && a->kind == 0) || (b->deg == 1 && b->kind == 0);
            if (stub && g_edge[i][2] < cfg_spur) continue;
            if (kept != i) memcpy(g_edge[kept], g_edge[i], 3);
            kept++;
        }
        g_nedge = kept;
    }
}

/* joints, then edges, in the 54x42 coordinate space the layers use. */
static void emit_graph(void) {
    uint8_t buf[2 + MAX_JOINTS * 3 + MAX_EDGES * 2];
    uint16_t n = 0;

    buf[n++] = g_njoint;
    buf[n++] = g_nedge;
    for (uint8_t i = 0; i < g_njoint; i++) {
        buf[n++] = g_joint[i].x;
        buf[n++] = g_joint[i].y;
        buf[n++] = g_joint[i].kind;
    }
    for (uint8_t i = 0; i < g_nedge; i++) {
        buf[n++] = (uint8_t)(g_edge[i][0] - 1);
        buf[n++] = (uint8_t)(g_edge[i][1] - 1);
    }
    emit(TYPE_GRAPH, VISION_W, VISION_H, buf, n);
}

// ---------------------------------------------------------------- pipeline

/*
 * Otsu: pick the level that best splits the field into two groups, by
 * maximising the variance between them.
 *
 * A fixed threshold is fine against a difference image, where "no change" is
 * always zero. Against raw brightness there is no such anchor - the level that
 * separates a person from a wall moves with the light, the exposure and where
 * the camera points - so the number has to come from the picture itself.
 */
static uint8_t otsu(const uint8_t *v, int n) {
    static uint32_t hist[256];
    memset(hist, 0, sizeof hist);
    for (int i = 0; i < n; i++) hist[v[i]]++;

    uint32_t sum = 0;
    for (int i = 0; i < 256; i++) sum += (uint32_t)i * hist[i];

    uint32_t wb = 0, sumb = 0;
    uint64_t best_var = 0;
    uint8_t best = 0;

    for (int t = 0; t < 256; t++) {
        wb += hist[t];
        if (!wb) continue;
        uint32_t wf = (uint32_t)n - wb;
        if (!wf) break;
        sumb += (uint32_t)t * hist[t];

        /* wb*wf*(mb-mf)^2, kept in integers: the means are scaled by wb*wf, so
         * compare (sumb*wf - (sum-sumb)*wb)^2 / (wb*wf) instead. */
        int64_t d = (int64_t)sumb * wf - (int64_t)(sum - sumb) * wb;
        uint64_t var = (uint64_t)((d < 0 ? -d : d)) * (uint64_t)((d < 0 ? -d : d))
                       / ((uint64_t)wb * wf);
        if (var > best_var) { best_var = var; best = (uint8_t)t; }
    }
    return best;
}

static void process(const uint8_t *coverage, bool is_diff) {
    /* t0 means "work it out from the histogram" */
    uint8_t thr = cfg_threshold;
    bool flat = false;

    if (!thr) {
        uint8_t lo = 255, hi = 0;
        for (int i = 0; i < VN; i++) {
            if (coverage[i] < lo) lo = coverage[i];
            if (coverage[i] > hi) hi = coverage[i];
        }
        /* Otsu always returns a split, even when there is nothing to split -
         * an empty scene would come back as a full-frame mask and a skeleton
         * made of noise. Below this spread, call it all background. */
        flat = (uint8_t)(hi - lo) < 8;
        thr = flat ? 255 : otsu(coverage, VN);
    }
    cfg_effective = thr;

    /*
     * Otsu's level belongs to the low class, so the test is strict. Inverting
     * takes the complement, for raw brightness where the subject is the darker
     * of the two groups.
     *
     * Never on a difference image, though: low there means "did not change",
     * so inverting asks for the background by definition, and the answer comes
     * back as a confident mask of the furniture.
     */
    bool invert = cfg_invert && !is_diff;
    for (int i = 0; i < VN; i++) {
        bool on = !flat && (invert ? coverage[i] <= thr : coverage[i] > thr);
        g_mask[i] = on ? 255 : 0;
    }

    /* Fill before opening, not after. An erosion widens a hole, and a hole
     * within one cell of the silhouette's edge gets widened into a notch open
     * to the background - at which point it is no longer interior and nothing
     * can fill it. Sealing first costs nothing and removes that whole class of
     * failure; the speckle opening is meant to remove is in the background,
     * where filling never looks. */
    if (cfg_fill)  fill_holes(g_mask);
    if (cfg_open)  morph(g_mask, cfg_open, true);
    if (cfg_close) morph(g_mask, cfg_close, false);
    /* Last, so the opening has already thinned the fragments it will drop. */
    if (cfg_largest) keep_largest(g_mask);

    if (cfg_coverage) emit(TYPE_COVERAGE, VISION_W, VISION_H, coverage, VN);
    emit(TYPE_MASK, VISION_W, VISION_H, g_mask, VN);

    if (cfg_skeleton || cfg_graph) {
        memcpy(g_skel, g_mask, VN);
        thin(g_skel);
        for (int i = 0; i < VN; i++) g_skel[i] = g_skel[i] ? 255 : 0;
        if (cfg_skeleton) emit(TYPE_SKELETON, VISION_W, VISION_H, g_skel, VN);
        if (cfg_graph) { graph_build(g_skel); emit_graph(); }
    }
}

static void dispatch(uint8_t type, uint16_t w, uint16_t h, const uint8_t *payload, uint16_t len) {
    stat_frames++;

    if (!tud_cdc_n_connected(CDC_VISION)) return;

    /* Still busy shipping the last one: drop this frame whole rather than
     * interleave, and rather than let the DMA ring back up behind USB. */
    if (out_len) { stat_dropped++; return; }

    if ((type == TYPE_COVERAGE || type == TYPE_RAW54) && w == VISION_W && h == VISION_H) {
        process(payload, type == TYPE_COVERAGE);
    } else {
        emit(type, w, h, payload, len);   /* preview and anything else, verbatim */
    }
}

// ---------------------------------------------------------------- parser

static enum { S_IDLE, S_MAGIC, S_HDR, S_PAYLOAD } state;
static uint32_t have, want;
static uint8_t f_type;
static uint16_t f_w, f_h, f_len, f_crc;

bool vision_rx_byte(uint8_t b) {
    switch (state) {
    case S_IDLE:
        if (b != MAGIC0) return false;
        state = S_MAGIC;
        return true;

    case S_MAGIC:
        if (b == MAGIC1) { state = S_HDR; have = 2; return true; }
        if (b == MAGIC0) return true;       /* 0xA5 0xA5 ... keep looking */
        state = S_IDLE;
        /* The swallowed 0xA5 cannot have been log text - ASCII never reaches
         * 0x80 - so nothing legible was lost. This byte still might be. */
        return false;

    case S_HDR:
        in_buf[have++] = b;
        if (have < HDR_LEN) return true;
        f_type = in_buf[2];
        f_w   = (uint16_t)(in_buf[4] | (in_buf[5] << 8));
        f_h   = (uint16_t)(in_buf[6] | (in_buf[7] << 8));
        f_len = (uint16_t)(in_buf[8] | (in_buf[9] << 8));
        f_crc = (uint16_t)(in_buf[10] | (in_buf[11] << 8));
        if (f_len == 0 || f_len > MAX_PAYLOAD || (uint32_t)f_w * f_h != f_len) {
            state = S_IDLE;                 /* not a header after all */
            return true;
        }
        want = f_len;
        have = 0;
        state = S_PAYLOAD;
        return true;

    case S_PAYLOAD:
    default:
        in_buf[have++] = b;
        if (have < want) return true;
        state = S_IDLE;
        if (crc16(in_buf, f_len) == f_crc) dispatch(f_type, f_w, f_h, in_buf, f_len);
        else                               stat_bad_crc++;
        return true;
    }
}

// ---------------------------------------------------------------- commands

static void status(void) {
    char line[224];
    snprintf(line, sizeof line,
             "\n# rp2040: t=%u eff=%u inv=%u open=%u fill=%u close=%u skel=%u"
             " graph=%u spur=%u big=%u cov=%u baud=%lu joints=%u links=%u"
             " frames=%lu crcerr=%lu dropped=%lu\n",
             cfg_threshold, cfg_effective, cfg_invert, cfg_open, cfg_fill, cfg_close, cfg_skeleton,
             cfg_graph, cfg_spur, cfg_largest, cfg_coverage,
             (unsigned long)esp_link_get_vision_baud(),
             g_njoint, g_nedge,
             (unsigned long)stat_frames, (unsigned long)stat_bad_crc,
             (unsigned long)stat_dropped);
    say(line);
}

/* Commands this board owns; everything else goes on to the ESP32 untouched, so
 * the sketch's own letters keep working from the same terminal. '?' is both. */
static void command(char *line, uint32_t len) {
    unsigned v = 0;
    for (uint32_t i = 1; i < len; i++)
        if (line[i] >= '0' && line[i] <= '9') v = v * 10 + (unsigned)(line[i] - '0');

    switch (line[0]) {
    case 't': cfg_threshold = (uint8_t)(v > 255 ? 255 : v);         return;
    case 'o': cfg_open      = (uint8_t)(v > 4 ? 4 : v);             return;
    case 'l': cfg_close     = (uint8_t)(v > 4 ? 4 : v);             return;
    case 'h': cfg_fill      = v != 0;                               return;
    case 'k': cfg_skeleton  = v != 0;                               return;
    case 'j': cfg_graph     = v != 0;                               return;
    case 'w': cfg_spur      = (uint8_t)(v > 20 ? 20 : v);           return;
    case 'i': cfg_invert    = v != 0;                               return;
    case 'z': cfg_largest   = v != 0;                               return;
    case 'v': cfg_coverage  = v != 0;                               return;
    case 'u': if (v >= 300 && v <= 6000000) esp_link_set_vision_baud(v); return;
    case '?': status();     break;      /* fall through to the ESP32 too */
    default:  break;
    }

    line[len] = '\n';
    esp_link_write((const uint8_t *)line, len + 1);
}

static void poll_commands(void) {
    static char line[64];
    static uint32_t len;

    while (tud_cdc_n_available(CDC_VISION)) {
        int32_t c = tud_cdc_n_read_char(CDC_VISION);
        if (c < 0) break;
        if (c == '\n' || c == '\r') {
            if (len) { command(line, len); len = 0; }
        } else if (len < sizeof line - 2) {
            line[len++] = (char)c;
        }
    }
}

// ---------------------------------------------------------------- entry points

void vision_init(void) {
    state = S_IDLE;
    out_reset();
}

void vision_task(void) {
    if (!tud_cdc_n_connected(CDC_VISION)) {
        out_reset();
        poll_commands();        /* ESP 로 넘길 명령은 뷰어가 없어도 받는다 */
        return;
    }
    out_drain();
    /* status() 가 say() 로 텍스트를 뱉으므로, 프레임이 다 나간 뒤에만 명령을
     * 처리한다. 그러지 않으면 '?' 가 자기 답과 함께 프레임을 하나 깨뜨린다. */
    if (out_len == 0) poll_commands();
    say_flush();
}

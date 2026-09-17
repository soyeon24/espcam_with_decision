/*
 * esp32cam_sender - AI-Thinker ESP32-CAM as a stand-in for a VL53L9CX.
 *
 * Captures QQVGA greyscale, subtracts a slowly-adapting background model, and
 * box-downscales the absolute difference to a 54x42 coverage field - the same
 * grid the ToF sensor produced. The RP2040 thresholds it, cleans it up and
 * thins it to a skeleton.
 *
 * Everything tunable is a runtime command over the same UART, so this firmware
 * only ever has to be flashed once.
 *
 * Wiring is UART0 (U0T/U0R) to the RP2040-Zero, which is both the esptool
 * bridge and the frame link. Text written with Serial.print is forwarded to the
 * host untouched, because ASCII never contains 0xA5 and so can never be
 * mistaken for a frame header.
 */

#include "esp_camera.h"

// ---------------------------------------------------------------- AI-Thinker pins

#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0      /* also the bootstrap pin - see the README */
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

// ---------------------------------------------------------------- geometry

#define SRC_W   160
#define SRC_H   120
#define OUT_W    54
#define OUT_H    42

/* 54:42 is 9:7; the sensor gives 4:3. Crop the width so the aspect matches
 * rather than squashing the image: 120 * 9/7 = 154.3. */
#define CROP_X    3
#define CROP_W  154

#define LINK_BAUD 921600

// ---------------------------------------------------------------- frame protocol

#define MAGIC0 0xA5
#define MAGIC1 0x5A

enum {
    TYPE_DIFF54  = 1,   /* 54x42 coverage: box(|cur - bg|) */
    TYPE_PREVIEW = 4,   /* 160x120 raw greyscale           */
    TYPE_RAW54   = 6,   /* 54x42 box(cur), no subtraction  */
};

/* CRC-16/CCITT-FALSE, nibble at a time. The RP2040 uses the same table. */
static const uint16_t crc_tab[16] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF
};

static uint16_t crc16(const uint8_t *p, size_t n) {
    uint16_t c = 0xFFFF;
    while (n--) {
        c = (uint16_t)((c << 4) ^ crc_tab[((c >> 12) ^ (*p >> 4)) & 0x0F]);
        c = (uint16_t)((c << 4) ^ crc_tab[((c >> 12) ^ (*p & 0x0F)) & 0x0F]);
        p++;
    }
    return c;
}

// ---------------------------------------------------------------- state

static uint8_t  bg[SRC_W * SRC_H];      /* background model, 19.2 KB */
static uint8_t  field[OUT_W * OUT_H];   /* 54x42 output              */
static uint16_t colb[OUT_W + 1];        /* box boundaries in the source */
static uint16_t rowb[OUT_H + 1];
static uint8_t  seq;

static bool     bg_valid   = false;
static uint16_t bg_settle  = 0;         /* frames to discard before seeding */
static uint8_t  bg_period  = 4;         /* sigma-delta update every N frames, 0 = frozen */
static uint8_t  bg_guard   = 24;        /* do not absorb pixels this far off, 0 = absorb all */
static uint8_t  gain_q4    = 16;        /* difference gain, 16 = 1.0 */

static bool     send_diff  = true;      /* false: raw downscale, for aiming */
static uint8_t  prev_every = 0;         /* full preview every N frames, 0 = off */
static uint8_t  prev_count = 0;
static uint16_t min_period = 0;         /* ms between frames, 0 = free run */

/* Long enough for the sensor's own AEC/AGC loop to converge before we freeze
 * it - about 1.7 s at the rate this runs. */
#define SETTLE_FRAMES 60

static bool     auto_lock_pending;
static uint32_t bg_delay_until;         /* millis: hold off seeding until then */

/* Seeding the model from a frame taken before the sensor settled makes every
 * later frame differ by the whole image, which looks exactly like a working
 * camera with a broken threshold. */
static void bg_reset(void) {
    bg_valid = false;
    bg_settle = SETTLE_FRAMES;
}

/*
 * Let the sensor expose itself, then freeze it there.
 *
 * A hard-coded exposure is a guess about someone else's room, and getting it
 * wrong does not look like a bad guess - it looks like broken hardware. Too
 * high and every pixel pins at 255, so a person walking in changes nothing and
 * only something touching the lens registers at all.
 *
 * AEC still has to be off while running: the moment somebody walks in, an
 * active loop re-exposes and every pixel in the frame moves at once, which is
 * exactly what background subtraction cannot survive. So converge, then lock -
 * disabling the loop leaves the sensor holding whatever it had arrived at, no
 * read-back needed.
 */
static void recalibrate(uint32_t delay_s) {
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_exposure_ctrl(s, 1);
        s->set_gain_ctrl(s, 1);
    }
    auto_lock_pending = true;
    /* Whoever presses the key is usually the subject, and a camera on the desk
     * points straight at where they sit. Capturing immediately files them as
     * background, after which they can never differ from it by much - which
     * reads as a camera that simply cannot see people. */
    bg_delay_until = millis() + delay_s * 1000;
    bg_reset();
    if (delay_s) Serial.printf("# bg in %lus - step out of shot\n", (unsigned long)delay_s);
}

// ---------------------------------------------------------------- downscale

static void spans_init(void) {
    for (int i = 0; i <= OUT_W; i++) colb[i] = CROP_X + (uint16_t)((i * CROP_W) / OUT_W);
    for (int j = 0; j <= OUT_H; j++) rowb[j] = (uint16_t)((j * SRC_H) / OUT_H);
}

/*
 * Area-average each 54x42 cell. The absolute difference is taken here, per
 * source pixel, rather than after downscaling: box(|cur-bg|) is not the same as
 * |box(cur)-box(bg)|, and the latter cancels out whenever a cell holds both a
 * brighter-than-background and a darker-than-background part of the subject -
 * which is most of a person's outline.
 *
 * Nearest-neighbour would alias badly at a 2.85:1 ratio; the area average is
 * what gives each cell a "how much of me is subject" value instead of a coin
 * flip, and that sub-cell information is most of what makes a 54x42 skeleton
 * usable.
 */
static void downscale(const uint8_t *cur, bool diff) {
    for (int j = 0; j < OUT_H; j++) {
        const int y0 = rowb[j], y1 = rowb[j + 1];
        for (int i = 0; i < OUT_W; i++) {
            const int x0 = colb[i], x1 = colb[i + 1];
            uint32_t sum = 0;
            for (int y = y0; y < y1; y++) {
                const uint8_t *pc = cur + y * SRC_W;
                const uint8_t *pb = bg + y * SRC_W;
                if (diff) {
                    for (int x = x0; x < x1; x++) {
                        int d = (int)pc[x] - (int)pb[x];
                        sum += (uint32_t)(d < 0 ? -d : d);
                    }
                } else {
                    for (int x = x0; x < x1; x++) sum += pc[x];
                }
            }
            uint32_t v = sum / (uint32_t)((y1 - y0) * (x1 - x0));
            if (diff) {
                v = (v * gain_q4) >> 4;
                if (v > 255) v = 255;
            }
            field[j * OUT_W + i] = (uint8_t)v;
        }
    }
}

/*
 * Sigma-delta background estimation: step each pixel one level towards the
 * current frame. No second buffer and no fixed-point drift - a running average
 * in 8 bits would stall, because (cur - bg) >> 7 is zero for any difference
 * under 128 and the model would simply never move.
 *
 * At 30 fps with bg_period 4 the model travels ~7 levels per second, so a
 * lighting change is absorbed in a few seconds while someone walking through
 * is not.
 */
static void bg_step(const uint8_t *cur) {
    for (int i = 0; i < SRC_W * SRC_H; i++) {
        int d = (int)cur[i] - (int)bg[i];
        /* Leave anything that currently reads as foreground alone. Without
         * this, somebody who stands still is absorbed into the background in a
         * few seconds and their silhouette dissolves - which is precisely the
         * case this has to survive. A background that really did change for
         * good is what 'b' is for. */
        if (bg_guard && (d > bg_guard || d < -bg_guard)) continue;
        if (d > 0) bg[i]++;
        else if (d < 0) bg[i]--;
    }
}

// ---------------------------------------------------------------- transmit

static void send_frame(uint8_t type, uint16_t w, uint16_t h, const uint8_t *data, uint16_t len) {
    uint8_t hdr[12];
    uint16_t crc = crc16(data, len);
    hdr[0] = MAGIC0;      hdr[1] = MAGIC1;
    hdr[2] = type;        hdr[3] = seq++;
    hdr[4] = (uint8_t)w;  hdr[5] = (uint8_t)(w >> 8);
    hdr[6] = (uint8_t)h;  hdr[7] = (uint8_t)(h >> 8);
    hdr[8] = (uint8_t)len; hdr[9] = (uint8_t)(len >> 8);
    hdr[10] = (uint8_t)crc; hdr[11] = (uint8_t)(crc >> 8);
    Serial.write(hdr, sizeof hdr);
    Serial.write(data, len);
}

// ---------------------------------------------------------------- commands

static void status(void) {
    Serial.printf("\n# mode=%s bg=%s period=%u guard=%u gain=%u/16 preview=%u fps_cap=%s\n",
                  send_diff ? "diff" : "raw",
                  bg_valid ? "set" : "unset",
                  bg_period, bg_guard, gain_q4, prev_every,
                  min_period ? String(1000 / min_period).c_str() : "free");
    sensor_t *s = esp_camera_sensor_get();
    if (s) Serial.printf("# aec=%u agc=%u exposure=%d gain=%d\n",
                         s->status.aec, s->status.agc,
                         s->status.aec_value, s->status.agc_gain);
}

static void command(const char *line) {
    sensor_t *s = esp_camera_sensor_get();
    int v = atoi(line + 1);

    switch (line[0]) {
    case 'b': recalibrate((uint32_t)v);                                                break;
    case 'B': bg_reset();             Serial.println("# recapture, keep exposure"); break;
    case 'a': bg_period = (uint8_t)constrain(v, 0, 255);                               break;
    case 's': bg_guard  = (uint8_t)constrain(v, 0, 255);                               break;
    case 'g': gain_q4   = (uint8_t)constrain(v, 1, 255);                               break;
    case 'p': prev_every = (uint8_t)constrain(v, 0, 255); prev_count = 0;               break;
    /* Skip the separator: the argument arrives as "m r", and testing line[1]
     * saw the space, so this silently never left difference mode. */
    case 'm': { const char *a = line + 1;
                while (*a == ' ' || *a == '\t') a++;
                send_diff = (*a != 'r');
                Serial.printf("# mode %s\n", send_diff ? "diff" : "raw"); }        break;
    case 'f': min_period = v > 0 ? (uint16_t)(1000 / v) : 0;                           break;

    /* Manual overrides, for when the automatic pass lands somewhere you do not
     * want. Setting either implies locking, and both shift the whole image, so
     * the background model has to be rebuilt after. */
    case 'x': if (s) { s->set_exposure_ctrl(s, v != 0); s->set_gain_ctrl(s, v != 0); }
              auto_lock_pending = false; bg_reset();                                   break;
    case 'e': if (s) { s->set_exposure_ctrl(s, 0); s->set_aec_value(s, constrain(v, 0, 1200)); }
              auto_lock_pending = false; bg_reset();                                   break;
    case 'n': if (s) { s->set_gain_ctrl(s, 0); s->set_agc_gain(s, constrain(v, 0, 30)); }
              auto_lock_pending = false; bg_reset();                                   break;

    case '?': status();                                                                break;
    default:  Serial.printf("# ? '%c'\n", line[0]);                                     break;
    }
}

static void poll_commands(void) {
    static char line[32];
    static uint8_t len;

    while (Serial.available()) {
        int c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (len) { line[len] = 0; command(line); len = 0; }
        } else if (len < sizeof line - 1) {
            line[len++] = (char)c;
        }
    }
}

// ---------------------------------------------------------------- setup / loop

void setup() {
    Serial.setTxBufferSize(8192);   /* a frame is 2280 B; do not block the loop */
    Serial.setRxBufferSize(256);
    Serial.begin(LINK_BAUD);

    camera_config_t c = {};
    c.ledc_channel = LEDC_CHANNEL_0;
    c.ledc_timer   = LEDC_TIMER_0;
    c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;
    c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
    c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;
    c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
    c.pin_xclk  = XCLK_GPIO_NUM;  c.pin_pclk  = PCLK_GPIO_NUM;
    c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href  = HREF_GPIO_NUM;
    c.pin_pwdn  = PWDN_GPIO_NUM;  c.pin_reset = RESET_GPIO_NUM;
    c.pin_sccb_sda = SIOD_GPIO_NUM;     /* the sscb_ spelling is deprecated */
    c.pin_sccb_scl = SIOC_GPIO_NUM;
    c.xclk_freq_hz = 20000000;
    c.pixel_format = PIXFORMAT_GRAYSCALE;
    c.frame_size   = FRAMESIZE_QQVGA;
    c.fb_count     = 2;
    c.grab_mode    = CAMERA_GRAB_LATEST;
    /* QQVGA greyscale is 19.2 KB a frame, so two of them plus the background
     * model fit internal DRAM with room to spare. Keeping them out of PSRAM is
     * faster and means the build does not depend on PSRAM being enabled. */
    c.fb_location  = CAMERA_FB_IN_DRAM;

    esp_err_t err = esp_camera_init(&c);
    if (err != ESP_OK) {
        Serial.printf("\n# camera init failed: 0x%x\n", err);
        while (true) delay(1000);
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_hmirror(s, 0);
        s->set_vflip(s, 0);
    }

    spans_init();
    recalibrate(0);     /* expose automatically, lock, then seed the model */
    Serial.println("\n# esp32cam_sender ready, '?' for status");
}

void loop() {
    static uint32_t next_ms, frames;

    poll_commands();

    if (min_period && (int32_t)(millis() - next_ms) < 0) return;

    /* Only start a frame the link can actually swallow. This throttles to the
     * UART rate on its own and keeps latency at one frame instead of letting
     * the TX buffer build a queue. */
    if (Serial.availableForWrite() < (int)(12 + sizeof field)) return;

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return;

    if (fb->len >= SRC_W * SRC_H) {
        if (!bg_valid) {
            if ((int32_t)(millis() - bg_delay_until) < 0) {
                /* counting down so you can get out of shot; raw keeps streaming */
            } else if (bg_settle) {
                bg_settle--;
            } else if (auto_lock_pending) {
                sensor_t *s = esp_camera_sensor_get();
                if (s) {
                    s->set_exposure_ctrl(s, 0);     /* holds at what AEC found */
                    s->set_gain_ctrl(s, 0);
                }
                auto_lock_pending = false;
                bg_settle = 8;                      /* let the lock take effect */
                Serial.println("# exposure locked");
            } else {
                memcpy(bg, fb->buf, SRC_W * SRC_H);
                bg_valid = true;
                Serial.println("# bg captured");
            }
        }

        /* Send the plain downscale until the model is good, so the view never
         * goes dark and the frame type says which one you are looking at. */
        bool diff = send_diff && bg_valid;
        downscale(fb->buf, diff);
        send_frame(diff ? TYPE_DIFF54 : TYPE_RAW54, OUT_W, OUT_H, field, sizeof field);

        /*
         * The full preview is 19,212 bytes and the TX buffer is 8 KB, so the
         * "is there room" test the 54x42 frames use can never pass for it -
         * that is why the preview never appeared. Write it blocking instead and
         * rate-limit it: one costs ~208 ms of link at 921600, so every Nth
         * frame keeps the small stream alive underneath it.
         */
        if (prev_every && ++prev_count >= prev_every) {
            prev_count = 0;
            send_frame(TYPE_PREVIEW, SRC_W, SRC_H, fb->buf, SRC_W * SRC_H);
        }

        if (bg_period && ++frames >= bg_period) {
            frames = 0;
            bg_step(fb->buf);
        }
    }

    esp_camera_fb_return(fb);
    next_ms = millis() + min_period;
}

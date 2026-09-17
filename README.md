# pico_esp32-cam_ftdi

An RP2040-Zero standing in for the FTDI adapter an AI-Thinker ESP32-CAM would
otherwise need. It enumerates as a USB CDC-ACM port, bridges it to a PIO UART,
follows the host's baud rate, and reproduces the DTR/RTS reset circuit esptool
expects — so `esptool` and the Arduino IDE flash the ESP32 with no buttons
pressed and no IO0 jumper.

## Quick start

Two boards to flash and one command to run. [firmware/](firmware/) holds both
images already built, so none of the toolchains below are needed to try it -
see [firmware/README.md](firmware/README.md) for the wiring and the order.

```bash
# RP2040: hold BOOT, plug in, copy firmware/pico_esp32-cam_ftdi.uf2 to RPI-RP2
# ESP32:  jumper GP3-GND, press its RST, then
esptool --chip esp32 --port COM6 --baud 460800 write-flash -z   0x1000 firmware/bootloader.bin 0x8000 firmware/partitions.bin   0xe000 firmware/boot_app0.bin 0x10000 firmware/firmware.bin
# then remove the jumper and press RST again - IO0 is the camera's XCLK

uv run tools/posture_viewer.py --source esp        # lists the ports
uv run tools/posture_viewer.py --source esp --port COM7
```

Two COM ports appear. The one to give the viewer is **Vision Stream**,
interface 2; interface 0 is the bridge esptool talks to. Running without
`--port` prints both and says which is which.

Then: SPACE and step out of shot, sit upright and press SPACE again, and it
judges. [The posture section](#posture) covers what the panel is showing and
what to do when the numbers look wrong.

Without `uv`: `pip install pyserial numpy opencv-python` and run
`tools/posture_viewer.py` with any Python 3.10+.

## The bridge

The UART is on PIO, not either hardware UART, so **all four signal pins are
free choices**: edit the defines at the top of
[pico_esp32-cam_ftdi.c](pico_esp32-cam_ftdi.c) and rewire. Both hardware UARTs
stay unused and available.

## Wiring — GP0 to GP3

| RP2040-Zero | ESP32-CAM (AI-Thinker) | |
|---|---|---|
| 5V | 5V | not the 3V3 pin — that bypasses the module's regulator |
| GND | GND | common ground is mandatory |
| GP0 | U0R (GPIO3) | PIO UART TX |
| GP1 | U0T (GPIO1) | PIO UART RX |
| GP2 | IO0 | open drain |
| GP3 | — | jumper to **GND** while flashing |

GP16 drives the on-board status LED and needs no wiring. Both boards are 3V3,
so no level shifting.

GP3 is an input with an internal pull-up — it is only ever read, never driven,
so the jumper carries about 60 µA.

### EN is not wired

It is not on the AI-Thinker header; the only electrical access is the RST
button pad underneath the module. Rather than solder to it, reset by hand:

1. Jumper **GP3 to GND** — the LED turns bright magenta and IO0 is held low.
2. Press the module's own RST button (underside).
3. `esptool --before no_reset --after no_reset ... write_flash ...`
4. Remove the jumper, press RST again.

The jumper is needed because the DTR/RTS rule below deliberately releases IO0
whenever the host asserts both lines, and esptool only drives IO0 low for about
50 ms inside its reset sequence — too narrow to hit with a button press.
**Take the jumper out before the final reset: IO0 is the camera's XCLK.**

**Power is the usual failure.** The ESP32-CAM pulls 250–310 mA in bursts and
browns out long before the UART does anything wrong. Put **470 µF + 0.1 µF**
across 5V/GND at the ESP32-CAM end, and prefer a separate 5V supply with only
GND shared.

If you cannot reach the EN pad, leave GP3 unconnected and enter the bootloader
by hand: jumper IO0 to GND, tap RST, flash, remove the jumper, tap RST again.

## Flashing, in order

The RP2040 has to go first: it is the thing that programs the ESP32.

### 1. RP2040-Zero

Hold **BOOT** while plugging in the USB cable. An `RPI-RP2` drive appears; copy
`build/pico_esp32-cam_ftdi.uf2` onto it. The board reboots and two COM ports
show up.

After the first time the BOOT button is optional — the firmware answers the
Arduino 1200 bps touch, so this reboots it into the bootloader:

```bash
uv run --with pyserial python -c "import serial; serial.Serial('COM4', 1200).close()"
```

### 2. ESP32-CAM

`uv run tools/viewer.py --list` labels the two ports. Use the **esptool** one
here, and close the viewer first — a serial port has one owner.

[esp32cam_sender/platformio.ini](esp32cam_sender/platformio.ini) builds it.
The partition scheme is `huge_app` there because the camera driver does not fit
the default table, and `src_dir = .` keeps the Arduino layout so the IDE can
open the same sketch.

1. Jumper **GP3 to GND**. The RP2040's LED turns bright magenta.
2. Press the module's own **RST** button, underneath. The ROM bootloader waits
   indefinitely, so there is no window to hit — the compile can take as long as
   it likes.
3. ```bash
   pio run -t upload
   ```
4. **Remove the jumper**, press RST again.

Step 4 is not optional: IO0 is the camera's XCLK, and the camera cannot start
while the jumper holds it down.

PlatformIO's default `--before default_reset --after hard_reset` is harmless
here: the reset half only toggles RTS, which goes nowhere with EN unwired, and
the jumper holds IO0 down throughout. If the sync is flaky, lower
`upload_speed` — the bridge follows whatever rate the host asks for.

Flashing by hand needs four images, not one — the app alone leaves a partition
table that no longer matches:

```bash
uv run --with esptool esptool --chip esp32 --port COM4 --baud 921600 write-flash -z 0x1000 .pio/build/esp32cam/bootloader.bin 0x8000 .pio/build/esp32cam/partitions.bin 0xe000 <core>/tools/partitions/boot_app0.bin 0x10000 .pio/build/esp32cam/firmware.bin
```

### 3. Check it

```bash
uv run tools/viewer.py COM5
```

COM5 being the **Vision Stream** port. Frames only appear there — the vision
stage consumes them off the wire, so the bridge port now carries just the
ESP32's log text.

## Why the control lines are open drain

`IO0` is also the camera's **XCLK** on this module. Once the ESP32 is running,
the pin has to be *released*, not driven high, or the RP2040 fights the clock
output. So GP2/GP3 are only ever driven low; otherwise they sit in Hi-Z and the
ESP32's own pull-ups do the work.

## Why DTR and RTS cross-couple

esptool's classic reset (from its own `reset.py`) means:

```
RTS asserted -> EN  low   (hold in reset)
DTR asserted -> IO0 low   (boot the serial bootloader)
```

EN is not wired here, so only the DTR half does anything — but the real
two-transistor adapter circuit cross-couples the pair so that asserting **both**
drives neither, and that half matters on its own: Windows and many Linux
drivers raise DTR and RTS together the moment a port is opened, and without the
cross-coupling that would hold IO0 low, shorting the camera's XCLK.

```c
io0_low = force_boot || ((dtr != rts) && dtr);
```

## The PIO UART

[pio_uart.pio](pio_uart.pio), adapted from pico-examples. 8 PIO cycles per bit,
so `clkdiv = clk_sys / (8 × baud)` — 300 baud to 6 Mbps sits comfortably inside
the 16.8-bit divider's range.

* **TX** is 4 instructions: side-set holds the line high while stalled on
  `pull`, drives the start bit for 8 cycles, then shifts 8 bits out LSB-first.
  80 cycles per frame, exactly 10 bit times.
* **RX** is 9 instructions. `wait 0 pin` catches the start edge, `set x,7 [10]`
  delays to the middle of bit 0, and each bit is sampled 8 cycles later — the
  stop-bit check lands at cycle 76 of 80, leaving 2 cycles of slack for a
  transmitter running slightly fast. A bad stop bit sets PIO IRQ 4 (internal
  only, never reaches the NVIC) and the byte is discarded rather than pushed.

13 of PIO0's 32 instruction slots and 2 of its 4 state machines. The LED runs
on PIO1, so PIO0 keeps 2 free state machines for a later camera link.

### RX is DMA'd into a hardware-wrapped ring

The PIO RX FIFO is 8 entries even joined — 87 µs of slack at 921600 baud,
versus the 347 µs a PL011's 32-byte FIFO would give. That is still far more
than any interrupt latency here, but DMA removes the question entirely and ends
up *better* than the hardware UART, because the whole 4 KiB ring becomes the
buffer (~43 ms at 921600).

Two channels, no ISR:

* The data channel does 8-bit transfers from `&pio->rxf[sm] + 3` — the program
  shifts right with an explicit `push`, so the byte lands in bits 31:24, and a
  byte read of that lane still pops the whole FIFO entry. `channel_config_set_ring`
  wraps the write address in hardware, which is why `rx_ring` is 4096-byte
  aligned.
* When it finishes a lap its write address has already wrapped to the start, so
  rearming it is one word written to `al1_transfer_count_trig`. A second
  channel chained from the first does exactly that, so the ring never stops
  accepting bytes.

`rx_head()` reads the live `write_addr`; the tail is software. A `__dmb()`
between the pointer read and the buffer read keeps the byte visible behind it.

## Status LED (on-board WS2812, GP16)

| Colour | Meaning |
|---|---|
| magenta (bright) | GP3 boot jumper is in — IO0 held low |
| red | USB not enumerated |
| magenta (dim) | esptool is driving IO0 low |
| cyan | data moving |
| green | port open |
| blue (dim) | idle |

## Build

The Git Bash / MSYS shell mangles the linker's `-L` paths and the link fails
with `cannot open linker script file memmap_default.incl`. **Build from
PowerShell** (or the VS Code Pico extension):

```powershell
$env:Path = "$env:USERPROFILE\.pico-sdk\cmake\v4.3.4\bin;$env:USERPROFILE\.pico-sdk\ninja\v1.13.2;$env:USERPROFILE\.pico-sdk\toolchain\15_2_Rel1\bin;" + $env:Path
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Flash `build/pico_esp32-cam_ftdi.uf2` with BOOT held while plugging in.
Roughly 44 KB of flash and 68 KB of RAM - the vision stage's frame buffers are most of the latter, since `in_buf` and `out_buf` are each sized for a whole 160x120 preview.

## Use

Arduino IDE: board **AI Thinker ESP32-CAM**, port = the RP2040's COM port.

esptool:

```bash
esptool --chip esp32 --port COM5 --baud 460800 write_flash -z 0x1000 firmware.bin
```

The bridge follows whatever baud the host sets, up to 6 Mbps. 460800 and
921600 are both comfortable.

## The vision stage

[vision.c](vision.c) turns the ESP32's 54x42 coverage field into a mask and a
skeleton and publishes all three on CDC 1.

```
coverage -> threshold -> fill holes -> open -> [close] -> Zhang-Suen -> skeleton
```

**Hole filling rather than a morphological closing.** At 54x42 a person's legs
are one cell apart, and a single dilation bridges that gap: the legs fuse and
the skeleton loses both limbs. Flooding the background inwards from the border
and promoting whatever it never reached repairs a hole of any size and *cannot*
join things that were separate, because it only touches background the outside
could not get to. Closing is still available (`l<n>`) but defaults to off.

**Filling runs before opening.** An erosion widens a hole, and a hole within one
cell of the silhouette's edge becomes a notch open to the background — no longer
interior, so nothing can fill it afterwards. Sealing first costs nothing.

Watch out for `o` on a thin subject: an opening severs any connection a single
cell wide, so a narrow neck detaches the head into its own blob.

### Commands on CDC 1

Anything this board does not recognise is forwarded to the ESP32 verbatim, so
the sketch's own letters keep working from the same terminal.

| | |
|---|---|
| `t<n>` | threshold, 0–255 |
| `o<n>` | opening iterations |
| `h<0\|1>` | fill holes |
| `l<n>` | closing iterations |
| `k<0\|1>` | skeleton layer |
| `v<0\|1>` | coverage layer |
| `u<baud>` | UART rate used while CDC 0 is closed |
| `?` | status from both boards |

### Baud

Two things want to own the UART rate: esptool moves it mid-session, and the
vision link needs a fixed rate that survives nobody having CDC 0 open. So CDC 0's
line coding applies only while CDC 0 is actually open, and 921600 is restored
the moment it closes.

### Host test

The pipeline is hardware-independent, so it runs on a PC with USB stubs — it
builds a synthetic figure with speckle and an interior hole, then checks the
speckle is gone, the hole is filled, the skeleton is one pixel wide everywhere,
thinning preserved the component count, and all three emitted frames carry
valid CRCs.

## Viewer

[tools/viewer.py](tools/viewer.py) shows the 54x42 field live and sends tuning
commands back down the same port.

```bash
uv run tools/viewer.py --list      # label the two ports
uv run tools/viewer.py COM5        # the Vision Stream one
```

It carries PEP 723 dependency metadata, so `uv run` builds the environment on
first use and nothing needs installing. With a plain interpreter instead:
`pip install pyserial numpy opencv-python`.

Opening the port is what sets the bridge's UART rate — nothing else configures
it — so `--baud` has to match [esp32cam_sender](esp32cam_sender/esp32cam_sender.ino).

During bring-up point it at the bridge port (CDC 0). The RP2040 is still a
transparent pipe there, so frames and the ESP32's text log arrive interleaved —
ASCII never contains 0xA5, so the two can never be confused — and keystrokes go
straight to the sketch.

| key | goes to | |
|---|---|---|
| `[` `]` | RP2040 | threshold; down to 0 means Otsu picks it |
| `i` | RP2040 | invert — subject darker than background |
| `o` `O` | RP2040 | opening iterations |
| `h` | RP2040 | fill holes |
| `k` | RP2040 | skeleton layer |
| `f` | ESP32 | freeze the background model |
| `b` | ESP32 | re-expose and recapture background |
| `d` `r` | ESP32 | difference / raw downscale |
| `p` | ESP32 | 160x120 camera window |
| `g` `G` | ESP32 | difference gain |
| `e` `E` | ESP32 | exposure |
| `n` `N` | ESP32 | sensor gain |
| `x` | ESP32 | auto exposure and gain |
| `a` | viewer | stretch the coverage panel to its own range |
| `c` | viewer | colour map |
| `s` | viewer | save a PNG |
| `/` | both | print status |
| `q` | | quit |

### Just the camera

```bash
uv run tools/camview.py COM5
```

One window, the 160x120 picture, and the numbers that say whether it is usable.
Framing, focus and exposure are settled at the camera and no later stage can
recover from getting them wrong, so it is worth being able to look at that on
its own without four panels and twenty keys in the way. It switches the preview
stream on at startup and off again on exit, and borrows viewer.py's frame
parser rather than keeping a second copy of it.

## Other design notes

* **Backpressure comes from leaving bytes in the CDC FIFO.** The USB→UART path
  only pulls a packet when the 2 KiB TX ring can take a whole one; that stalls
  the host, which is the only flow control an ESP32-CAM link has (no RTS/CTS).
* **`CFG_TUD_CDC_RX_BUFSIZE` is 1024, not TinyUSB's default 64.** esptool sends
  a whole 1 KB flash block at a time; at 64 bytes the device NAKs constantly
  and throughput roughly halves.
* **A baud change drains TX first, then restarts the RX state machine.**
  esptool switches rate right after a command it expects to have gone out in
  full, and a byte half-received at the old rate would otherwise be completed
  at the new one. PIO has no "transmitter idle" flag, so the drain waits out
  one frame time after the FIFO empties.
* **8N1 only.** Data bits, parity and stop bits from the host's line coding are
  ignored — it is all the ESP32 ROM bootloader ever uses, and encoding the rest
  would cost PIO instructions for nothing.
* Raw TinyUSB rather than `pico_stdio_usb`, because stdio does not expose
  DTR/RTS or the line coding — which is the entire point here.

## Posture

The posture stage answers "what is the person doing" as a label plus two
scalars - `phi` (focused) and `delta` (fatigued) - for a state machine
downstream.

```bash
uv run tools/posture_viewer.py --source camera   # webcam stand-in, no board
uv run tools/posture_viewer.py --source esp --port COM5
uv run tools/test_posture.py                     # synthetic check, no board
```

### Calibration is two layers, in order

Press SPACE and the viewer counts eight seconds for you to get out of shot,
then tells the *sensor* to rebuild its own background - `b`, which on the ESP32
also re-runs AEC and locks the exposure it lands on. Only once that reports
back does it take the zone reference the judgement compares against.

The order is the whole point. The ESP32's background is what turns a picture
into a coverage field; seed it with you still in the chair and you become part
of the background, after which nothing downstream can find you. Pressing SPACE
again during the countdown skips it, for when the desk is already empty.

| | |
|---|---|
| SPACE | next step (during the countdown: capture now) |
| `n` / `b` | redo background / redo posture baseline |
| `1`–`4`, `0` | tag the log with a ground-truth label, stop tagging |
| `v` | cycle the left panel: coverage / zone / mask / skeleton |
| `c` `a` `g` | palette / contrast stretch / grid |
| `s` / `p` | save a PNG / the camera photo window |
| `[` `]` | threshold, 0 means Otsu picks it |
| `o` `O` | opening iterations |
| `h` / `i` | fill holes / invert |
| `-` `=` | difference gain |
| `,` `.` | exposure — the board rebuilds its background, so redo step 1 |
| `x` | auto exposure and gain |
| `/` | status from both boards |

This is the only tool that judges. [tools/viewer.py](tools/viewer.py) stays as
the tuning view - it draws mask and skeleton as they come off the wire and
does no judgement at all, which is what its four plain panels are for.

The left panel draws the **coverage field**, not the reconstructed distances:
coverage is what the board actually streams and it carries a gradient, while
the distance array is the mask pushed back through an apparent-size estimate
and so holds two values. `v` switches to it when you want to see exactly what
the judgement sees.

The panel's link line is there because a dead link looks like a working one -
the last frame keeps being drawn. It shows the board's own log (`# bg
captured`, `# exposure locked`) while things are healthy, and says so when the
mask stops arriving or the reader thread stops.

### The sensor is the only swappable part

[tools/posture.py](tools/posture.py) takes one thing: a 54x42 array of
distances. Everything that differs between sensors lives in an adapter that
produces that array, so the judgement is written once.

| | |
|---|---|
| [tools/esp_source.py](tools/esp_source.py) | bridge frames -> distances |
| [tools/camera_source.py](tools/camera_source.py) | a webcam through the ESP32's and vision.c's stages, for when the boards are elsewhere |
| *(later)* | a real depth sensor, unchanged downstream |

**Neither adapter has depth**, so both estimate it from apparent size: torso
width, because shoulders keep their width through a slump and only distance
moves them. The distances that come out are geometrically right and
absolutely approximate - fine for telling leaning back from folding forward,
not to be read as millimetres.

### Why it is built this way

**Head height cannot separate slumping from reclining.** Both drop the head in
the image - leaning back sinks it exactly as folding forward does. So head
height only measures *how far* from the reference posture, and the change in
apparent size decides *which way*.

**Slumping and nodding differ in time, not in shape.** At the bottom of a nod
the geometry is a slump. Slumping is therefore defined as staying down, and
only counts after three continuous seconds, which a nod never reaches.

**Everything is relative to a captured reference posture.** That is what lets
one set of thresholds cover different people and sitting distances - nothing is
calibrated in pixels.

### Before tuning

A head touching row 0 means the frame cut it off, and a reference captured that
way reads every later posture as "head has dropped". The panel says
`baseline clipped` when it happens; tilt the camera down and recapture.

The judgement is only as good as the mask. If `coverage` is an outline with
nothing inside it, that is the subject matching the background in brightness,
and no threshold recovers what was never there.

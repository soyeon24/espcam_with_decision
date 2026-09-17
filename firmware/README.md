# Prebuilt firmware

Flash these and skip both toolchains. Nothing here needs the Pico SDK, CMake,
PlatformIO or a compiler - only `esptool` for the ESP32 half, and the RP2040
half is a file copy.

| file | board | |
|---|---|---|
| `pico_esp32-cam_ftdi.uf2` | RP2040-Zero | bridge + vision stage |
| `bootloader.bin` `partitions.bin` `boot_app0.bin` `firmware.bin` | ESP32-CAM | camera, background subtraction, 54x42 field |

The ESP32 takes four images, not one: flashing the application alone leaves a
partition table that no longer matches it.

Rebuild from source and these go stale - the parent README has both build
recipes. Check `git log firmware/` against the source if you are unsure.

## 1. RP2040-Zero

Hold **BOOT**, plug in USB, and an `RPI-RP2` drive appears. Copy
`pico_esp32-cam_ftdi.uf2` onto it. It reboots itself and two COM ports come up.

## 2. ESP32-CAM

The RP2040 is what programs it, so do it in this order.

1. Close anything holding a COM port - a port has one owner.
2. Jumper the RP2040's **GP3 to GND**. Its LED turns bright magenta.
3. Press the ESP32-CAM's own **RST** button, underneath the module. The ROM
   bootloader then waits indefinitely, so there is no window to hit.
4. Flash, against the **bridge** port - interface 0, the one
   `uv run tools/posture_viewer.py --source esp` labels for you:

   ```bash
   esptool --chip esp32 --port COM6 --baud 460800 write-flash -z \
     0x1000 bootloader.bin 0x8000 partitions.bin \
     0xe000 boot_app0.bin 0x10000 firmware.bin
   ```

   Lower `--baud` if the sync is flaky; the bridge follows whatever the host asks
   for, and its UART is a PIO program with less margin than a hardware one.
5. **Remove the jumper**, press RST again.

Step 5 is not optional. IO0 is the camera's XCLK, and while the jumper holds it
down `esp_camera_init` fails and the sketch stops there - no frames, no log, a
board that looks dead.

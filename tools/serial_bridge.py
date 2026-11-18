"""
Serial-to-WebSocket Audio Bridge with Bidirectional Control
Reads audio samples from ESP32 via USB Serial and streams to ws://localhost:8000/ws/audio.
Also listens for server control commands and signals the ESP32 to blink its blue LED.
"""

import argparse
import asyncio
import json
import logging
import sys
import time

import numpy as np
import serial
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("serial_bridge")


async def serial_reader_task(ser, ws):
    """Reads serial samples from ESP32 and streams as PCM bytes to WebSocket."""
    buffer = []
    last_flush = time.time()

    while True:
        try:
            if not ser.is_open:
                logger.error("Serial port closed unexpectedly.")
                break

            line = ser.readline()
            if line:
                try:
                    val = int(line.decode("utf-8", errors="ignore").strip())
                    val = max(-32768, min(32767, val))
                    for _ in range(8):
                        buffer.append(val)
                except ValueError:
                    pass

            if len(buffer) >= 1024 or (time.time() - last_flush > 0.064 and len(buffer) > 128):
                samples_arr = np.array(buffer, dtype=np.int16)
                pcm_bytes = samples_arr.tobytes()
                await ws.send(pcm_bytes)
                buffer = []
                last_flush = time.time()

            await asyncio.sleep(0.001)
        except serial.SerialException as e:
            logger.error(f"Serial error in reader: {e}")
            break
        except Exception as e:
            logger.error(f"Error in serial reader: {e}")
            break


async def ws_receiver_task(ser, ws):
    """Listens for control commands from WebSocket server and commands ESP32 LED."""
    while True:
        try:
            msg_text = await ws.recv()
            if isinstance(msg_text, str):
                try:
                    data = json.loads(msg_text)
                    if data.get("type") == "control":
                        cmd = data.get("command")
                        if cmd == "start_recording":
                            dur_ms = int(data.get("duration", 30.0) * 1000)
                            logger.info(f"ESP32 LED: REC:START:{dur_ms}ms")
                            ser.write(f"REC:START:{dur_ms}\n".encode("utf-8"))
                            ser.flush()
                        elif cmd == "stop_recording":
                            logger.info("ESP32 LED: REC:STOP")
                            ser.write(b"REC:STOP\n")
                            ser.flush()
                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket closed by server.")
            break
        except Exception as e:
            logger.error(f"Error in WS receiver: {e}")
            break


async def stream_serial_to_ws(serial_port: str, baud_rate: int, ws_url: str):
    logger.info("=" * 60)
    logger.info(f"  ESP32 Serial Bridge (with Blue LED Blink Control)")
    logger.info(f"  Serial Port: {serial_port} ({baud_rate} baud)")
    logger.info(f"  Target Server: {ws_url}")
    logger.info("=" * 60)

    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=0.05)
    except Exception as e:
        logger.error(f"Failed to open serial port {serial_port}: {e}")
        return

    logger.info(f"Connected to ESP32 on {serial_port}!")

    while True:
        try:
            async with websockets.connect(ws_url, ping_interval=10, ping_timeout=30) as ws:
                logger.info(f"Connected to {ws_url}! Live audio streaming & LED control active.")
                reader = asyncio.create_task(serial_reader_task(ser, ws))
                receiver = asyncio.create_task(ws_receiver_task(ser, ws))
                done, pending = await asyncio.wait(
                    [reader, receiver],
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check if serial port is still valid
                if not ser.is_open:
                    logger.error("Serial port lost. Attempting to reopen...")
                    try:
                        ser = serial.Serial(serial_port, baud_rate, timeout=0.05)
                        logger.info("Serial port reopened successfully.")
                    except Exception as e:
                        logger.error(f"Could not reopen serial port: {e}")
                        logger.info("Waiting 5s before retry...")
                        await asyncio.sleep(5)

        except websockets.exceptions.ConnectionRefused:
            logger.warning(f"Server not available at {ws_url}. Retrying in 3s...")
        except Exception as e:
            logger.error(f"Connection error: {e}")

        await asyncio.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP32 Serial-to-WebSocket Audio Bridge")
    parser.add_argument("--port", default="/dev/cu.usbserial-0001", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--ws-url", default="ws://localhost:8000/ws/audio", help="WebSocket server URL")
    args = parser.parse_args()

    try:
        asyncio.run(stream_serial_to_ws(args.port, args.baud, args.ws_url))
    except KeyboardInterrupt:
        logger.info("Serial bridge stopped.")

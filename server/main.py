"""
FastAPI Server for Cough Prescreening System
Endpoints:
  - GET  /         : Serves static/index.html
  - WS   /ws/audio : Binary PCM audio ingestion from ESP32
  - WS   /ws/ui    : Real-time telemetry, levels, and predictions to Web UI
  - GET  /api/status : Health and inference engine diagnostics
  - GET  /api/events : Recent detection event history
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from typing import Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.config import config
from server.model import CoughClassifierEngine
from server.pipeline import AudioPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cough_screener.main")

# Initialize Classifier Engine and Audio Pipeline
classifier = CoughClassifierEngine()
pipeline = AudioPipeline(classifier=classifier)

# Connected UI WebSocket clients
ui_clients: Set[WebSocket] = set()
ui_clients_lock = asyncio.Lock()

# Connected Audio WebSocket clients (ESP32 or serial_bridge)
audio_clients: Set[WebSocket] = set()
audio_clients_lock = asyncio.Lock()


def broadcast_to_ui_sync(message: Dict):
    """Bridge synchronous callback from pipeline thread to asyncio event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    asyncio.run_coroutine_threadsafe(_broadcast_to_ui(message), loop)


async def _broadcast_to_ui(message: Dict):
    """Sends JSON message to all connected web clients."""
    if not ui_clients:
        return

    text_data = json.dumps(message)
    dead_clients = []
    
    async with ui_clients_lock:
        for ws in ui_clients:
            try:
                await ws.send_text(text_data)
            except Exception:
                dead_clients.append(ws)

        for ws in dead_clients:
            if ws in ui_clients:
                ui_clients.remove(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("==================================================")
    logger.info("  Starting Cough Prescreening Server")
    logger.info(f"  Target: http://{config.HOST}:{config.PORT}")
    logger.info(f"  Sample Rate: {config.SAMPLE_RATE} Hz (16-bit PCM)")
    logger.info(f"  Model Engine: {'DUMMY CLASSIFIER (Simulated)' if classifier.is_dummy else 'PYTORCH CNN (Trained)'}")
    logger.info("==================================================")

    # Wire pipeline callbacks to UI broadcast
    pipeline.subscribe_level(broadcast_to_ui_sync)
    pipeline.subscribe_event(broadcast_to_ui_sync)

    # Start audio background analysis loop
    await pipeline.start()

    yield

    # Shutdown
    logger.info("Stopping Audio Pipeline...")
    await pipeline.stop()
    async with ui_clients_lock:
        for ws in list(ui_clients):
            try:
                await ws.close()
            except Exception:
                pass
        ui_clients.clear()
    logger.info("Cough Prescreening Server stopped.")


app = FastAPI(
    title="Cough Prescreening Server",
    description="Live acoustic pipeline based on MIT COVID-19 AI Cough paper",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
if config.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main real-time dashboard."""
    index_file = config.STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse(
            content="<h1>404: server/static/index.html not found</h1>",
            status_code=404,
        )
    return FileResponse(index_file)


@app.get("/api/status")
async def get_status():
    """Health status and telemetry."""
    return {
        "status": "online",
        "sample_rate": config.SAMPLE_RATE,
        "is_dummy_model": classifier.is_dummy,
        "model_path": str(config.MODEL_PATH),
        "model_exists": config.MODEL_PATH.exists(),
        "has_norm_stats": classifier.has_norm_stats,
        "active_ui_clients": len(ui_clients),
        "total_audio_samples": pipeline.ring_buffer.total_samples_written,
        "events_dir": str(config.EVENTS_DIR),
    }


@app.get("/api/events")
async def list_recent_events(limit: int = 50):
    """Returns the most recent detected event metadata from JSON sidecars."""
    json_files = sorted(config.EVENTS_DIR.glob("event_*.json"), key=os.path.getmtime, reverse=True)
    events = []
    for f in json_files[:limit]:
        try:
            with open(f, "r") as fp:
                events.append(json.load(fp))
        except Exception:
            continue
    return JSONResponse(content={"count": len(events), "events": events})


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ESP32 INMP441 audio streaming.
    Receives binary 16-bit PCM chunks (little-endian, 16 kHz mono).
    """
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"[AUDIO WS] ESP32/Bridge connected from {client_ip}")

    async with audio_clients_lock:
        audio_clients.add(websocket)

    total_bytes = 0
    start_time = asyncio.get_event_loop().time()

    try:
        while True:
            # Can receive bytes or text control ping
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                data = msg["bytes"]
                total_bytes += len(data)
                pipeline.feed_pcm(data)
            elif "text" in msg and msg["text"]:
                pass

    except WebSocketDisconnect:
        elapsed = max(0.1, asyncio.get_event_loop().time() - start_time)
        kbps = (total_bytes * 8) / (elapsed * 1000)
        logger.info(f"[AUDIO WS] Client disconnected ({client_ip}). Streamed {total_bytes / 1024:.1f} KB at {kbps:.1f} kbps.")
    except Exception as e:
        logger.error(f"[AUDIO WS ERROR] Exception on audio socket: {e}")
    finally:
        async with audio_clients_lock:
            if websocket in audio_clients:
                audio_clients.remove(websocket)


@app.websocket("/ws/ui")
async def websocket_ui_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for browser dashboard.
    Pushes live audio level telemetry and cough prediction events.
    Listens for user trigger commands to start recording sessions.
    """
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"[UI WS] Dashboard client connected from {client_ip}")

    async with ui_clients_lock:
        ui_clients.add(websocket)

    try:
        init_msg = {
            "type": "init",
            "message": "Connected to Cough Prescreening Server",
            "is_dummy_model": classifier.is_dummy,
            "sample_rate": config.SAMPLE_RATE,
            "classes": list(config.CLASSES),
            "class_labels": list(config.CLASS_LABELS),
        }
        await websocket.send_text(json.dumps(init_msg))

        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
                action = data.get("action")

                if action == "start_session":
                    logger.info("[UI WS] User triggered recording session (open-ended). Commanding ESP32 blue LED...")

                    # 1. Trigger hardware LED on ESP32 via audio clients
                    control_msg = json.dumps({
                        "type": "control",
                        "command": "start_recording",
                        "duration": 30.0,
                    })
                    async with audio_clients_lock:
                        for ws_audio in list(audio_clients):
                            try:
                                await ws_audio.send_text(control_msg)
                            except Exception:
                                pass

                    # 2. Start open-ended audio accumulation session in pipeline
                    pipeline.start_manual_session()

                    # 3. Broadcast session status to UI clients
                    await _broadcast_to_ui({
                        "type": "session_state",
                        "state": "recording",
                    })

                elif action == "stop_session":
                    symptoms = data.get("symptoms", {})
                    logger.info(f"[UI WS] User stopped recording. Symptoms: {symptoms}")

                    # 1. Stop LED on ESP32
                    control_msg = json.dumps({
                        "type": "control",
                        "command": "stop_recording",
                    })
                    async with audio_clients_lock:
                        for ws_audio in list(audio_clients):
                            try:
                                await ws_audio.send_text(control_msg)
                            except Exception:
                                pass

                    # 2. Stop pipeline recording and trigger analysis
                    stopped = pipeline.stop_manual_session(symptoms)

                    if stopped:
                        # Broadcast analyzing state
                        await _broadcast_to_ui({
                            "type": "session_state",
                            "state": "analyzing",
                        })
                    else:
                        await _broadcast_to_ui({
                            "type": "session_error",
                            "message": "No recording session was active.",
                        })

                elif action == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except Exception as e:
                logger.debug(f"Error handling UI message: {e}")

    except WebSocketDisconnect:
        logger.info(f"[UI WS] Dashboard client disconnected ({client_ip})")
    except Exception as e:
        logger.debug(f"[UI WS] Connection closed: {e}")
    finally:
        async with ui_clients_lock:
            if websocket in ui_clients:
                ui_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        reload=False,
    )

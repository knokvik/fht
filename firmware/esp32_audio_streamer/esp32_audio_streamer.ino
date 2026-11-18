/*
 * ESP32 Audio Streamer for Cough Prescreening Pipeline
 * 
 * Hardware: ESP32 Dev Board + INMP441 I2S MEMS Microphone
 * 
 * Wiring (INMP441 -> ESP32):
 *   VDD  <-->  3.3V  (CAUTION: Do NOT connect to 5V; max rating is 3.6V)
 *   GND  <-->  GND
 *   SCK  <-->  GPIO 14 (Bit Clock / BCLK)
 *   WS   <-->  GPIO 15 (Word Select / LRCLK)
 *   SD   <-->  GPIO 32 (Serial Data Out / DOUT)
 *   L/R  <-->  GND     (Left Channel)
 * 
 * Dependencies:
 *   - Arduino Core for ESP32 (by Espressif)
 *   - WebSockets library by Markus Sattler (links2004) version >= 2.4.0
 *     Install via Arduino IDE Library Manager: Search "WebSockets" -> install "WebSockets by Markus Sattler"
 * 
 * Testing Modes:
 *   - Set USE_SERIAL_LOOPBACK to 1: Reads I2S and prints to Serial Plotter (no WiFi needed).
 *   - Set USE_SERIAL_LOOPBACK to 0: Streams 16 kHz 16-bit PCM audio over WebSocket to PC.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include "driver/i2s.h"

// ============================================================================
// CONFIGURATION FLAGS & NETWORK SETTINGS
// ============================================================================

// Set to 1 for initial mic test using Arduino Serial Plotter (no WiFi needed).
// Set to 0 for live WiFi WebSocket streaming to the Python PC server.
#define USE_SERIAL_LOOPBACK 0

// WiFi Credentials (same hotspot connected to your laptop and ESP32)
const char* WIFI_SSID     = "regmig";
const char* WIFI_PASSWORD = "YOUR_HOTSPOT_PASSWORD";

// PC Server Settings
// Laptop IP on the hotspot network (172.24.167.24)
const char* SERVER_HOST   = "172.24.167.24";
const int   SERVER_PORT   = 8000;
const char* SERVER_PATH   = "/ws/audio";

// ============================================================================
// I2S PIN CONFIGURATION (INMP441)
// ============================================================================
#define I2S_SCK_PIN   14  // Serial Clock (BCLK)
#define I2S_WS_PIN    15  // Word Select (LRCLK)
#define I2S_SD_PIN    32  // Serial Data (DOUT)
#define I2S_PORT      I2S_NUM_0

// Audio Sample Settings
#define SAMPLE_RATE       16000     // 16 kHz standard for speech/respiratory ML
#define CHUNK_SAMPLES     1024      // Number of 16-bit samples per WebSocket frame (2048 bytes)
#define BIT_SHIFT         14        // INMP441 24-bit output in 32-bit slot: shift right by 14
                                    // (14 bits provides +12 dB clean analog gain for cough acoustic capture)

// ============================================================================
// GLOBAL BUFFERS & CLIENTS
// ============================================================================
WebSocketsClient webSocket;
bool isWebSocketConnected = false;

// 32-bit raw DMA buffer from I2S and 16-bit processed PCM output buffer
int32_t raw_i2s_buffer[CHUNK_SAMPLES];
int16_t pcm_buffer[CHUNK_SAMPLES];

// Telemetry counters
unsigned long totalBytesSent = 0;
unsigned long totalFramesSent = 0;
unsigned long lastLogTime = 0;
unsigned long lastRmsPrintTime = 0;

// ============================================================================
// I2S DRIVER INITIALIZATION
// ============================================================================
esp_err_t setupI2S() {
    Serial.println("[I2S] Initializing I2S peripheral...");

    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // INMP441 L/R tied to GND = Left channel
        .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD_PIN
    };

    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[I2S] ERROR: Failed installing I2S driver: %d\n", err);
        return err;
    }

    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[I2S] ERROR: Failed setting I2S pins: %d\n", err);
        return err;
    }

    // Set clock configuration explicitly
    err = i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
    if (err != ESP_OK) {
        Serial.printf("[I2S] ERROR: Failed setting I2S clock: %d\n", err);
        return err;
    }

    Serial.println("[I2S] Driver successfully installed and configured (16kHz, 32-bit slot, Left channel).");
    return ESP_OK;
}

// ============================================================================
// WEBSOCKET EVENT CALLBACK
// ============================================================================
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            isWebSocketConnected = false;
            Serial.println("[WS] Disconnected from server.");
            break;
        case WStype_CONNECTED:
            isWebSocketConnected = true;
            Serial.printf("[WS] Connected to: %s:%d%s\n", SERVER_HOST, SERVER_PORT, SERVER_PATH);
            break;
        case WStype_TEXT:
            Serial.printf("[WS] Server message: %s\n", payload);
            break;
        case WStype_BIN:
            // Server binary response if any
            break;
        case WStype_ERROR:
            Serial.printf("[WS] Connection Error: %s\n", payload);
            break;
        case WStype_PING:
        case WStype_PONG:
            break;
    }
}

// ============================================================================
// WIFI SETUP
// ============================================================================
void setupWiFi() {
    Serial.printf("[WiFi] Connecting to SSID: %s\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("[WiFi] Connected! ESP32 IP address: ");
        Serial.println(WiFi.localIP());
        Serial.printf("[WiFi] RSSI: %d dBm\n", WiFi.RSSI());
    } else {
        Serial.println("[WiFi] ERROR: Connection failed! Verify credentials and 2.4GHz network.");
    }
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n=======================================================");
    Serial.println("  ESP32 Cough Prescreening - Audio Streamer (INMP441)");
    Serial.println("=======================================================");

#if USE_SERIAL_LOOPBACK
    Serial.println("[MODE] *** SERIAL LOOPBACK TEST MODE ACTIVE ***");
    Serial.println("[MODE] WiFi and WebSocket are disabled.");
    Serial.println("[MODE] Open Arduino 'Serial Plotter' (115200 baud) to view real-time waveform.");
    Serial.println("-------------------------------------------------------");
#else
    Serial.println("[MODE] *** LIVE WEBSOCKET STREAMING MODE ACTIVE ***");
    Serial.printf("[TARGET] ws://%s:%d%s\n", SERVER_HOST, SERVER_PORT, SERVER_PATH);
    Serial.println("-------------------------------------------------------");
#endif

    // Initialize I2S
    if (setupI2S() != ESP_OK) {
        Serial.println("[FATAL] Halting due to I2S init failure.");
        while (1) { delay(1000); }
    }

#if !USE_SERIAL_LOOPBACK
    // Connect to WiFi
    setupWiFi();

    // Setup WebSocket client
    webSocket.begin(SERVER_HOST, SERVER_PORT, SERVER_PATH);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(2000); // Reconnect every 2 seconds if disconnected
    webSocket.enableHeartbeat(15000, 3000, 2); // 15s ping, 3s timeout, 2 retries
#endif

    Serial.println("[SYSTEM] Setup completed successfully. Capturing audio...");
}

// ============================================================================
// MAIN LOOP
// ============================================================================
void loop() {
#if !USE_SERIAL_LOOPBACK
    // Maintain WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WiFi] Connection lost. Reconnecting...");
        WiFi.reconnect();
        delay(1000);
        return;
    }

    // Pump WebSocket events
    webSocket.loop();
#endif

    // Read 32-bit raw samples from INMP441 over I2S
    size_t bytes_read = 0;
    esp_err_t result = i2s_read(
        I2S_PORT,
        (void*)raw_i2s_buffer,
        sizeof(raw_i2s_buffer),
        &bytes_read,
        portMAX_DELAY
    );

    if (result != ESP_OK || bytes_read == 0) {
        return;
    }

    size_t samples_read = bytes_read / sizeof(int32_t);
    int64_t sum_squares = 0;
    int16_t peak_val = 0;

    // Process 32-bit I2S data -> 16-bit PCM with clipping guard
    for (size_t i = 0; i < samples_read; i++) {
        int32_t sample = raw_i2s_buffer[i];
        
        // Arithmetic right-shift with sign preservation
        int32_t shifted = sample >> BIT_SHIFT;

        // Hard clipping guard to prevent 16-bit overflow distortion
        if (shifted > 32767) {
            shifted = 32767;
        } else if (shifted < -32768) {
            shifted = -32768;
        }

        int16_t pcm16 = (int16_t)shifted;
        pcm_buffer[i] = pcm16;

        // Stats calculation
        sum_squares += ((int64_t)pcm16 * (int64_t)pcm16);
        int16_t abs_val = (pcm16 < 0) ? -pcm16 : pcm16;
        if (abs_val > peak_val) {
            peak_val = abs_val;
        }
    }

    double rms = sqrt((double)sum_squares / (double)samples_read);

#if USE_SERIAL_LOOPBACK
    // Serial Plotter Output:
    // Subsample to avoid saturating 115200 baud serial UART buffer
    // Prints clean waveform samples for Arduino Serial Plotter
    for (size_t i = 0; i < samples_read; i += 8) {
        Serial.println(pcm_buffer[i]);
    }

    // Periodic numerical summary to Serial Monitor every 500 ms
    if (millis() - lastRmsPrintTime >= 500) {
        lastRmsPrintTime = millis();
        // Serial monitor diagnostics format
        // Serial.printf(">> RMS: %5.1f | Peak: %5d | Samples: %d\n", rms, peak_val, samples_read);
    }

#else
    // STREAMING MODE: Send binary 16-bit PCM chunk over WebSocket
    if (isWebSocketConnected) {
        size_t pcm_bytes = samples_read * sizeof(int16_t);
        bool sent = webSocket.sendBIN((uint8_t*)pcm_buffer, pcm_bytes);

        if (sent) {
            totalBytesSent += pcm_bytes;
            totalFramesSent++;
        }
    }

    // Telemetry log every 5 seconds
    if (millis() - lastLogTime >= 5000) {
        lastLogTime = millis();
        float streamRateKbps = (totalBytesSent * 8.0f) / 5000.0f; // kbps over last 5s
        Serial.printf("[STREAM] WS: %s | Sent: %lu frames (%lu KB) | Rate: %.1f kbps | RMS: %.1f | Peak: %d | RSSI: %d dBm\n",
            isWebSocketConnected ? "ONLINE" : "OFFLINE (reconnecting)",
            totalFramesSent,
            totalBytesSent / 1024,
            streamRateKbps,
            rms,
            peak_val,
            WiFi.RSSI()
        );
        totalBytesSent = 0;
    }
#endif
}

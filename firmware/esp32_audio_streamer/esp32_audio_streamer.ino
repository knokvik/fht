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
 * Onboard LED:
 *   GPIO 2 (Blue LED on standard ESP32 DevKit)
 *   - Blinks fast (~70ms) during active cough recording session
 *   - OFF when idle
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include "driver/i2s.h"

// ============================================================================
// CONFIGURATION FLAGS & NETWORK SETTINGS
// ============================================================================

// Set to 1: Reads I2S and prints to Serial (works with tools/serial_bridge.py).
// Set to 0: Streams directly over WiFi WebSocket to PC.
#define USE_SERIAL_LOOPBACK 1

// Onboard Blue LED pin
#define LED_PIN 2

// WiFi Credentials (same hotspot connected to your laptop and ESP32)
const char* WIFI_SSID     = "regmig";
const char* WIFI_PASSWORD = "YOUR_HOTSPOT_PASSWORD";

// PC Server Settings
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
#define SAMPLE_RATE       16000     // 16 kHz standard
#define CHUNK_SAMPLES     1024      // Number of 16-bit samples per WebSocket frame
#define BIT_SHIFT         14        // INMP441 24-bit output in 32-bit slot: shift right by 14

// ============================================================================
// GLOBAL BUFFERS & STATE
// ============================================================================
WebSocketsClient webSocket;
bool isWebSocketConnected = false;

// 32-bit raw DMA buffer from I2S and 16-bit processed PCM output buffer
int32_t raw_i2s_buffer[CHUNK_SAMPLES];
int16_t pcm_buffer[CHUNK_SAMPLES];

// Recording state & LED fast blink control
bool isRecording = false;
unsigned long recordingEndTime = 0;
unsigned long lastBlinkTime = 0;
bool ledState = false;

// Telemetry counters
unsigned long totalBytesSent = 0;
unsigned long totalFramesSent = 0;
unsigned long lastLogTime = 0;

// ============================================================================
// LED BLINK HANDLER
// ============================================================================
void startRecordingLed(unsigned long durationMs) {
    isRecording = true;
    recordingEndTime = millis() + durationMs;
    lastBlinkTime = millis();
    ledState = true;
    digitalWrite(LED_PIN, HIGH);
}

void stopRecordingLed() {
    isRecording = false;
    digitalWrite(LED_PIN, LOW);
}

void updateRecordingLed() {
    unsigned long now = millis();
    if (isRecording) {
        if (now >= recordingEndTime) {
            stopRecordingLed();
        } else {
            // Fast blink: toggle every 70ms
            if (now - lastBlinkTime >= 70) {
                lastBlinkTime = now;
                ledState = !ledState;
                digitalWrite(LED_PIN, ledState ? HIGH : LOW);
            }
        }
    }
}

// ============================================================================
// SERIAL COMMAND PARSER
// ============================================================================
void checkSerialCommands() {
    while (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.startsWith("REC:START")) {
            unsigned long dur = 3000;
            int colonIdx = cmd.lastIndexOf(':');
            if (colonIdx > 8) {
                dur = cmd.substring(colonIdx + 1).toInt();
                if (dur <= 0) dur = 3000;
            }
            startRecordingLed(dur);
        } else if (cmd == "REC:STOP") {
            stopRecordingLed();
        }
    }
}

// ============================================================================
// I2S DRIVER INITIALIZATION
// ============================================================================
esp_err_t setupI2S() {
    Serial.println("[I2S] Initializing I2S peripheral...");

    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
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
    if (err != ESP_OK) return err;

    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) return err;

    err = i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
    if (err != ESP_OK) return err;

    return ESP_OK;
}

// ============================================================================
// WEBSOCKET EVENT CALLBACK
// ============================================================================
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            isWebSocketConnected = false;
            stopRecordingLed();
            break;
        case WStype_CONNECTED:
            isWebSocketConnected = true;
            break;
        case WStype_TEXT: {
            String text = String((char*)payload);
            if (text.indexOf("start_recording") >= 0) {
                startRecordingLed(3000);
            } else if (text.indexOf("stop_recording") >= 0) {
                stopRecordingLed();
            }
            break;
        }
        default:
            break;
    }
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
    Serial.begin(115200);
    delay(500);

    // Initialize onboard LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Initial 3 brief blinks to confirm startup
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(80);
        digitalWrite(LED_PIN, LOW);
        delay(80);
    }

    if (setupI2S() != ESP_OK) {
        while (1) {
            digitalWrite(LED_PIN, HIGH);
            delay(500);
            digitalWrite(LED_PIN, LOW);
            delay(500);
        }
    }

#if !USE_SERIAL_LOOPBACK
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 25) {
        delay(400);
        attempts++;
    }
    webSocket.begin(SERVER_HOST, SERVER_PORT, SERVER_PATH);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(2000);
#endif
}

// ============================================================================
// MAIN LOOP
// ============================================================================
void loop() {
    // Check for serial recording triggers
    checkSerialCommands();

    // Handle fast LED blinking
    updateRecordingLed();

#if !USE_SERIAL_LOOPBACK
    if (WiFi.status() == WL_CONNECTED) {
        webSocket.loop();
    }
#endif

    // Read 32-bit samples from I2S
    size_t bytes_read = 0;
    esp_err_t result = i2s_read(
        I2S_PORT,
        (void*)raw_i2s_buffer,
        sizeof(raw_i2s_buffer),
        &bytes_read,
        portMAX_DELAY
    );

    if (result != ESP_OK || bytes_read == 0) return;

    size_t samples_read = bytes_read / sizeof(int32_t);

    for (size_t i = 0; i < samples_read; i++) {
        int32_t sample = raw_i2s_buffer[i];
        int32_t shifted = sample >> BIT_SHIFT;

        if (shifted > 32767) shifted = 32767;
        else if (shifted < -32768) shifted = -32768;

        pcm_buffer[i] = (int16_t)shifted;
    }

#if USE_SERIAL_LOOPBACK
    // Print subsampled values to Serial for serial_bridge.py
    for (size_t i = 0; i < samples_read; i += 8) {
        Serial.println(pcm_buffer[i]);
    }
#else
    if (isWebSocketConnected) {
        webSocket.sendBIN((uint8_t*)pcm_buffer, samples_read * sizeof(int16_t));
    }
#endif
}

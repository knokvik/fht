// Diagnostic sketch for ESP32 INMP441 microphone
// Blinks the built‑in LED (GPIO 2) when microphone data is received (non‑zero samples)
// Also prints occasional sample statistics to Serial for debugging.

#include "driver/i2s.h"

// I2S configuration constants – match wiring from the hand‑off doc
#define I2S_SAMPLE_RATE   16000
#define I2S_BITS_PER_SAMPLE  I2S_BITS_PER_SAMPLE_32BIT // 32‑bit frames, we shift later
#define I2S_CHANNEL_NUM   2   // stereo slot, we use left channel only
#define I2S_BUZZER_PIN    -1  // not used

// Pin connections for INMP441
#define I2S_PIN_BCK   14   // Bit Clock (SCK)
#define I2S_PIN_WS    15   // Word Select (WS)
#define I2S_PIN_DATA  32   // Data (SD)

// LED pin – most ESP32 dev boards have a built‑in LED on GPIO2
#define LED_PIN 2

// Number of samples to read per loop iteration
#define READ_LEN 1024

// Shift needed to convert 32‑bit raw I2S value to 16‑bit PCM
#define BIT_SHIFT 14

void setupI2S() {
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // we only need left channel
    .communication_format = I2S_COMM_FORMAT_I2S_MSB,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_PIN_BCK,
    .ws_io_num = I2S_PIN_WS,
    .data_out_num = I2S_PIN_BCK, // not used for RX
    .data_in_num = I2S_PIN_DATA
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.println("=== Diagnostic Mic Sketch ===");
  setupI2S();
}

void loop() {
  int32_t i2s_buffer[READ_LEN];
  size_t bytes_read = 0;
  // Read raw 32‑bit I2S data
  esp_err_t err = i2s_read(I2S_NUM_0, (void *)i2s_buffer, sizeof(i2s_buffer), &bytes_read, 1000 / portTICK_RATE_MS);
  if (err != ESP_OK) {
    Serial.printf("I2S read error: %d\n", err);
    delay(200);
    return;
  }
  // bytes_read contains number of bytes actually read
  size_t samples = bytes_read / sizeof(int32_t);
  bool gotNonZero = false;
  int32_t minVal = INT32_MAX;
  int32_t maxVal = INT32_MIN;
  for (size_t i = 0; i < samples; ++i) {
    int32_t raw = i2s_buffer[i];
    // Shift to 16‑bit PCM
    int16_t pcm = (int16_t)(raw >> BIT_SHIFT);
    if (pcm != 0) gotNonZero = true;
    if (pcm < minVal) minVal = pcm;
    if (pcm > maxVal) maxVal = pcm;
  }
  // Blink LED if we saw any non‑zero sample
  if (gotNonZero) {
    digitalWrite(LED_PIN, HIGH);
    // Keep it on for a short time then off – creates a visible blink
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  } else {
    // No audio – keep LED off
    digitalWrite(LED_PIN, LOW);
  }
  // Print occasional stats (once per second approx)
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 1000) {
    Serial.printf("Samples: %d, non‑zero: %s, min: %d, max: %d\n", (int)samples, gotNonZero ? "yes" : "no", minVal, maxVal);
    lastPrint = millis();
  }
}

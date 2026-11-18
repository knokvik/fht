// Diagnostic sketch for ESP32 INMP441 microphone
// Configured with the user's EXACT wiring:
//   SCK (BCK) -> GPIO 4  (D4)
//   WS        -> GPIO 19 (D19)
//   SD (DATA) -> GPIO 23 (D23)
//   L/R       -> GND
//   GND       -> GND
//   VDD       -> 3.3V
//
// Blinks the built-in blue LED (GPIO 2) continuously when mic data is received.

#include "driver/i2s.h"

#define I2S_SAMPLE_RATE      16000
#define I2S_PIN_BCK          4    // D4 (SCK / Bit Clock)
#define I2S_PIN_WS           19   // D19 (WS / Word Select)
#define I2S_PIN_DATA         23   // D23 (SD / Serial Data)

#define LED_PIN              2    // Built-in LED on GPIO 2
#define SAMPLES_PER_READ     256  // Stereo pairs (512 total 32-bit words)

void setupI2S() {
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT, // Read both channels
    .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_STAND_I2S),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 512,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_PIN_BCK,
    .ws_io_num = I2S_PIN_WS,
    .data_out_num = -1,
    .data_in_num = I2S_PIN_DATA
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
  i2s_zero_dma_buffer(I2S_NUM_0);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Power-on flash sequence: 3 fast blinks to verify LED works
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }

  Serial.println("\n=== INMP441 Diagnostic with User Pins ===");
  Serial.println("SCK -> D4, WS -> D19, SD -> D23, L/R -> GND, VDD -> 3.3V");
  setupI2S();
}

void loop() {
  int32_t buffer[SAMPLES_PER_READ * 2]; // interleaved [L, R, L, R, ...]
  size_t bytes_read = 0;

  esp_err_t err = i2s_read(I2S_NUM_0, (void *)buffer, sizeof(buffer), &bytes_read, portMAX_DELAY);
  if (err != ESP_OK) {
    Serial.printf("I2S read error: %d\n", err);
    delay(200);
    return;
  }

  int total_words = bytes_read / sizeof(int32_t);
  int non_zero_count = 0;
  int32_t min_sample = 0;
  int32_t max_sample = 0;
  int32_t raw_preview[4] = {0, 0, 0, 0};

  for (int i = 0; i < total_words; i++) {
    int32_t raw = buffer[i];
    // INMP441 outputs 24-bit audio in MSB of 32-bit slot
    int16_t pcm = (int16_t)(raw >> 14);

    if (i < 4) {
      raw_preview[i] = raw;
    }

    if (raw != 0) {
      non_zero_count++;
      if (pcm < min_sample) min_sample = pcm;
      if (pcm > max_sample) max_sample = pcm;
    }
  }

  // If microphone is active (we get non-zero values), blink the LED continuously!
  if (non_zero_count > 0) {
    digitalWrite(LED_PIN, HIGH);
    delay(50);
    digitalWrite(LED_PIN, LOW);
  } else {
    digitalWrite(LED_PIN, LOW);
  }

  static unsigned long last_report = 0;
  if (millis() - last_report > 1000) {
    last_report = millis();
    if (non_zero_count > 0) {
      Serial.printf("[MIC WORKING!] Words: %d, Non-zero: %d, Min: %d, Max: %d\n",
                    total_words, non_zero_count, min_sample, max_sample);
    } else {
      Serial.printf("[MIC SILENT] All %d words read were 0! Raw[0..3]: 0x%08X 0x%08X 0x%08X 0x%08X\n",
                    total_words, raw_preview[0], raw_preview[1], raw_preview[2], raw_preview[3]);
    }
  }
}

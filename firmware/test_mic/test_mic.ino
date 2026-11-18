#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_SCK_PIN   14
#define I2S_WS_PIN    15
#define I2S_SD_PIN    32
#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000

#define SAMPLES_NUM   256
int32_t samples[SAMPLES_NUM * 2]; // stereo buffer

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== INMP441 DIAGNOSTIC TEST ===");

    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT, // Stereo to capture both Left and Right
        .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 256,
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
    Serial.printf("i2s_driver_install: %d\n", err);

    err = i2s_set_pin(I2S_PORT, &pin_config);
    Serial.printf("i2s_set_pin: %d\n", err);
}

void loop() {
    size_t bytes_read = 0;
    esp_err_t res = i2s_read(I2S_PORT, (void*)samples, sizeof(samples), &bytes_read, portMAX_DELAY);

    if (res != ESP_OK || bytes_read == 0) {
        Serial.printf("Read error: %d, bytes: %d\n", res, bytes_read);
        delay(500);
        return;
    }

    int total_words = bytes_read / sizeof(int32_t);
    int32_t min_l = 2147483647, max_l = -2147483648;
    int32_t min_r = 2147483647, max_r = -2147483648;
    int non_zero_l = 0, non_zero_r = 0;

    for (int i = 0; i < total_words; i += 2) {
        int32_t left = samples[i];
        int32_t right = samples[i + 1];

        if (left != 0) non_zero_l++;
        if (right != 0) non_zero_r++;

        if (left < min_l) min_l = left;
        if (left > max_l) max_l = left;

        if (right < min_r) min_r = right;
        if (right > max_r) max_r = right;
    }

    Serial.printf("Words: %d | Left[non0: %d, min: %d, max: %d, raw0: %d, >>14: %d] | Right[non0: %d, min: %d, max: %d]\n",
        total_words,
        non_zero_l, min_l, max_l, samples[0], (samples[0] >> 14),
        non_zero_r, min_r, max_r
    );

    delay(200);
}

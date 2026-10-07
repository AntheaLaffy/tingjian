/* Validation-only adapter for https://github.com/antirez/qwen-asr (MIT).
 * This is never linked into the Rust application. */
#include "qwen_asr.h"
#include "qwen_asr_audio.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: reference_dump MODEL WAV MEL_F32 ENCODER_F32\n"); return 2; }
    qwen_ctx_t *ctx = qwen_load(argv[1]);
    if (!ctx) return 1;
    int samples_count, frames, tokens;
    float *samples = qwen_load_wav(argv[2], &samples_count);
    if (!samples) return 1;
    float *mel = qwen_mel_spectrogram(samples, samples_count, &frames);
    if (!mel) return 1;
    FILE *out = fopen(argv[3], "wb");
    if (!out) return 1;
    fwrite(mel, sizeof(float), 128 * frames, out); fclose(out);
    float *encoded = qwen_encoder_forward(ctx, mel, frames, &tokens);
    if (!encoded) return 1;
    out = fopen(argv[4], "wb");
    if (!out) return 1;
    fwrite(encoded, sizeof(float), tokens * ctx->config.dec_hidden, out); fclose(out);
    fprintf(stderr, "frames=%d tokens=%d dim=%d\n", frames, tokens, ctx->config.dec_hidden);
    free(encoded); free(mel); free(samples); qwen_free(ctx);
    return 0;
}

/**
 * ============================================================
 *  つながるフィルタ — ニューラルセルオートマン
 * ============================================================
 *
 *  ハードウェア:
 *    XIAO ESP32S3 Sense + MSP2807 (ILI9341)
 *
 * ============================================================
 */

#include "esp_camera.h"
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include "nca_weights.h"

// ─────────────────────────────────────────────
//  ピン定義
// ─────────────────────────────────────────────
#define TFT_CS    3
#define TFT_DC    4
#define TFT_RST   5
#define TFT_MOSI  9
#define TFT_SCLK  7
#define TFT_MISO  8

// ─────────────────────────────────────────────
//  定数
// ─────────────────────────────────────────────
#define CAM_W  320
#define CAM_H  240

#define NCA_W   80
#define NCA_H   60
#define NCA_CH  NCA_STATE_CH  // 4

#define DISP_W  320
#define DISP_H  240
#define SCALE   4

#define NCA_STEPS_PER_FRAME  8      // 1フレームあたりのNCA更新回数。多いほどパターンが遠くまで伝播する
#define NCA_STEP_SIZE        0.08f  // 1ステップあたりの状態変化量。大きいほど激しく変化するが飽和しやすい
#define NCA_STRIDE_SMALL     1      // ch0,ch1のSobelサンプリング間隔。1=隣1マス参照（小さい模様）
#define NCA_STRIDE_LARGE     5      // ch2,ch3のSobelサンプリング間隔。5=隣5マス参照（大きな構造）
#define CAM_INJECT_BASE      0.01f  // カメラ静止時の注入率。小さいほどNCAが自律的にパターンを維持できる
#define CAM_INJECT_MOTION    0.80f  // 動き検出時に追加される最大注入率。動いた場所のNCA状態をカメラ値で上書きする
#define MOTION_THRESHOLD     0.04f  // 前フレームとの輝度差がこの値を超えると「動きあり」と判定する（0〜1スケール）
#define STOCHASTIC_RATE      0.5f   // 各ステップで更新されるセルの割合。0.5=50%のセルのみ更新
#define NOISE_INJECT_RATE    0.006f // 毎フレーム全セルに加えるランダムなゆらぎの大きさ。パターンの停滞を防ぐ
#define SEED_INTERVAL_MIN    5000   // 幾何学シード（円）を注入する最短間隔ms
#define SEED_INTERVAL_MAX    10000  // 幾何学シード（円）を注入する最長間隔
#define TARGET_FPS           4      // 目標フレームレート。処理が追いつかない場合は実FPSが下がる
#define FRAME_INTERVAL_MS    (1000 / TARGET_FPS)  // 1フレームあたりの目標時間

// カメラピン (XIAO ESP32S3 Sense)
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ─────────────────────────────────────────────
//  Sobelフィルタ
// ─────────────────────────────────────────────
const float SOBEL_X[9] = {
  -1.0f/8, 0, 1.0f/8,
  -2.0f/8, 0, 2.0f/8,
  -1.0f/8, 0, 1.0f/8
};
const float SOBEL_Y[9] = {
  -1.0f/8, -2.0f/8, -1.0f/8,
   0,       0,       0,
   1.0f/8,  2.0f/8,  1.0f/8
};

// ─────────────────────────────────────────────
//  グローバル変数
// ─────────────────────────────────────────────
Adafruit_ILI9341 tft = Adafruit_ILI9341(TFT_CS, TFT_DC, TFT_RST);

float*    stateA       = NULL;
float*    stateB       = NULL;
uint8_t*  camFrame     = NULL;
uint8_t*  camFramePrev = NULL;
uint16_t* dispBuf      = NULL;

float perception[NCA_PERCEPT_DIM];
float hidden[NCA_HIDDEN_DIM];

unsigned long nextSeedTime = 0;

// ─────────────────────────────────────────────
//  ユーティリティ
// ─────────────────────────────────────────────
inline float& stateAt(float* s, int y, int x, int ch) {
  return s[(y * NCA_W + x) * NCA_CH + ch];
}

inline float stateAtClamped(float* s, int y, int x, int ch) {
  if (x < 0) x = 0; if (x >= NCA_W) x = NCA_W - 1;
  if (y < 0) y = 0; if (y >= NCA_H) y = NCA_H - 1;
  return s[(y * NCA_W + x) * NCA_CH + ch];
}

inline float clampf(float v, float lo, float hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

// ─────────────────────────────────────────────
//  カメラ初期化
// ─────────────────────────────────────────────
bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size   = FRAMESIZE_QVGA;
  config.fb_count     = 2;
  config.grab_mode    = CAMERA_GRAB_LATEST;
  config.jpeg_quality = 12;
  config.fb_location  = CAMERA_FB_IN_PSRAM;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

// ─────────────────────────────────────────────
//  カメラ取得 → 縮小
// ─────────────────────────────────────────────
bool captureCamera() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return false;
  memcpy(camFramePrev, camFrame, NCA_W * NCA_H);
  for (int py = 0; py < NCA_H; py++) {
    for (int px = 0; px < NCA_W; px++) {
      int sx = px * SCALE, sy = py * SCALE;
      uint32_t sum = 0;
      for (int dy = 0; dy < SCALE; dy++)
        for (int dx = 0; dx < SCALE; dx++)
          sum += fb->buf[(sy + dy) * CAM_W + (sx + dx)];
      camFrame[py * NCA_W + px] = sum / (SCALE * SCALE);
    }
  }
  esp_camera_fb_return(fb);
  return true;
}

// ─────────────────────────────────────────────
//  カメラ注入（モーション検出型）
// ─────────────────────────────────────────────
void injectCamera(float* state) {
  for (int y = 0; y < NCA_H; y++) {
    for (int x = 0; x < NCA_W; x++) {
      float lum = (float)camFrame[y * NCA_W + x] / 255.0f;

      float sx = 0, sy_v = 0;
      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
          int cy = (y+ky<0)?0:(y+ky>=NCA_H)?NCA_H-1:y+ky;
          int cx = (x+kx<0)?0:(x+kx>=NCA_W)?NCA_W-1:x+kx;
          float v = (float)camFrame[cy*NCA_W+cx] / 255.0f;
          sx   += v * SOBEL_X[(ky+1)*3+(kx+1)];
          sy_v += v * SOBEL_Y[(ky+1)*3+(kx+1)];
        }
      }

      float diff = fabsf((float)camFrame[y*NCA_W+x]
                       - (float)camFramePrev[y*NCA_W+x]) / 255.0f;
      float rate = CAM_INJECT_BASE;
      if (diff > MOTION_THRESHOLD) {
        float m = clampf((diff - MOTION_THRESHOLD) / 0.3f, 0.0f, 1.0f);
        rate += CAM_INJECT_MOTION * m;
      }

      stateAt(state,y,x,0) = stateAt(state,y,x,0)*(1-rate) + lum*rate;
      stateAt(state,y,x,1) = stateAt(state,y,x,1)*(1-rate)
                              + clampf(sx   +0.5f,0,1)*rate;
      stateAt(state,y,x,2) = stateAt(state,y,x,2)*(1-rate)
                              + clampf(sy_v +0.5f,0,1)*rate;
    }
  }
}

// ─────────────────────────────────────────────
//  [1] カメラ画像から円シードを注入
//
//  輝度重心 = 画像の「明るい部分の中心」を円の中心とする
//  半径     = 明るい画素の重心からの輝度加重平均距離
//  白/黒    = 画像全体の平均輝度が明るければ黒、暗ければ白
// ─────────────────────────────────────────────
void injectCameraBasedSeed(float* state) {

  // 輝度重心と加重平均距離を計算
  float totalLum = 0;
  float weightX = 0, weightY = 0;

  for (int y = 0; y < NCA_H; y++) {
    for (int x = 0; x < NCA_W; x++) {
      float v = (float)camFrame[y * NCA_W + x] / 255.0f;
      totalLum += v;
      weightX  += v * x;
      weightY  += v * y;
    }
  }

  // 重心座標
  float cx = (totalLum > 0.01f) ? (weightX / totalLum) : (NCA_W / 2.0f);
  float cy = (totalLum > 0.01f) ? (weightY / totalLum) : (NCA_H / 2.0f);

  // 重心からの輝度加重平均距離 → 半径
  float weightedDist = 0;
  for (int y = 0; y < NCA_H; y++) {
    for (int x = 0; x < NCA_W; x++) {
      float v = (float)camFrame[y * NCA_W + x] / 255.0f;
      float dx = x - cx, dy = y - cy;
      weightedDist += v * sqrtf(dx*dx + dy*dy);
    }
  }
  int radius = (totalLum > 0.01f)
               ? (int)clampf(weightedDist / totalLum, 4.0f, 25.0f)
               : 12;

  // 平均輝度が明るければ黒、暗ければ白
  float seedVal = (totalLum / (NCA_W * NCA_H) > 0.5f) ? 0.0f : 1.0f;

  // ch3 に塗りつぶし円を書き込む
  for (int y = 0; y < NCA_H; y++) {
    for (int x = 0; x < NCA_W; x++) {
      float ddx = x - cx, ddy = y - cy;
      if (ddx*ddx + ddy*ddy < (float)(radius*radius)) {
        stateAt(state, y, x, 3) = seedVal;
      }
    }
  }

  Serial.printf("[SEED] circle center=(%.0f,%.0f) r=%d val=%.0f\n",
                cx, cy, radius, seedVal);
}

// ─────────────────────────────────────────────
//  ノイズ注入
// ─────────────────────────────────────────────
void injectNoise(float* state, float rate) {
  for (int i = 0; i < NCA_W * NCA_H * NCA_CH; i++) {
    float noise = ((float)random(2001) / 1000.0f - 1.0f) * rate;
    state[i] = clampf(state[i] + noise, 0.0f, 1.0f);
  }
}

// ─────────────────────────────────────────────
//  ホメオスタシス
// ─────────────────────────────────────────────
void applyHomeostasis(float* state) {
  for (int ch = 0; ch < NCA_CH; ch++) {
    float sum = 0;
    for (int y = 0; y < NCA_H; y++)
      for (int x = 0; x < NCA_W; x++)
        sum += stateAt(state, y, x, ch);
    float mean = sum / (NCA_W * NCA_H);
    float correction = (0.5f - mean) * 0.05f;
    for (int y = 0; y < NCA_H; y++)
      for (int x = 0; x < NCA_W; x++) {
        float& v = stateAt(state, y, x, ch);
        v = clampf(v + correction, 0.0f, 1.0f);
      }
  }
}

// ─────────────────────────────────────────────
//  NCA 1ステップ（マルチスケール知覚）
// ─────────────────────────────────────────────
void ncaStep(float* src, float* dst) {
  float fc1_w[NCA_HIDDEN_DIM * NCA_PERCEPT_DIM];
  float fc1_b[NCA_HIDDEN_DIM];
  float fc2_w[NCA_UPDATE_DIM * NCA_HIDDEN_DIM];
  float fc2_b[NCA_UPDATE_DIM];

  memcpy_P(fc1_w, nca_fc1_weight, sizeof(fc1_w));
  memcpy_P(fc1_b, nca_fc1_bias,   sizeof(fc1_b));
  memcpy_P(fc2_w, nca_fc2_weight, sizeof(fc2_w));
  memcpy_P(fc2_b, nca_fc2_bias,   sizeof(fc2_b));

  for (int y = 0; y < NCA_H; y++) {
    for (int x = 0; x < NCA_W; x++) {

      bool doUpdate = (random(100) < (int)(STOCHASTIC_RATE * 100));
      if (!doUpdate) {
        for (int ch = 0; ch < NCA_CH; ch++)
          stateAt(dst, y, x, ch) = stateAt(src, y, x, ch);
        continue;
      }

      int pidx = 0;
      for (int ch = 0; ch < NCA_CH; ch++) {
        int stride = (ch < 2) ? NCA_STRIDE_SMALL : NCA_STRIDE_LARGE;
        perception[pidx++] = stateAtClamped(src, y, x, ch);
        float gx = 0, gy = 0;
        for (int ky = -1; ky <= 1; ky++)
          for (int kx = -1; kx <= 1; kx++) {
            float v = stateAtClamped(src, y+ky*stride, x+kx*stride, ch);
            gx += v * SOBEL_X[(ky+1)*3+(kx+1)];
            gy += v * SOBEL_Y[(ky+1)*3+(kx+1)];
          }
        perception[pidx++] = gx;
        perception[pidx++] = gy;
      }

      for (int h = 0; h < NCA_HIDDEN_DIM; h++) {
        float s = fc1_b[h];
        const float* w = &fc1_w[h * NCA_PERCEPT_DIM];
        for (int p = 0; p < NCA_PERCEPT_DIM; p++) s += w[p] * perception[p];
        hidden[h] = (s > 0) ? s : 0;
      }

      for (int ch = 0; ch < NCA_UPDATE_DIM; ch++) {
        float delta = fc2_b[ch];
        const float* w = &fc2_w[ch * NCA_HIDDEN_DIM];
        for (int h = 0; h < NCA_HIDDEN_DIM; h++) delta += w[h] * hidden[h];
        float newVal = stateAt(src, y, x, ch) + delta * NCA_STEP_SIZE;
        stateAt(dst, y, x, ch) = clampf(newVal, 0.0f, 1.0f);
      }
    }
  }
}

// ─────────────────────────────────────────────
//  レンダリング
// ─────────────────────────────────────────────
void renderToDisplay(float* state) {
  float chMin[3]={1,1,1}, chMax[3]={0,0,0};
  for (int y=0;y<NCA_H;y++) for (int x=0;x<NCA_W;x++) for (int ch=0;ch<3;ch++) {
    float v=stateAt(state,y,x,ch);
    if(v<chMin[ch])chMin[ch]=v; if(v>chMax[ch])chMax[ch]=v;
  }
  for (int ch=0;ch<3;ch++) {
    float range=chMax[ch]-chMin[ch];
    if(range<0.02f){float c=(chMax[ch]+chMin[ch])*0.5f;chMin[ch]=c-0.01f;chMax[ch]=c+0.01f;}
  }
  for (int dy=0;dy<DISP_H;dy++) {
    int sy=dy/SCALE; if(sy>=NCA_H)sy=NCA_H-1;
    for (int dx=0;dx<DISP_W;dx++) {
      int sx=dx/SCALE; if(sx>=NCA_W)sx=NCA_W-1;
      int sx1=(sx+1<NCA_W)?sx+1:sx, sy1=(sy+1<NCA_H)?sy+1:sy;
      float fx=(float)(dx%SCALE)/SCALE, fy=(float)(dy%SCALE)/SCALE;
      float r=0,g=0,b=0;
      for (int ch=0;ch<3;ch++) {
        float v00=stateAt(state,sy,sx,ch),  v10=stateAt(state,sy,sx1,ch);
        float v01=stateAt(state,sy1,sx,ch), v11=stateAt(state,sy1,sx1,ch);
        float val=v00*(1-fx)*(1-fy)+v10*fx*(1-fy)+v01*(1-fx)*fy+v11*fx*fy;
        val=clampf((val-chMin[ch])/(chMax[ch]-chMin[ch]),0,1);
        if(ch==0)r=val; else if(ch==1)g=val; else b=val;
      }
      uint8_t r8=(uint8_t)(r*255),g8=(uint8_t)(g*255),b8=(uint8_t)(b*255);
      uint16_t color=((r8>>3)<<11)|((g8>>2)<<5)|(b8>>3);
      dispBuf[dy*DISP_W+dx]=(color>>8)|(color<<8);
    }
  }
}

// ─────────────────────────────────────────────
//  ディスプレイ転送
// ─────────────────────────────────────────────
void pushToDisplay() {
  tft.startWrite();
  tft.setAddrWindow(0, 0, DISP_W, DISP_H);
  tft.writePixels(dispBuf, DISP_W * DISP_H);
  tft.endWrite();
}

// ─────────────────────────────────────────────
//  NCA状態初期化
// ─────────────────────────────────────────────
void initState(float* state) {
  for (int y=0;y<NCA_H;y++) for (int x=0;x<NCA_W;x++) for (int ch=0;ch<NCA_CH;ch++)
    stateAt(state,y,x,ch)=(float)random(100)/1000.0f;
}

// ─────────────────────────────────────────────
//  setup()
// ─────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== つながるフィルタ (NCA) ===");
  Serial.printf("TARGET_FPS=%d  FRAME_INTERVAL=%dms\n",
                TARGET_FPS, FRAME_INTERVAL_MS);

  SPI.begin(TFT_SCLK, TFT_MISO, TFT_MOSI, TFT_CS);

  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  int stateSize = NCA_W * NCA_H * NCA_CH * sizeof(float);
  stateA       = (float*)   ps_malloc(stateSize);
  stateB       = (float*)   ps_malloc(stateSize);
  camFrame     = (uint8_t*) ps_malloc(NCA_W * NCA_H);
  camFramePrev = (uint8_t*) ps_malloc(NCA_W * NCA_H);
  dispBuf      = (uint16_t*)ps_malloc(DISP_W * DISP_H * sizeof(uint16_t));

  if (!stateA || !stateB || !camFrame || !camFramePrev || !dispBuf) {
    Serial.println("PSRAM alloc failed!");
    while (1) delay(1000);
  }
  memset(camFramePrev, 0, NCA_W * NCA_H);
  Serial.printf("PSRAM OK: state=%dBx2 cam=%dBx2 disp=%dB\n",
                stateSize, NCA_W*NCA_H, DISP_W*DISP_H*2);

  tft.begin(40000000);
  tft.setRotation(1);
  tft.fillScreen(ILI9341_BLACK);
  tft.setTextColor(ILI9341_WHITE);
  tft.setTextSize(2);
  tft.setCursor(20, 100);
  tft.println("NCA Initializing...");
  Serial.println("TFT OK.");

  if (!initCamera()) {
    tft.fillScreen(ILI9341_RED);
    tft.setCursor(20, 110);
    tft.println("Camera FAILED!");
    while (1) delay(1000);
  }
  Serial.println("Camera OK.");

  initState(stateA);
  initState(stateB);
  for (int i = 0; i < 5; i++) { captureCamera(); delay(100); }
  injectCamera(stateA);

  nextSeedTime = millis() + SEED_INTERVAL_MIN;

  Serial.printf("NCA ready: stride small=%d large=%d steps=%d\n",
                NCA_STRIDE_SMALL, NCA_STRIDE_LARGE, NCA_STEPS_PER_FRAME);
  Serial.println("Starting main loop.");
}

// ─────────────────────────────────────────────
//  loop()
// ─────────────────────────────────────────────
void loop() {
  unsigned long t0 = millis();

  bool gotFrame = captureCamera();

  applyHomeostasis(stateA);
  injectNoise(stateA, NOISE_INJECT_RATE);

  if (millis() >= nextSeedTime) {
    injectCameraBasedSeed(stateA);
    nextSeedTime = millis()
                 + SEED_INTERVAL_MIN
                 + random(SEED_INTERVAL_MAX - SEED_INTERVAL_MIN);
  }

  float* src = stateA;
  float* dst = stateB;
  for (int step = 0; step < NCA_STEPS_PER_FRAME; step++) {
    ncaStep(src, dst);
    float* tmp = src; src = dst; dst = tmp;
  }
  if (src != stateA) {
    float* tmp = stateA; stateA = src; stateB = tmp;
  }

  unsigned long tNca = millis();

  if (gotFrame) injectCamera(stateA);

  renderToDisplay(stateA);
  unsigned long tRender = millis();
  pushToDisplay();

  unsigned long elapsed = millis() - t0;
  if (elapsed < FRAME_INTERVAL_MS) {
    delay(FRAME_INTERVAL_MS - elapsed);
  }
  unsigned long totalElapsed = millis() - t0;

  static int frameCount = 0;
  static float prevAvg = 0;
  if (frameCount % 10 == 0) {
    for (int ch=0;ch<3;ch++) {
      float cMin=999,cMax=-999,cSum=0;
      for (int y=0;y<NCA_H;y++) for (int x=0;x<NCA_W;x++) {
        float v=stateAt(stateA,y,x,ch);
        if(v<cMin)cMin=v; if(v>cMax)cMax=v; cSum+=v;
      }
      Serial.printf("  ch%d: [%.3f..%.3f avg=%.3f]\n",
                    ch,cMin,cMax,cSum/(NCA_W*NCA_H));
    }
    float sSum=0;
    for (int i=0;i<NCA_W*NCA_H*NCA_CH;i++) sSum+=stateA[i];
    float avg=sSum/(NCA_W*NCA_H*NCA_CH);
    float delta=fabsf(avg-prevAvg); prevAvg=avg;
    Serial.printf("F%d: nca=%lums rnd=%lums proc=%lums total=%lums (%.1ffps) avg=%.3f d=%.5f\n",
                  frameCount, tNca-t0, tRender-tNca,
                  elapsed, totalElapsed,
                  1000.0f/max(totalElapsed,1UL), avg, delta);
  }
  frameCount++;
}

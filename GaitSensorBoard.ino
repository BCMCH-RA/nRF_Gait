/*=============================================================================
  GaitSensorBoard.ino  -  Seeed XIAO nRF52840 Sense / Seeed XIAO nRF52840 Sense Plus
  Gait Assessment platform - BLE IMU streaming node

  One of these sketches is flashed on EACH of the 3 modules (Right Leg,
  Left Leg, Torso).  Edit the three BOARD IDENTITY lines at the top so every
  module has a unique name/position, then upload the SAME sketch to all three.

  Board skin   : Seeed Studio XIAO nRF52840 Sense (Tools > Board > Seeed
                 nRF52840 mbed-enabled Boards > XIAO nRF52840 Sense)
  Libraries    : ArduinoBLE (bundled with the board package).  The LSM6DS3TR-C
                 is configured and read directly over I2C - no IMU library is
                 required, so the sketch is immune to LSM6DS3 library clashes.

  Behaviour
  ----------------------------------------------------------------------------
  1. Advertises a custom BLE GATT service (see UUIDs below).
  2. Accepts a 64-bit unix epoch-milliseconds write (TimeSync char) from the
     Web Bluetooth central.  The value is captured against the local free
     running microsecond clock, so every module shares the same absolute
     time base (epoch UTC ms; Indian Standard Time = epoch + 5 h 30 m is
     applied on the host side).
  3. On the "start streaming" command it samples the on-board LSM6DS3TR-C
     six-axis IMU at 208 Hz ODR (the closest LSM6DS3 rate to the requested
     200 Hz) and batches 19 samples (~91 ms of data) into one BLE
     notification.  ~10.5 packets/s per module and 235-byte payloads keep
     the over-the-air traffic low -> fewer missed packets, longer battery.
  4. Packet format (all little-endian):
       [ 0..7 ]   uint64  base sample epoch-ms (UTC)
       [ 8..9 ]   uint16  packet sequence (wrap-around; used for loss check)
       [10   ]   uint8   sample count N (always 19)
       [11.. ]   N x 12 bytes : ax,ay,az,gx,gy,gz as int16 (LSM6DS3 raw)
     Physical units: accel ±4 g  => raw * 0.000122 g
                     gyro 2000 dps => raw * 0.070 dps
  5. "Stop streaming" parks the stream; "read battery" reports the LiPo %
     computed from the on-board battery voltage divider (P0.31 ADC).
=============================================================================*/

/* ------------------------- BOARD IDENTITY (edit per module) --------------- */
#define MODULE_POSITION  "Right Leg"            // "Right Leg" | "Left Leg" | "Torso"
#define DEVICE_NAME      "Gait-Right"           // unique, findable name
#define DEVICE_ID        0                       // 0 = Right, 1 = Left, 2 = Torso
#define FW_VERSION       "1.1.0"

/* ------------------------- GATT UUIDs (shared with index.html) ----------- */
#define UUID_SERVICE  "d5f3b000-0001-4000-8000-00000000c001"
#define UUID_TIME     "d5f3b000-0001-4000-8000-00000000c002"
#define UUID_CMD      "d5f3b000-0001-4000-8000-00000000c003"
#define UUID_DATA     "d5f3b000-0001-4000-8000-00000000c004"
#define UUID_BATT     "d5f3b000-0001-4000-8000-00000000c005"
#define UUID_INFO     "d5f3b000-0001-4000-8000-00000000c006"

/* ------------------------- Commands --------------------------------------- */
#define CMD_STOP            0x00
#define CMD_START           0x01
#define CMD_READ_BATTERY    0x02
#define CMD_STOP            0x00
#define CMD_START           0x01
#define CMD_READ_BATTERY    0x02
#define CMD_PROBE           0x03

/* ------------------------- Streaming / IMU config ------------------------- */
#define SAMPLING_HZ         208                  // LSM6DS3 ODR closest to 200 Hz
#define SAMPLE_PERIOD_US    4808                 // 1'000'000 / 208
#define SAMPLES_PER_PACKET  19                   // fits in 247-byte MTU: 11 + 19*12 = 239 B
#define PACKET_SIZE         (11 + SAMPLES_PER_PACKET * 12)
#define ACCEL_SCALE_G       0.000122f            // ±4 g range  -> g per LSB
#define GYRO_SCALE_DPS      0.070f               // 2000 dps    -> dps per LSB

/* ------------------------- I2C / IMU address ------------------------------ */
#define IMU_I2C_ADDR        0x6A                 // LSM6DS3TR-C on XIAO Sense
#define IMU_OUT_REG_GX      0x22                 // OUTX_L_G .. OUTZ_L_G then OUTX_L_XL..

/* ------------------------- Battery divider (XIAO nRF52840) ---------------- */
#define BAT_DIV_RTOP        1510.0f
#define BAT_DIV_RBOT        510.0f
#define BAT_ADC_FS_V        3.6f                 // SAADC full-scale @ divider node (V)
#define BAT_V_FULL          4.20f                // LiPo fully charged
#define BAT_V_EMPTY         3.30f                // LiPo cut-off
// NOTE: if battery % looks off, check BAT_ADC_FS_V against a multimeter reading
//       on the battery pads; the on-board divider factor is (1510+510)/510 = 2.96.

#if defined(DEBUG_SERIAL)
#define DBG(...)  Serial.printf(__VA_ARGS__)
#else
#define DBG(...)
#endif

#include <Arduino.h>
#include <Wire.h>
#include <ArduinoBLE.h>

/*===========================================================================
   GATT objects
===========================================================================*/
BLEService        service(UUID_SERVICE);
BLECharacteristic timeChar(UUID_TIME,  BLEWrite,                                 8);
BLECharacteristic cmdChar (UUID_CMD,   BLEWrite,                                 1);
BLECharacteristic dataChar(UUID_DATA,  BLENotify,                                PACKET_SIZE);
BLECharacteristic battChar(UUID_BATT,  BLERead | BLENotify,                      1);
BLECharacteristic infoChar(UUID_INFO,  BLERead,                                  48);

/*===========================================================================
   Globals
===========================================================================*/
static bool    imuOk      = false;
static bool    connected  = false;
static bool    streaming  = false;
static bool    synced     = false;
static volatile uint64_t syncEpochMs = 0;   // host epoch-ms at sync moment
static volatile uint64_t syncUs     = 0;    // gUs at sync moment

static uint64_t gUs     = 0;                // 64-bit microsecond clock
static uint32_t lastUs  = 0;

static uint8_t  pkt[PACKET_SIZE];
static char     infoStr[48];

/*===========================================================================
   Time helpers
===========================================================================*/
static uint64_t nowEpochMs() {
  if (!synced) return 0;
  return syncEpochMs + (gUs - syncUs) / 1000ULL;
}

static void putU64(uint8_t* p, uint64_t v) {
  for (uint8_t i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

/*===========================================================================
   IMU - single burst read of gyro+accel (12 bytes starting at 0x22)
   Returns 6 int16 in order: gx, gy, gz, ax, ay, az
===========================================================================*/
static bool readIMU(int16_t out[6]) {
  if (!imuOk) return false;
  Wire.beginTransmission(IMU_I2C_ADDR);
  Wire.write(IMU_OUT_REG_GX);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)IMU_I2C_ADDR, (uint8_t)12) != 12) return false;
  for (uint8_t i = 0; i < 6; i++) {
    uint16_t lo = Wire.read();
    uint16_t hi = Wire.read();
    out[i] = (int16_t)((hi << 8) | lo);
  }
  return true;
}

/*-------------------- LSM6DS3TR-C register map (direct I2C, no library) -----*/
#define REG_WHO_AM_I    0x0F
#define REG_CTRL1_XL    0x10
#define REG_CTRL2_G     0x11
#define REG_CTRL3_C     0x12
#define IMU_WHO_AM_I    0x6A
#define CTRL1_XL_208HZ_4G  0x54   // ODR=0101(208Hz) | FS=01(+-4g)
#define CTRL2_G_208HZ_2000  0x5C   // ODR=0101(208Hz) | FS=11(+-2000dps)
#define CTRL3_C_IFINC_BDU   0x44   // IF_INC=1, BDU=1

static bool imuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(IMU_I2C_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static void initIMU() {
#if defined(PIN_LSM6DS3TR_C_POWER)
  pinMode(PIN_LSM6DS3TR_C_POWER, OUTPUT);
  digitalWrite(PIN_LSM6DS3TR_C_POWER, HIGH);      // gate 6-D module power
  delay(10);
#endif
  Wire.begin();
  Wire.setClock(400000);

  Wire.beginTransmission(IMU_I2C_ADDR);
  Wire.write(REG_WHO_AM_I);
  if (Wire.endTransmission(false) != 0) { imuOk = false; return; }
  if (Wire.requestFrom((uint8_t)IMU_I2C_ADDR, (uint8_t)1) != 1) { imuOk = false; return; }
  uint8_t who = Wire.read();

  imuOk = (who == IMU_WHO_AM_I) &&
          imuWrite(REG_CTRL3_C,  CTRL3_C_IFINC_BDU) &&
          imuWrite(REG_CTRL1_XL, CTRL1_XL_208HZ_4G) &&
          imuWrite(REG_CTRL2_G,  CTRL2_G_208HZ_2000);
  DBG("IMU init: %s (who=0x%02X)\n", imuOk ? "OK" : "FAILED", who);
}

/*===========================================================================
   Battery (P0.31 AIN7 divider, enabled by P0.14 = LOW)
===========================================================================*/
static uint8_t readBatteryPct() {
#if defined(PIN_VBAT)
#if defined(PIN_VBAT_ENABLE)
  pinMode(PIN_VBAT_ENABLE, OUTPUT);
  digitalWrite(PIN_VBAT_ENABLE, LOW);             // keep reading path enabled
#endif
  uint32_t sum = 0;
  for (uint8_t i = 0; i < 8; i++) {
    sum += analogRead(PIN_VBAT);
    delayMicroseconds(200);
  }
  float raw   = (float)sum / 8.0f;
  float nodeV = raw * (BAT_ADC_FS_V / 4095.0f);   // 12-bit ADC
  float vbat  = nodeV * (BAT_DIV_RTOP + BAT_DIV_RBOT) / BAT_DIV_RBOT;
  int pct = (int)((vbat - BAT_V_EMPTY) / (BAT_V_FULL - BAT_V_EMPTY) * 100.0f);
  if (pct < 0) pct = 0;
  if (pct > 100) pct = 100;
  return (uint8_t)pct;
#else
  return 0xFF;                                     // no battery channel -> unknown
#endif
}

static void notifyBattery() {
  uint8_t pctV = readBatteryPct();
  if (battChar.writeValue(&pctV, 1) >= 0) DBG("batt notify sent\n");
}

/*===========================================================================
   PROBE: a guaranteed 23-byte single-sample notify, emitted IMMEDIATELY and
   directly (no batched full-MTU attempt, no cadence, no timing). This is the
   mtbr-decisive test: if the browser receives this, the notify path works and
   anything else that "produces no data" is a firmware batching/timing bug.
   Payload is one sample of zeros with a valid epoch base and seq → 11 + 12 = 23 B.
===========================================================================*/
static void emitProbeNotify() {
  static uint16_t probeSeq = 0;              // monotonically increasing per probe
  uint8_t one[23];
  putU64(one, nowEpochMs());                 // 8 B base
  one[8]  = (uint8_t)(probeSeq & 0xFF);      // 2 B seq
  one[9]  = (uint8_t)(probeSeq >> 8);
  one[10] = 1;                               // 1 sample
  memset(&one[11], 0, 12);                   // 12 B of zeros = the sample
  if (dataChar.writeValue(one, 23) >= 0) {
    probeSeq++;
    DBG("probe notify sent (23 B)\n");
  } else {
    DBG("probe notify writeValue FAILED\n");
  }
}

/*===========================================================================
   Status LED (active LOW)
     - Boot / init error : rapid blink (150 ms)
     - Advertising only   : slow blink (1 s period, 25% on)
     - Connected, idle    : fast blink (400 ms period, 25% on)
     - Streaming          : solid ON
===========================================================================*/
static void updateLED() {
  if (streaming && connected) {
    digitalWrite(LED_BUILTIN, LOW);               // solid ON while streaming
    return;
  }
  uint32_t period = connected ? 400 : 1000;
  uint32_t phase  = millis() % period;
  digitalWrite(LED_BUILTIN, phase < (period >> 2) ? LOW : HIGH);
}

/*===========================================================================
   BLE events
===========================================================================*/
static void onBLEConnected(BLEDevice central) {
  connected = true;
  DBG("central connected\n");
  notifyBattery();
}

static void onBLEDisconnected(BLEDevice central) {
  connected   = false;
  streaming   = false;
  DBG("central disconnected\n");
}

/*===========================================================================
   Write handlers (polled in loop)
===========================================================================*/
static void handleTimeSyncWrite() {
  const uint8_t* v = timeChar.value();
  uint64_t e = 0;
  for (uint8_t i = 0; i < 8; i++) e |= ((uint64_t)v[i]) << (8 * i);
  syncEpochMs = e;
  syncUs      = gUs;
  synced      = true;
  DBG("time synced epoch=%llu ms\n", (unsigned long long)e);
}

static void handleCmdWrite() {
  const uint8_t c = cmdChar.value()[0];
  switch (c) {
    case CMD_START:
      streaming = true;
      DBG("stream start\n");
      break;
    case CMD_STOP:
      streaming = false;
      DBG("stream stop\n");
      break;
    case CMD_READ_BATTERY:
      notifyBattery();
      DBG("battery request\n");
      break;
    case CMD_PROBE:
      emitProbeNotify();                 // guaranteed 23 B notify RIGHT NOW
      DBG("probe request\n");
      break;
  }
}

/*===========================================================================
   Streaming
===========================================================================*/
static void sampleAndStream() {
  static uint32_t lastSample = 0;
  static uint8_t  batchCount = 0;
  static uint16_t seq        = 0;

  uint32_t nowUs = micros();
  if ((uint32_t)(nowUs - lastSample) < SAMPLE_PERIOD_US) return;

  lastSample += SAMPLE_PERIOD_US;                 // fixed cadence, no drift
  if ((uint32_t)(nowUs - lastSample) > (uint32_t)SAMPLE_PERIOD_US)
    lastSample = nowUs;                           // recovered from a long stall

  int16_t m[6];                                     // m = gx,gy,gz,ax,ay,az
  if (!readIMU(m)) return;

  if (batchCount == 0) putU64(pkt, nowEpochMs());

  uint8_t* s = &pkt[11 + batchCount * 12];
  static const uint8_t order[6] = {3,4,5,0,1,2};    // reorder to ax,ay,az,gx,gy,gz
  for (uint8_t i = 0; i < 6; i++) {
    s[i * 2]     = (uint8_t)(m[order[i]] & 0xFF);
    s[i * 2 + 1] = (uint8_t)(m[order[i]] >> 8);
  }
  batchCount++;

  if (batchCount == SAMPLES_PER_PACKET) {
    pkt[8]  = (uint8_t)(seq & 0xFF);
    pkt[9]  = (uint8_t)(seq >> 8);
    pkt[10] = SAMPLES_PER_PACKET;
    if (dataChar.writeValue(pkt, PACKET_SIZE) >= 0) {   // full 239 B OK
      seq++; batchCount = 0; return;
    }
    // MTU too small for a 239 B notify -> drain as 1-sample packets
    DBG("MTU small, falling back to 23 B packets\n");
    for (uint8_t k = 0; k < SAMPLES_PER_PACKET; k++) {
      uint64_t base = nowEpochMs() - (uint64_t)(SAMPLES_PER_PACKET - 1 - k) * (SAMPLE_PERIOD_US / 1000);
      putU64(pkt, base);                             // per-sample base, monotic
      uint8_t* one = &pkt[11 + 12 * k];
      pkt[8] = (uint8_t)(seq & 0xFF);
      pkt[9] = (uint8_t)(seq >> 8);
      pkt[10] = 1;
      memmove(&pkt[11], one, 12);                    // first 12 B = this sample
      if (dataChar.writeValue(pkt, 23) < 0) break;   // give up this cycle
      seq++;
    }
    batchCount = 0;
  }
}

/*===========================================================================
   Setup / Loop
===========================================================================*/
void setup() {
#if defined(DEBUG_SERIAL)
  Serial.begin(115200);
  while (!Serial) { }
#endif

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);                // LED off (active LOW)

  analogReadResolution(12);

  initIMU();

  snprintf(infoStr, sizeof(infoStr), "GaitSensor|%s|%d|%s|%d",
           MODULE_POSITION, SAMPLING_HZ, FW_VERSION, DEVICE_ID);

  if (!BLE.begin()) {
    DBG("BLE init failed\n");
    while (1) {                                  // rapid blink = init error
      digitalWrite(LED_BUILTIN, (millis() / 150) % 2 ? HIGH : LOW);
      delay(25);
    }
  }

  BLE.setLocalName(DEVICE_NAME);
  BLE.setDeviceName(DEVICE_NAME);
  BLE.setAdvertisedService(service);
  BLE.setEventHandler(BLEConnected, onBLEConnected);
  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);

  service.addCharacteristic(timeChar);
  service.addCharacteristic(cmdChar);
  service.addCharacteristic(dataChar);
  service.addCharacteristic(battChar);
  service.addCharacteristic(infoChar);
  BLE.addService(service);

  infoChar.writeValue((const uint8_t*)infoStr, strlen(infoStr));
  battChar.writeValue((const uint8_t*)"\xFF", 1);

  BLE.advertise();
}

void loop() {
  uint32_t nowUs = micros();
  gUs += (uint64_t)(uint32_t)(nowUs - lastUs);
  lastUs = nowUs;

  BLE.poll();

  if (timeChar.written()) handleTimeSyncWrite();
  if (cmdChar.written())  handleCmdWrite();

  if (streaming && connected) {
    sampleAndStream();
  }

  updateLED();

  static uint32_t lastBatt = 0;
  if (connected && (uint32_t)(millis() - lastBatt) >= 30000UL) {
    lastBatt = millis();
    notifyBattery();
  }
}
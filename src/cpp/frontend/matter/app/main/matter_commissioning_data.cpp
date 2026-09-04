/*
 * ESPectre - Matter Commissioning Data Provider
 *
 * Loads, validates, and generates per-device Matter commissioning data.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "matter_commissioning_data.h"

#include <bootloader_random.h>
#include <esp_partition.h>
#include <setup_payload/SetupPayload.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace {

constexpr char kPartitionLabel[] = "matter_factory";
constexpr uint8_t kRecordMagic[] = {'E', 'S', 'P', 'M', 'T', 'R', '0', '1'};
constexpr uint16_t kRecordVersion = 1;
constexpr size_t kSaltLength = 16;

struct __attribute__((packed)) MatterFactoryRecord {
  uint8_t magic[sizeof(kRecordMagic)];
  uint16_t version;
  uint16_t size;
  uint32_t setup_passcode;
  uint16_t setup_discriminator;
  uint16_t reserved;
  uint32_t iteration_count;
  uint8_t salt[kSaltLength];
  uint32_t crc32;
};

uint32_t crc32(const uint8_t *data, size_t length) {
  uint32_t crc = 0xffffffffU;
  for (size_t index = 0; index < length; ++index) {
    crc ^= data[index];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
  }
  return ~crc;
}

bool record_is_valid(const MatterFactoryRecord &record) {
  return std::memcmp(record.magic, kRecordMagic, sizeof(kRecordMagic)) == 0 &&
         record.version == kRecordVersion && record.size == sizeof(record) &&
         chip::SetupPayload::IsValidSetupPIN(record.setup_passcode) &&
         record.setup_discriminator <= chip::kMaxDiscriminatorValue &&
         record.iteration_count >= chip::Crypto::kSpake2p_Min_PBKDF_Iterations &&
         record.iteration_count <= chip::Crypto::kSpake2p_Max_PBKDF_Iterations &&
         record.crc32 == crc32(reinterpret_cast<const uint8_t *>(&record),
                               offsetof(MatterFactoryRecord, crc32));
}

const esp_partition_t *find_factory_partition() {
  return esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY,
                                  kPartitionLabel);
}

}  // namespace

namespace espectre {

CHIP_ERROR MatterCommissioningDataProvider::initialize() {
  const esp_partition_t *partition = find_factory_partition();
  VerifyOrReturnError(partition != nullptr, CHIP_ERROR_NOT_FOUND);

  MatterFactoryRecord record{};
  VerifyOrReturnError(esp_partition_read(partition, 0, &record, sizeof(record)) == ESP_OK,
                      CHIP_ERROR_READ_FAILED);

  if (!record_is_valid(record)) {
    return generate_and_store();
  }

  setup_passcode_ = record.setup_passcode;
  setup_discriminator_ = record.setup_discriminator;
  iteration_count_ = record.iteration_count;
  std::copy(std::begin(record.salt), std::end(record.salt), salt_.begin());
  salt_length_ = sizeof(record.salt);
  generated_on_boot_ = false;
  return CHIP_NO_ERROR;
}

CHIP_ERROR MatterCommissioningDataProvider::generate_and_store() {
  const esp_partition_t *partition = find_factory_partition();
  VerifyOrReturnError(partition != nullptr, CHIP_ERROR_NOT_FOUND);

  MatterFactoryRecord record{};
  std::memcpy(record.magic, kRecordMagic, sizeof(kRecordMagic));
  record.version = kRecordVersion;
  record.size = sizeof(record);
  record.iteration_count = iteration_count_;

  // RF is not initialized yet, so temporarily enable the ADC entropy source
  // before reading the hardware RNG during early application startup.
  bootloader_random_enable();
  do {
    bootloader_fill_random(&record.setup_passcode, sizeof(record.setup_passcode));
    record.setup_passcode = (record.setup_passcode % chip::kMaxSetupPasscode) + 1;
  } while (!chip::SetupPayload::IsValidSetupPIN(record.setup_passcode));

  bootloader_fill_random(&record.setup_discriminator, sizeof(record.setup_discriminator));
  record.setup_discriminator &= chip::kMaxDiscriminatorValue;
  bootloader_fill_random(record.salt, sizeof(record.salt));
  bootloader_random_disable();
  record.crc32 = crc32(reinterpret_cast<const uint8_t *>(&record),
                       offsetof(MatterFactoryRecord, crc32));

  VerifyOrReturnError(esp_partition_erase_range(partition, 0, partition->size) == ESP_OK,
                      CHIP_ERROR_WRITE_FAILED);
  VerifyOrReturnError(esp_partition_write(partition, 0, &record, sizeof(record)) == ESP_OK,
                      CHIP_ERROR_WRITE_FAILED);

  setup_passcode_ = record.setup_passcode;
  setup_discriminator_ = record.setup_discriminator;
  std::copy(std::begin(record.salt), std::end(record.salt), salt_.begin());
  salt_length_ = sizeof(record.salt);
  generated_on_boot_ = true;
  return CHIP_NO_ERROR;
}

CHIP_ERROR MatterCommissioningDataProvider::ensure_verifier() {
  if (verifier_ready_) {
    return CHIP_NO_ERROR;
  }

  chip::Crypto::Spake2pVerifier verifier;
  ReturnErrorOnFailure(verifier.Generate(iteration_count_,
                                         chip::ByteSpan(salt_.data(), salt_length_),
                                         setup_passcode_));
  chip::MutableByteSpan serialized(verifier_.data(), verifier_.size());
  ReturnErrorOnFailure(verifier.Serialize(serialized));
  VerifyOrReturnError(serialized.size() == verifier_.size(), CHIP_ERROR_INTERNAL);
  verifier_ready_ = true;
  return CHIP_NO_ERROR;
}

CHIP_ERROR MatterCommissioningDataProvider::GetSetupDiscriminator(uint16_t &setup_discriminator) {
  setup_discriminator = setup_discriminator_;
  return CHIP_NO_ERROR;
}

CHIP_ERROR MatterCommissioningDataProvider::SetSetupDiscriminator(uint16_t setup_discriminator) {
  (void) setup_discriminator;
  return CHIP_ERROR_NOT_IMPLEMENTED;
}

CHIP_ERROR MatterCommissioningDataProvider::GetSpake2pIterationCount(uint32_t &iteration_count) {
  iteration_count = iteration_count_;
  return CHIP_NO_ERROR;
}

CHIP_ERROR MatterCommissioningDataProvider::GetSpake2pSalt(chip::MutableByteSpan &salt_buffer) {
  VerifyOrReturnError(salt_buffer.size() >= salt_length_, CHIP_ERROR_BUFFER_TOO_SMALL);
  std::memcpy(salt_buffer.data(), salt_.data(), salt_length_);
  salt_buffer.reduce_size(salt_length_);
  return CHIP_NO_ERROR;
}

CHIP_ERROR MatterCommissioningDataProvider::GetSpake2pVerifier(
    chip::MutableByteSpan &verifier_buffer, size_t &verifier_length) {
  verifier_length = verifier_.size();
  VerifyOrReturnError(verifier_buffer.size() >= verifier_length, CHIP_ERROR_BUFFER_TOO_SMALL);
  ReturnErrorOnFailure(ensure_verifier());
  std::memcpy(verifier_buffer.data(), verifier_.data(), verifier_length);
  verifier_buffer.reduce_size(verifier_length);
  return CHIP_NO_ERROR;
}

CHIP_ERROR MatterCommissioningDataProvider::GetSetupPasscode(uint32_t &setup_passcode) {
  setup_passcode = setup_passcode_;
  return CHIP_NO_ERROR;
}

CHIP_ERROR MatterCommissioningDataProvider::SetSetupPasscode(uint32_t setup_passcode) {
  (void) setup_passcode;
  return CHIP_ERROR_NOT_IMPLEMENTED;
}

}  // namespace espectre

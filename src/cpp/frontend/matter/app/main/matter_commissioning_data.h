/*
 * ESPectre - Matter Commissioning Data Provider
 *
 * Loads, validates, and generates per-device Matter commissioning data.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <crypto/CHIPCryptoPAL.h>
#include <platform/CommissionableDataProvider.h>

#include <array>
#include <cstdint>

namespace espectre {

class MatterCommissioningDataProvider final : public chip::DeviceLayer::CommissionableDataProvider {
 public:
  CHIP_ERROR initialize();

  uint32_t setup_passcode() const { return setup_passcode_; }
  uint16_t setup_discriminator() const { return setup_discriminator_; }
  bool generated_on_boot() const { return generated_on_boot_; }

  CHIP_ERROR GetSetupDiscriminator(uint16_t &setup_discriminator) override;
  CHIP_ERROR SetSetupDiscriminator(uint16_t setup_discriminator) override;
  CHIP_ERROR GetSpake2pIterationCount(uint32_t &iteration_count) override;
  CHIP_ERROR GetSpake2pSalt(chip::MutableByteSpan &salt_buffer) override;
  CHIP_ERROR GetSpake2pVerifier(chip::MutableByteSpan &verifier_buffer,
                               size_t &verifier_length) override;
  CHIP_ERROR GetSetupPasscode(uint32_t &setup_passcode) override;
  CHIP_ERROR SetSetupPasscode(uint32_t setup_passcode) override;

 private:
  CHIP_ERROR generate_and_store();
  CHIP_ERROR ensure_verifier();

  uint32_t setup_passcode_ = 0;
  uint16_t setup_discriminator_ = 0;
  uint32_t iteration_count_ = 1000;
  std::array<uint8_t, chip::Crypto::kSpake2p_Max_PBKDF_Salt_Length> salt_{};
  size_t salt_length_ = 0;
  std::array<uint8_t, chip::Crypto::kSpake2p_VerifierSerialized_Length> verifier_{};
  bool verifier_ready_ = false;
  bool generated_on_boot_ = false;
};

}  // namespace espectre

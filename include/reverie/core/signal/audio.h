#pragma once
#include "reverie/core/base.h"
#include <filesystem>

namespace reverie {
namespace signal {

class Audio : public ReverieBase {
public:
    /// Default constructor: initialize empty Audio
	Audio(const Device& device = Device()) : ReverieBase(device) {}

    /// Read Audio from .wav file
	bool read(const std::filesystem::path& wavfile);

	/// Write Audio to .wav file
	void write(const std::filesystem::path& wavfile) const;

    const float* samples() const { return m_samples.data(); }
    uint32_t samplerate() const { return m_samplerate; }

    size_t num_samples() const { return m_samples.size() / m_num_channels; }
    uint16_t num_channels() const { return m_num_channels; }

    std::string to_string() const;

private:
    Buffer<float> m_samples;

    uint16_t m_num_channels = 0;
    uint32_t m_samplerate = 0;
};

} // namespace signal
} // namespace reverie
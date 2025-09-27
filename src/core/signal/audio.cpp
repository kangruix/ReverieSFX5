#include "reverie/core/signal/audio.h"
#include <fstream>
#include <sstream>
#include <cstring>

namespace reverie {
namespace signal {

bool Audio::read(const std::filesystem::path& wavfile) {
    std::ifstream fin(wavfile, std::ios::binary);
    if (!fin) throw std::runtime_error("Audio: failed to open file " + wavfile.string());

    auto readLE16 = [&](uint16_t& out) -> void {
        uint8_t b[2]; fin.read(reinterpret_cast<char*>(b), 2);
        out = b[0] | (b[1] << 8);
    };
    auto readLE32 = [&](uint32_t& out) -> void {
        uint8_t b[4]; fin.read(reinterpret_cast<char*>(b), 4);
        out = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
    };
    uint16_t format = 0, blockalign = 0, bitdepth = 0;
    uint32_t byterate = 0;
    std::vector<float> samples;

    // --- RIFF header ---
    char riff[4]; fin.read(riff, 4);
    uint32_t riffsize; readLE32(riffsize);
    char wave[4]; fin.read(wave, 4);

    while (fin) {
        char chunkID[4]; fin.read(chunkID, 4);
        uint32_t chunksize = 0; readLE32(chunksize);

        // --- fmt chunk ---
        if (std::strncmp(chunkID, "fmt ", 4) == 0) {
            readLE16(format);
            readLE16(m_num_channels);
            readLE32(m_samplerate);
            readLE32(byterate);
            readLE16(blockalign);
            readLE16(bitdepth);
            fin.seekg(chunksize - 16, std::ios::cur);  // skip any extra
        }
        // --- data chunk ---
        else if (std::strncmp(chunkID, "data", 4) == 0) {
            const uint32_t num_frames = chunksize / (bitdepth / 8 * m_num_channels);

            samples.reserve(num_frames * m_num_channels);
            for (uint32_t i = 0; i < num_frames; ++i) {
                for (uint16_t ch = 0; ch < m_num_channels; ++ch) {
                    float sample;
                    if (format == 1) {  // PCM
                        if (bitdepth == 16) {
                            int16_t s; fin.read(reinterpret_cast<char*>(&s), 2);
                            sample = s / 32768.f;
                        }
                        else if (bitdepth == 32) {
                            int32_t s; fin.read(reinterpret_cast<char*>(&s), 4);
                            sample = s / 2147483648.f;
                        }
                        else {
                            //LOGW("Audio: unsupported PCM bit depth" + std::to_string(bitdepth));
                            return false;
                        }
                    }
                    else if (format == 3) {  // IEEE float
                        if (bitdepth == 32) {
                            float s; fin.read(reinterpret_cast<char*>(&s), 4);
                            sample = s;
                        }
                        else if (bitdepth == 64) {
                            double s; fin.read(reinterpret_cast<char*>(&s), 8);
                            sample = static_cast<float>(s);
                        }
                        else {
                            //LOGE("Audio: unsupported IEEE float bit depth" + std::to_string(bitdepth));
                            return false;
                        }
                    }
                    else {
                        //LOGE("Audio: unsupported .wav format (only PCM and IEEE float)\n");
                        return false;
                    }
                    samples.push_back(sample);
                }
            }
            break;
        }
        else { fin.seekg(chunksize, std::ios::cur); }  // skip unknown chunk
    }
    m_samples = Buffer<float>(samples, m_device);
    return true;
}

void Audio::write(const std::filesystem::path& wavfile) const {
	std::ofstream fout(wavfile, std::ios::binary);
    
    auto writeLE16 = [&](uint16_t in) -> void {
        char b[2] = { static_cast<char>(in & 0xFF),
                      static_cast<char>((in >> 8) & 0xFF) };
        fout.write(b, 2);
    };
    auto writeLE32 = [&](uint32_t in) -> void {
        char b[4] = { static_cast<char>(in & 0xFF),
                      static_cast<char>((in >> 8) & 0xFF),
                      static_cast<char>((in >> 16) & 0xFF),
                      static_cast<char>((in >> 24) & 0xFF) };
        fout.write(b, 4);
    };
    const uint16_t format = 3;  // IEEE float
    const uint32_t num_frames = static_cast<uint32_t>(m_samples.size() / m_num_channels);
    const uint16_t bitdepth = 32;
    const uint16_t blockalign = m_num_channels * (bitdepth / 8);
    const uint32_t byterate = m_samplerate * blockalign;
    const uint32_t chunksize = num_frames * blockalign;
    const uint32_t riffsize = 4 /*WAVE*/ + 8 + 16 /*fmt*/ + 8 + chunksize;

    // --- RIFF header ---
    fout.write("RIFF", 4);
    writeLE32(riffsize);
    fout.write("WAVE", 4);

    // --- fmt chunk ---
    fout.write("fmt ", 4);
    writeLE32(16);
    writeLE16(format);
    writeLE16(m_num_channels);
    writeLE32(m_samplerate);
    writeLE32(byterate);
    writeLE16(blockalign);
    writeLE16(bitdepth);

    // --- data chunk ---
    fout.write("data", 4);
    writeLE32(chunksize);

    const float* samples = m_samples.view();
    for (uint32_t i = 0; i < num_frames; ++i) {
        for (uint16_t ch = 0; ch < m_num_channels; ++ch) {
            float sample = samples[i * m_num_channels + ch];
            fout.write(reinterpret_cast<const char*>(&sample), sizeof(float));
        }
    }
}

std::string Audio::to_string() const {
    std::stringstream ss;
    double dur = static_cast<double>(num_samples()) / m_samplerate;
    ss << "Audio[\n" <<
        "  duration = " << dur << " sec,\n" <<
        "  samplerate = " << m_samplerate << ",\n" <<
        "  num_channels = " << m_num_channels << ",\n" <<
        "  device = " << m_device.to_string() << "\n" <<
        "]";
    return ss.str();
}

} // namespace signal
} // namespace reverie
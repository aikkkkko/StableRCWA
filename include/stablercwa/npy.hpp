// Minimal NumPy NPY v1 writer: C layout, explicit little endian float64 / complex128 / uint8.
// No NumPy or third-party C++ dependency. A writer must be finish()ed before publishing its file.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "matrix.hpp"

namespace stablercwa {
namespace npy {

class Writer {
public:
    Writer(const std::filesystem::path& path, std::string dtype, const std::vector<std::size_t>& shape)
        : out_(path, std::ios::binary | std::ios::trunc), dtype_(std::move(dtype)) {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                      "NPY output needs IEEE-754 binary64");
        if (dtype_ != "<f8" && dtype_ != "<c16" && dtype_ != "|u1")
            throw std::invalid_argument("NPY: unsupported dtype");
        if (!out_) throw std::runtime_error("NPY: cannot create output file");
        std::string tuple = "(";
        for (std::size_t dimension : shape) {
            if (dimension != 0 && expected_ > (std::numeric_limits<std::size_t>::max)() / dimension)
                throw std::invalid_argument("NPY: shape overflow");
            expected_ *= dimension;
            tuple += std::to_string(dimension) + ", ";
        }
        tuple += ")";
        std::string header = "{'descr': '" + dtype_ + "', 'fortran_order': False, 'shape': " + tuple + ", }";
        const std::size_t padding = (64 - (10 + header.size() + 1) % 64) % 64;
        header.append(padding, ' ');
        header += '\n';
        if (header.size() > 65535) throw std::invalid_argument("NPY: header too long");
        const unsigned char prefix[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
            static_cast<unsigned char>(header.size() & 255), static_cast<unsigned char>(header.size() >> 8)};
        out_.write(reinterpret_cast<const char*>(prefix), sizeof(prefix));
        out_.write(header.data(), static_cast<std::streamsize>(header.size()));
        const std::uint16_t marker = 1;
        little_endian_ = *reinterpret_cast<const unsigned char*>(&marker) == 1;
    }
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    void value(double value) {
        need("<f8");
        binary64(value);
    }
    void value(cplx value) {
        need("<c16");
        binary64(value.real());
        binary64(value.imag());
    }
    void byte(std::uint8_t value) {
        need("|u1");
        append(&value, 1);
    }
    void finish() {
        if (count_ != expected_) throw std::runtime_error("NPY: element count does not match shape");
        flush_buffer();
        out_.flush();
        if (!out_) throw std::runtime_error("NPY: writing output failed");
        out_.close();
        if (!out_) throw std::runtime_error("NPY: closing output failed");
    }

private:
    void need(const char* dtype) {
        if (dtype_ != dtype || count_ >= expected_) throw std::invalid_argument("NPY: dtype or count mismatch");
        ++count_;
    }
    void binary64(double value) {
        unsigned char bytes[8];
        std::memcpy(bytes, &value, 8);
        if (!little_endian_) for (int i = 0; i < 4; ++i) std::swap(bytes[i], bytes[7 - i]);
        append(bytes, 8);
    }
    void append(const unsigned char* bytes, std::size_t count) {
        if (buffered_ + count > buffer_.size()) flush_buffer();
        std::memcpy(buffer_.data() + buffered_, bytes, count);
        buffered_ += count;
    }
    void flush_buffer() {
        out_.write(reinterpret_cast<const char*>(buffer_.data()), static_cast<std::streamsize>(buffered_));
        buffered_ = 0;
        if (!out_) throw std::runtime_error("NPY: writing output failed");
    }
    std::ofstream out_;
    std::string dtype_;
    std::size_t expected_ = 1, count_ = 0, buffered_ = 0;
    bool little_endian_ = true;
    std::array<unsigned char, 65536> buffer_{};
};

}  // namespace npy
}  // namespace stablercwa

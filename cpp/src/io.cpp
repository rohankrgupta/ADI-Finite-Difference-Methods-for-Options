#include "io.hpp"

#include <bit>
#include <fstream>
#include <stdexcept>

static_assert(std::endian::native == std::endian::little, "the .bin files are little-endian");

namespace adi {

std::vector<double> read_bin(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    const auto bytes = static_cast<std::size_t>(in.tellg());
    if (bytes % sizeof(double) != 0) {
        throw std::runtime_error(path.string() + " does not hold a whole number of float64 values");
    }
    std::vector<double> values(bytes / sizeof(double));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("cannot read " + path.string());
    return values;
}

void write_bin(const std::filesystem::path& path, const std::vector<double>& values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

}  // namespace adi

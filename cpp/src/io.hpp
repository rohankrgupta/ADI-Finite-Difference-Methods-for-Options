// Reading and writing the raw float64 files shared with Python (see fixtures/README.md).
#pragma once

#include <filesystem>
#include <vector>

namespace adi {

// Read a whole .bin file: float64, little-endian, no header.
std::vector<double> read_bin(const std::filesystem::path& path);

// Write values in the same format.
void write_bin(const std::filesystem::path& path, const std::vector<double>& values);

}  // namespace adi

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../report/gpu/program.cpp"

int main() {
  std::vector<uint64_t> values = {
      0x123456789abcdef0ULL,
      7,
      11,
  };

  std::cout << "making array with elems:";
  for (uint64_t value : values) {
    std::cout << " 0x" << std::hex << value;
  }
  std::cout << std::dec << std::endl;

  catena_mem_t mem = {
      values.data(),
      values.size() * sizeof(uint64_t),
  };

  uint64_t out = 0;
  program_array_head_u64(mem, &out);

  std::cout << "array_head_u64: 0x" << std::hex << out << std::dec << std::endl;

  if (out != values.front()) {
    std::cerr << "verification: FAILED" << std::endl;
    return 1;
  }

  std::cout << "verification: PASSED" << std::endl;
  return 0;
}

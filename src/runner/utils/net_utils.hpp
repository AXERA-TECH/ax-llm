#pragma once

#include <string>
#include <vector>

namespace axllm {

// Returns non-loopback IPv4 addresses of the local machine (best-effort).
// Useful for printing accessible server URLs when binding to 0.0.0.0.
std::vector<std::string> get_local_ipv4_addresses();

// Returns true if a TCP port is available.
bool is_port_available(int port, const char **error);

} // namespace axllm

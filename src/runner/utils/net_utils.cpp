#include "net_utils.hpp"

#include <unordered_set>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#else
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

namespace axllm {

static bool is_link_local_ipv4(const std::string &ip)
{
    return ip.rfind("169.254.", 0) == 0;
}

std::vector<std::string> get_local_ipv4_addresses()
{
    std::vector<std::string> ips;
    std::unordered_set<std::string> seen;

    auto push_unique = [&](std::string ip)
    {
        if (ip.empty()) return;
        if (ip == "127.0.0.1") return;
        if (is_link_local_ipv4(ip)) return;
        if (seen.insert(ip).second) ips.push_back(std::move(ip));
    };

#ifdef _WIN32
    ULONG flags = GAA_FLAG_INCLUDE_PREFIX;
    ULONG family = AF_INET; // IPv4 only
    ULONG buffer_size = 15000;

    std::vector<BYTE> buffer(buffer_size);
    PIP_ADAPTER_ADDRESSES adapter_addresses = reinterpret_cast<PIP_ADAPTER_ADDRESSES>(buffer.data());

    DWORD ret = GetAdaptersAddresses(family, flags, nullptr, adapter_addresses, &buffer_size);
    if (ret == ERROR_BUFFER_OVERFLOW)
    {
        buffer.resize(buffer_size);
        adapter_addresses = reinterpret_cast<PIP_ADAPTER_ADDRESSES>(buffer.data());
        ret = GetAdaptersAddresses(family, flags, nullptr, adapter_addresses, &buffer_size);
    }
    if (ret != ERROR_SUCCESS)
    {
        return ips;
    }

    for (PIP_ADAPTER_ADDRESSES adapter = adapter_addresses; adapter != nullptr; adapter = adapter->Next)
    {
        if (adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK || adapter->OperStatus != IfOperStatusUp)
        {
            continue;
        }

        for (PIP_ADAPTER_UNICAST_ADDRESS unicast = adapter->FirstUnicastAddress; unicast != nullptr; unicast = unicast->Next)
        {
            sockaddr *addr = unicast->Address.lpSockaddr;
            if (!addr) continue;
            if (addr->sa_family != AF_INET) continue;

            char ip_str[INET_ADDRSTRLEN] = {};
            auto *sin = reinterpret_cast<sockaddr_in *>(addr);
            if (!inet_ntop(AF_INET, &(sin->sin_addr), ip_str, INET_ADDRSTRLEN))
            {
                continue;
            }
            push_unique(std::string(ip_str));
        }
    }
#else
    struct ifaddrs *ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1)
    {
        return ips;
    }

    for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next)
    {
        if (!ifa->ifa_addr) continue;
        if (ifa->ifa_addr->sa_family != AF_INET) continue;

        if (ifa->ifa_flags & IFF_LOOPBACK) continue;
        if (!(ifa->ifa_flags & IFF_UP)) continue;

        char addrstr[INET_ADDRSTRLEN] = {};
        auto *sin = reinterpret_cast<sockaddr_in *>(ifa->ifa_addr);
        if (!inet_ntop(AF_INET, &sin->sin_addr, addrstr, INET_ADDRSTRLEN))
        {
            continue;
        }
        push_unique(std::string(addrstr));
    }

    freeifaddrs(ifaddr);
#endif

    return ips;
}

} // namespace axllm


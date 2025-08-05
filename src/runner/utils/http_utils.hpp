#pragma once
#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>
#include <thread>

/**
 * @brief Attempts to establish a TCP connection to a specified host and port.
 *
 * This function creates a socket and tries to connect to a server specified by
 * the host and port parameters. It returns true if the connection is successful,
 * otherwise it returns false and outputs an error message to standard error.
 *
 * @param host The IP address of the server to connect to.
 * @param port The port number of the server to connect to.
 *
 * @return true if the connection is successfully established, false otherwise.
 */
static bool test_connect(const std::string &host, int port)
{
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        // std::cerr << "Socket creation failed\n";
        return false;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0)
    {
        // std::cerr << "IP address conversion failed\n";
        close(sock);
        return false;
    }

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        // std::cerr << "Connection failed\n";
        close(sock);
        return false;
    }

    close(sock);
    return true;
}

/**
 * @brief Attempts to establish an HTTP connection to a specified URL with a timeout.
 *
 * This function parses the provided HTTP URL to extract the host and port information,
 * and attempts to establish a TCP connection using the `test_connect` function.
 * It retries the connection until the specified timeout is reached.
 *
 * @param http_url The HTTP URL of the server to connect to.
 * @param timeout The maximum number of seconds to keep attempting the connection.
 *
 * @return true if the connection is successfully established within the timeout period,
 * false otherwise.
 */
static bool test_connect_http(const std::string &http_url, int timeout)
{
    size_t pos = http_url.find("://");
    if (pos == std::string::npos)
        return false;
    std::string host = http_url.substr(pos + 3);
    pos = host.find('/');
    if (pos != std::string::npos)
        host = host.substr(0, pos);
    pos = host.find(':');
    int port = 80;
    if (pos != std::string::npos)
    {
        port = std::stoi(host.substr(pos + 1));
        host = host.substr(0, pos);
    }
    else
    {
        return false;
    }

    if (host == "localhost")
        host = "127.0.0.1";
    int tmp = timeout;
    while (timeout--)
    {
        if (test_connect(host, port))
            return true;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        printf("\033[1;30;31m"
               "connect failed %s, try again in %2d/%2d \n"
               "\033[0m",
               http_url.c_str(), timeout, tmp);
    }
    return false;
}
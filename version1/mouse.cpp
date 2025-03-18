#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <string>
#include <sstream>
#include "./json.hpp" // Ensure this file is in the same directory

#pragma comment(lib, "Ws2_32.lib")

int main() {
    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed: " << WSAGetLastError() << std::endl;
        return 1;
    }

    // Create UDP socket
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        std::cerr << "Socket creation failed: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }

    // Setup address structure
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY; // Listen on all interfaces
    serverAddr.sin_port = htons(12345);      // Match magic.py's port

    // Bind the socket
    if (bind(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        std::cerr << "Bind failed: " << WSAGetLastError() << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }

    std::cout << "UDP listener started on port 12345..." << std::endl;

    // Accumulators for smooth movement
    double accum_dx = 0.0;
    double accum_dy = 0.0;

    while (true) {
        char buffer[1024] = {0};
        sockaddr_in clientAddr;
        int clientAddrLen = sizeof(clientAddr);

        int bytesReceived = recvfrom(sock, buffer, sizeof(buffer) - 1, 0, (sockaddr*)&clientAddr, &clientAddrLen);
        if (bytesReceived == SOCKET_ERROR) {
            std::cout << "Receive failed: " << WSAGetLastError() << " (Code: " << WSAGetLastError() << ")" << std::endl;
            Sleep(100); // Brief pause to avoid spamming
            continue;
        }

        buffer[bytesReceived] = '\0';
        std::cout << "Received raw: " << buffer << " (Length: " << bytesReceived << ")" << std::endl;

        // Parse JSON using nlohmann/json
        try {
            std::string jsonStr(buffer);
            auto jsonData = nlohmann::json::parse(jsonStr);
            double dx = jsonData["dx"].get<double>();
            double dy = jsonData["dy"].get<double>();
            std::cout << "Parsed nudge: dx=" << dx << ", dy=" << dy << std::endl;

            // Accumulate nudges for smooth movement
            accum_dx += dx;
            accum_dy += dy;
            int move_x = static_cast<int>(accum_dx);
            int move_y = static_cast<int>(accum_dy);
            std::cout << "Accumulated: accum_dx=" << accum_dx << ", accum_dy=" << accum_dy << std::endl;

            if (move_x != 0 || move_y != 0) {
                POINT current_pos; // Declare POINT structure
                if (GetCursorPos(&current_pos)) { // Pass address with &
                    int new_x = current_pos.x + move_x;
                    int new_y = current_pos.y + move_y;

                    // Clamp to screen bounds (assuming 1920x1080)
                    new_x = std::max(0, std::min(new_x, 1919));
                    new_y = std::max(0, std::min(new_y, 1079));

                    SetCursorPos(new_x, new_y);
                    std::cout << "Moved to: (" << new_x << ", " << new_y << ")" << std::endl;

                    accum_dx -= move_x;
                    accum_dy -= move_y;
                } else {
                    std::cout << "GetCursorPos failed: " << GetLastError() << std::endl;
                }
            } else {
                std::cout << "Nudge too small" << std::endl;
            }
        } catch (const nlohmann::json::parse_error& e) {
            std::cout << "JSON parsing error: " << e.what() << " (Input: " << buffer << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    }

    closesocket(sock);
    WSACleanup();
    return 0;
}
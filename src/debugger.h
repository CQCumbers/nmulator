#ifndef DEBUGGER_H
#define DEBUGGER_H

#include <stdio.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>

#define __STDC_FORMAT_MACROS
#include <cinttypes>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace Debugger {
  constexpr unsigned buf_size = 2048;

  /* === GDB Socket interaction == */ 

  void recv_gdb(int sockfd, char *cmd_buf) {
    uint8_t cmd_c = 0, cmd_idx = 0;
    // ignore acks and invalid cmds
    recv(sockfd, &cmd_c, 1, MSG_WAITALL);
    if (cmd_c == '+') return;
    if (cmd_c != '$') printf("Invalid gdb command: %x\n", cmd_c), exit(1);
    // read cmd packet data (no RLE or escapes)
    while (recv(sockfd, &cmd_c, 1, MSG_WAITALL), cmd_c != '#')
      cmd_buf[cmd_idx++] = cmd_c;
    // ignore checksum and ack
    recv(sockfd, &cmd_c, 1, MSG_WAITALL);
    recv(sockfd, &cmd_c, 1, MSG_WAITALL);
    send(sockfd, "+", 1, 0);
  }

  unsigned checksum(const char *data, unsigned len) {
    return std::accumulate(data, data + len, 0) & 0xff;
  }

  void send_gdb(int sockfd, const char *data) {
    // use vargs to format send data
    char send_buf[buf_size] = {0};
    unsigned len = strlen(data);
    memcpy(send_buf + 1, data, len);
    send_buf[0] = '$'; send_buf[len + 1] = '#';
    sprintf(send_buf + len + 2, "%02x", checksum(data, len));
    char *ptr = send_buf, *end = send_buf + len + 4;
    while (ptr != end) ptr += send(sockfd, ptr, end - ptr, 0);
  }

  /* === GDB Protocol commands == */

  char cmd_buf[buf_size] = {0};
  char reg_names[33][3] = {
    "r0", "at", "v0", "v1", "a0", "a1", "a2", "a3",
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
    "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra", "pc",
  };
  int gdb_sock = 0;

  void query(int sockfd, const char *cmd_buf) {
    char buf[buf_size - 4] = {0};
    if (strncmp(cmd_buf, "qSupported", strlen("qSupported")) == 0)
      send_gdb(sockfd, "PacketSize=2047");
    else if (strncmp(cmd_buf, "qHostInfo", strlen("qHostInfo")) == 0)
      send_gdb(sockfd, "triple:6d6970732d6c696e75782d676e75;ptrsize:8;endian:big;");
    else if (strncmp(cmd_buf, "qProcessInfo", strlen("qProcessInfo")) == 0)
      send_gdb(sockfd, "triple:6d6970732d6c696e75782d676e75;pid:1;");
    else if (strncmp(cmd_buf, "qfThreadInfo", strlen("qfThreadInfo")) == 0)
      send_gdb(sockfd, "m-1");
    else if (strncmp(cmd_buf, "qsThreadInfo", strlen("qsThreadInfo")) == 0)
      send_gdb(sockfd, "l");
    else if (strncmp(cmd_buf, "qC", strlen("qC")) == 0)
      send_gdb(sockfd, "1");  // Dummy PID
    else if (strncmp(cmd_buf, "qRegisterInfo", strlen("qRegisterInfo")) == 0) {
      unsigned long i = strtoul(cmd_buf + strlen("qRegisterInfo"), nullptr, 16);
      if (i >= 33) return send_gdb(sockfd, "E45");
      char *ptr = buf + sprintf(buf, "name:%s;bitsize:64;", reg_names[i]);
      ptr += sprintf(ptr, "offset:%lu;encoding:sint;format:hex;", i * 8);
      if (i == 29) sprintf(ptr, "generic:sp;");
      if (i == 30) sprintf(ptr, "generic:fp;");
      if (i == 31) sprintf(ptr, "generic:ra;");
      if (i == 32) sprintf(ptr, "generic:pc;");
      send_gdb(sockfd, buf);
    } else send_gdb(sockfd, "");
  }

  void read_regs(int sockfd) {
    char buf[buf_size - 4] = {0};
    for (unsigned i = 0; i < 32; ++i)
      sprintf(buf + i * 16, "%016" PRIx64, R4300::reg_array[i]);
    sprintf(buf + 32 * 16, "%016" PRIx64, static_cast<uint64_t>(R4300::pc));
    send_gdb(sockfd, buf);
  }

  void read_reg(int sockfd, const char *cmd_buf) {
    char buf[17] = {0};
    uint32_t idx = strtoul(cmd_buf + 1, nullptr, 16);
    if (idx != 32) sprintf(buf, "%016" PRIx64, R4300::reg_array[idx]);
    else sprintf(buf, "%016" PRIx64, static_cast<uint64_t>(R4300::pc));
    send_gdb(sockfd, buf);
  }

  void write_reg(int sockfd, const char *cmd_buf) {
    char *ptr = (char*)cmd_buf;
    uint32_t idx = strtoul(cmd_buf + 1, &ptr, 16);
    printf("Writing to register %x: %lx\n", idx, strtoul(ptr + 1, nullptr, 16));
    if (idx != 32) R4300::reg_array[idx] = strtoul(ptr + 1, nullptr, 16);
    else R4300::pc = strtoul(ptr + 1, nullptr, 16);
    send_gdb(sockfd, "OK");
  }

  void read_mem(int sockfd, const char *cmd_buf) {
    char buf[buf_size - 4] = {0}, *ptr = (char*)cmd_buf;
    uint32_t addr = strtoul(cmd_buf + 1, &ptr, 16);
    uint32_t len = strtoul(ptr + 1, nullptr, 16);
    for (unsigned i = 0; i < len; i += 8)
      sprintf(buf + i * 2, "%016" PRIx64, R4300::read<uint64_t>(addr + i));
    send_gdb(sockfd, buf);
  }

  template <bool active>
  void set_break(int sockfd, const char *cmd_buf) {
    // Execution breakpoints only
    if (cmd_buf[1] != '0') return send_gdb(sockfd, "E01");
    uint32_t addr = strtoul(cmd_buf + 3, nullptr, 16);
    R4300::breaks[addr] = active; send_gdb(sockfd, "OK");
  }

  void update() {
    send_gdb(gdb_sock, "S05");
    while (true) {
      memset(cmd_buf, 0, buf_size);
      recv_gdb(gdb_sock, cmd_buf);
      switch (cmd_buf[0]) {
        case 0x0: break;
        case '?': send_gdb(gdb_sock, "S05"); break;  // Stopped due to trap
        case 'c': case 'C': R4300::broke = false;
        case 's': R4300::moved = false; return;
        case 'g': read_regs(gdb_sock); break;
        case 'p': read_reg(gdb_sock, cmd_buf); break;
        case 'P': write_reg(gdb_sock, cmd_buf); break;
        case 'H': send_gdb(gdb_sock, "OK"); break;   // Switch to thread
        case 'k': printf("Killed by gdb\n"); exit(0);
        case 'm': read_mem(gdb_sock, cmd_buf); break;
        case 'q': query(gdb_sock, cmd_buf); break;
        case 'z': set_break<false>(gdb_sock, cmd_buf); break;
        case 'Z': set_break<true>(gdb_sock, cmd_buf); break;
        default: send_gdb(gdb_sock, ""); break;
      }
    }
  }

  void init(int port) {
    sockaddr_in server_addr = {}, client_addr = {};
    unsigned addr_size = sizeof(sockaddr_in);
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    int tmpsock = socket(AF_INET, SOCK_STREAM, 0);
    if (tmpsock == -1) printf("Failed to create gdb socket\n"), exit(1);
    int reuse = setsockopt(tmpsock, SOL_SOCKET, SO_REUSEADDR, "\x01\x00\x00\x00", 4);
    if (reuse < 0) printf("Failed to set gdb socket option\n"), exit(1);
    int bound = bind(tmpsock, (sockaddr*)(&server_addr), addr_size);
    if (bound < 0) printf("Failed to bind gdb socket\n"), exit(1);
    listen(tmpsock, 5), printf("Listening on port %d\n", port);

    gdb_sock = accept(tmpsock, (sockaddr*)(&client_addr), &addr_size);
    if (gdb_sock < 0) printf("Failed to accept gdb client\n"), exit(1);
    if (tmpsock != -1) shutdown(tmpsock, SHUT_RDWR);
    printf("Connected to gdb client\n");
    R4300::broke = true, update();
  }

  void check() {
    fd_set sockset = {0};
    FD_SET(gdb_sock, &sockset);
    struct timeval timeout = {};
    int socks = select(0, &sockset, nullptr, nullptr, &timeout);
    printf("Selected sockets: %x\n", socks);
    if (socks < 1) return;
    R4300::broke = true; send_gdb(gdb_sock, "OK");
  }
}

#endif

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
  char reg_names[70][8] = {
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
    "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31",
    "sr", "lo", "hi", "bad", "cause", "pc",
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
    "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
    "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
    "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31",
  };
  int gdb_sock = 0;

  uint64_t reg_vals(uint32_t idx) {
    if (idx < 32) return R4300::reg_array[idx];
    if (idx >= 38) return R4300::reg_array[idx + R4300::dev_cop1 - 38];
    switch (idx) {
      case 32: return R4300::reg_array[12 + R4300::dev_cop0];
      case 33: return R4300::reg_array[R4300::lo];
      case 34: return R4300::reg_array[R4300::hi];
      case 35: return R4300::reg_array[8 + R4300::dev_cop0];
      case 36: return R4300::reg_array[13 + R4300::dev_cop0];
      case 37: return R4300::pc;
      default: printf("Invalid register: %x\n", idx), exit(1);
    }
  }

  void query(int sockfd, const char *cmd_buf) {
    char buf[buf_size - 4] = {0};
    if (strncmp(cmd_buf, "qSupported", strlen("qSupported")) == 0)
      send_gdb(sockfd, "PacketSize=2047");
    else if (strncmp(cmd_buf, "qHostInfo", strlen("qHostInfo")) == 0)
      send_gdb(sockfd, "triple:6d69707336342d7367692d69726978;ptrsize:8;endian:big;");
    else if (strncmp(cmd_buf, "qProcessInfo", strlen("qProcessInfo")) == 0)
      send_gdb(sockfd, "triple:6d69707336342d7367692d69726978;pid:1;");
    else if (strncmp(cmd_buf, "qfThreadInfo", strlen("qfThreadInfo")) == 0)
      send_gdb(sockfd, "m-1");
    else if (strncmp(cmd_buf, "qsThreadInfo", strlen("qsThreadInfo")) == 0)
      send_gdb(sockfd, "l");
    else if (strncmp(cmd_buf, "qC", strlen("qC")) == 0)
      send_gdb(sockfd, "1");  // Dummy PID
    else if (strncmp(cmd_buf, "qRegisterInfo", strlen("qRegisterInfo")) == 0) {
      unsigned long i = strtoul(cmd_buf + strlen("qRegisterInfo"), nullptr, 16);
      if (i >= 70) return send_gdb(sockfd, "E45");
      char *ptr = buf + sprintf(buf, "name:%s;bitsize:64;gcc:%lu;", reg_names[i], i);
      const char *set = (i < 32 ? "General Purpose" : "Control");
      if (i >= 38) set = "Floating Point";
      ptr += sprintf(ptr, "offset:%lu;encoding:uint;format:hex;", i * 16);
      ptr += sprintf(ptr, "set:%s Registers;dwarf:%lu;", set, i); 
      if (i == 29) sprintf(ptr, "generic:sp;");
      if (i == 30) sprintf(ptr, "generic:fp;");
      if (i == 31) sprintf(ptr, "generic:ra;");
      if (i == 37) sprintf(ptr, "generic:pc;");
      send_gdb(sockfd, buf);
    } else send_gdb(sockfd, "");
  }

  void read_regs(int sockfd) {
    char buf[buf_size - 4] = {0};
    for (unsigned idx = 0; idx < 70; ++idx)
      sprintf(buf + idx * 16, "%016llx", reg_vals(idx));
    send_gdb(sockfd, buf);
  }

  void read_reg(int sockfd, const char *cmd_buf) {
    char buf[17] = {0};
    uint32_t idx = strtoul(cmd_buf + 1, nullptr, 16);
    sprintf(buf, "%016llx", reg_vals(idx));
    send_gdb(sockfd, buf);
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
    uint32_t addr = strtoul(cmd_buf + 3, nullptr, 16);
    switch (cmd_buf[1]) {
      case '0':
        R4300::breaks[addr] = active;
        return send_gdb(sockfd, "OK");
      case '2':
        R4300::watch_w[addr] = active;
        return send_gdb(sockfd, "OK");
      default:
        return send_gdb(sockfd, "E01");
    }
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

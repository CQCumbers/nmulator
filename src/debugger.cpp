#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
#else
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <poll.h>
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nmulator.h"

using sock_t = decltype(socket(0, 0, 0));
const unsigned buf_size = 2048;

/* === GDB Socket interaction === */

static sock_t gdb_sock;
static char cmd_buf[buf_size];

#ifdef _WIN32
static void cleanup_socket() {
  int cleaned = WSACleanup();
}
#endif

// decode command data from gdb server
static void recv_gdb(sock_t sockfd, char *cmd_buf) {
  char cmd_c = 0; uint8_t cmd_idx = 0;
  // ignore acks and invalid cmds
  if (!recv(sockfd, &cmd_c, 1, MSG_WAITALL)) return;
  if (cmd_c == '+' || cmd_c == 3) return;
  if (cmd_c != '$') printf("Invalid gdb command: %x\n", cmd_c), exit(1);
  // read cmd packet data (no RLE or escapes)
  while (recv(sockfd, &cmd_c, 1, MSG_WAITALL) && cmd_c != '#')
    cmd_buf[cmd_idx++] = cmd_c;
  // ignore checksum and ack
  if (!recv(sockfd, &cmd_c, 1, MSG_WAITALL)) return;
  if (!recv(sockfd, &cmd_c, 1, MSG_WAITALL)) return;
  send(sockfd, "+", 1, 0);
}

// compute checksum for sent data
static unsigned checksum(const char *data, unsigned len) {
  unsigned sum = 0;
  for (unsigned i = 0; i < len; ++i) sum += data[i];
  return sum & 0xff;
}

// encode data to send to gdb server
static void send_gdb(sock_t sockfd, const char *data) {
  char send_buf[buf_size] = {0};
  unsigned len = strlen(data);
  memcpy(send_buf + 1, data, len);
  send_buf[0] = '$'; send_buf[len + 1] = '#';
  sprintf(send_buf + len + 2, "%02x", checksum(data, len));
  char *ptr = send_buf, *end = send_buf + len + 4;
  while (ptr != end) ptr += send(sockfd, ptr, end - ptr, 0);
}

/* === GDB Protocol commands === */

static DbgConfig cfg;
const unsigned n_regs = 70;
char reg_names[n_regs][8] = {
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

// handle gdb host architecture queries
static void query(sock_t sockfd, const char *cmd_buf) {
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
    unsigned long i = strtoul(cmd_buf + strlen("qRegisterInfo"), NULL, 16);
    if (i >= n_regs) return send_gdb(sockfd, "E45");
    char *ptr = buf + sprintf(buf, "name:%s;bitsize:64;gcc:%lu;", reg_names[i], i);
    const char *set = "Control";
    if (i <= 31) set = "General Purpose";
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

// read all available registers
static void read_regs(sock_t sockfd) {
  char buf[buf_size - 4] = {0};
  for (unsigned idx = 0; idx < n_regs; ++idx)
    sprintf(buf + idx * 16, "%016llx", cfg.read_reg(idx));
  send_gdb(sockfd, buf);
}

// write all available registers
static void write_regs(sock_t sockfd, const char *cmd_buf) {
  char buf[17] = {0};
  for (unsigned idx = 0; idx < n_regs; ++idx) {
    memcpy(buf, cmd_buf + 1 + idx * 16, 16);
    cfg.write_reg(idx, strtoul(buf, NULL, 16));
  }
  send_gdb(sockfd, "OK");
}

// read specific register via callback
static void read_reg(sock_t sockfd, const char *cmd_buf) {
  char buf[17] = {0};
  uint32_t idx = strtoul(cmd_buf + 1, NULL, 16);
  sprintf(buf, "%016llx", cfg.read_reg(idx));
  send_gdb(sockfd, buf);
}

// write specific register via callback
static void write_reg(sock_t sockfd, const char *cmd_buf) {
  char *ptr = (char*)cmd_buf;
  uint32_t idx = strtoul(cmd_buf + 1, &ptr, 16);
  cfg.write_reg(idx, strtoul(ptr + 1, NULL, 16));
  send_gdb(sockfd, "OK");
}

// read from memory address via callback
static void read_mem(sock_t sockfd, const char *cmd_buf) {
  char buf[buf_size - 4] = {0}, *ptr = (char*)cmd_buf;
  uint32_t addr = strtoul(cmd_buf + 1, &ptr, 16);
  uint32_t len = strtoul(ptr + 1, NULL, 16);
  for (unsigned i = 0; i < len; ++i)
    sprintf(buf + i * 2, "%02x", cfg.read_mem(addr + i));
  send_gdb(sockfd, buf);
}

// write to memory address via callback
static void write_mem(sock_t sockfd, const char *cmd_buf) {
  char buf[3] = {0}, *ptr = (char*)cmd_buf;
  uint32_t addr = strtoul(cmd_buf + 1, &ptr, 16);
  uint32_t len = strtoul(ptr + 1, &ptr, 16);
  for (unsigned i = 0; i < len; ++i) {
    memcpy(buf, ptr + 1 + i * 2, 2);
    cfg.write_mem(addr + i, strtoul(buf, NULL, 16));
  }
  send_gdb(sockfd, "OK");
}

// set breakpoint address via callback
static void set_break(sock_t sockfd, const char *cmd_buf, bool active) {
  char *ptr = (char*)cmd_buf;
  uint32_t addr = strtoul(cmd_buf + 3, &ptr, 16);
  uint32_t len = strtoul(ptr + 1, NULL, 16);
  switch (cmd_buf[1]) {
    case '0': case '1':
      if (active != 0) cfg.set_break(addr);
      if (active == 0) cfg.clr_break(addr);
      return send_gdb(sockfd, "OK");
    case '2': case '3': case '4':
      if (active != 0) cfg.set_watch(addr, len, cmd_buf[1] - '1');
      if (active == 0) cfg.clr_watch(addr, len, cmd_buf[1] - '1');
      return send_gdb(sockfd, "OK");
    default:
      return send_gdb(sockfd, "E01");
  }
}

// setup socket for communicating with gdb
void Debugger::init(uint32_t port, DbgConfig conf) {
  sockaddr_in server_addr = {}, client_addr = {};
  socklen_t addr_size = sizeof(sockaddr_in);
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  server_addr.sin_addr.s_addr = INADDR_ANY;

#ifdef _WIN32
  WSADATA wsa_data;
  if (WSAStartup(0x0202, &wsa_data) != 0) exit(1);
  atexit(cleanup_socket);
#endif

  // open, config, and bind socket to port
  sock_t tmpsock = socket(AF_INET, SOCK_STREAM, 0);
  if (tmpsock < 0) printf("Failed to create gdb socket\n"), exit(1);
  int reuse = setsockopt(tmpsock, SOL_SOCKET, SO_REUSEADDR, "\x01\x00\x00\x00", 4);
  if (reuse != 0) printf("Failed to set gdb socket option\n"), exit(1);
  int bound = bind(tmpsock, (sockaddr*)(&server_addr), addr_size);
  if (bound != 0) printf("Failed to bind gdb socket\n"), exit(1);
  listen(tmpsock, 5), printf("Listening for gdb on port %d\n", port);

  // accept socket connection from gdb server
  gdb_sock = accept(tmpsock, (sockaddr*)(&client_addr), &addr_size);
  if (gdb_sock < 0) printf("Failed to accept gdb client\n"), exit(1);
  if (tmpsock != -1) shutdown(tmpsock, 2);
  cfg = conf, printf("Connected to gdb client\n");
}

// check socket for ctrl-c from gdb
bool Debugger::poll() {
  if (!gdb_sock) return 0;
#ifdef _WIN32
  WSAPOLLFD pfd = {gdb_sock, POLLIN, 0};
  return WSAPoll(&pfd, 1, 0) == 1;
#else
  pollfd pfd = {gdb_sock, POLLIN, 0};
  return poll(&pfd, 1, 0) == 1;
#endif
}

// process gdb remote protocol commands
bool Debugger::update(uint32_t step) {
  char buf[20] = "S05";
  if (step > 1) sprintf(buf + 3, "watch:%x;", ~step);
  send_gdb(gdb_sock, buf);
  while (true) {
    memset(cmd_buf, 0, buf_size);
    recv_gdb(gdb_sock, cmd_buf);
    switch (cmd_buf[0]) {
      case 0x0: break;
      case '?': send_gdb(gdb_sock, "S05"); break;
      case 'c': case 'C': return false;
      case 's': return true;
      case 'g': read_regs(gdb_sock); break;
      case 'G': write_regs(gdb_sock, cmd_buf); break;
      case 'p': read_reg(gdb_sock, cmd_buf); break;
      case 'P': write_reg(gdb_sock, cmd_buf); break;
      case 'H': send_gdb(gdb_sock, "OK"); break;
      case 'k': printf("Killed by gdb\n"); exit(0);
      case 'm': read_mem(gdb_sock, cmd_buf); break;
      case 'M': write_mem(gdb_sock, cmd_buf); break;
      case 'q': query(gdb_sock, cmd_buf); break;
      case 'z': set_break(gdb_sock, cmd_buf, 0); break;
      case 'Z': set_break(gdb_sock, cmd_buf, 1); break;
      default: send_gdb(gdb_sock, ""); break;
    }
  }
}

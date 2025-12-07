#ifndef LWIP_SOCKETS_H
#define LWIP_SOCKETS_H

// On native platform, use standard POSIX socket functions
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

// IPPROTO_UDP is defined in netinet/in.h on both Linux and macOS

#endif // LWIP_SOCKETS_H

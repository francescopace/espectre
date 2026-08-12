/*
 * ESPectre - Mock sockets.h
 *
 * Host-side mock of sockets.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
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

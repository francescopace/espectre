#ifndef LWIP_INET_H
#define LWIP_INET_H

// On native platform, use standard network headers
#include <stdint.h>
#include <arpa/inet.h>
#include <netinet/in.h>

// IP address macros for formatting (ESP-IDF style)
#define IPSTR "%d.%d.%d.%d"
#define IP2STR(ipaddr) ((uint8_t *)&(ipaddr)->addr)[0], \
                       ((uint8_t *)&(ipaddr)->addr)[1], \
                       ((uint8_t *)&(ipaddr)->addr)[2], \
                       ((uint8_t *)&(ipaddr)->addr)[3]

// htons, htonl, ntohs, ntohl are provided by arpa/inet.h

#endif // LWIP_INET_H

/*
 * Mock for lwip/ip_addr.h
 * 
 * Provides IP address types and macros for native testing
 */

#ifndef LWIP_IP_ADDR_H
#define LWIP_IP_ADDR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// IP address types
typedef struct {
    uint32_t addr;
} ip4_addr_t;

typedef struct {
    uint8_t addr[16];
} ip6_addr_t;

typedef struct {
    union {
        ip4_addr_t ip4;
        ip6_addr_t ip6;
    } u_addr;
    uint8_t type;
} ip_addr_t;

// IP address type constants
#define IPADDR_TYPE_V4 0
#define IPADDR_TYPE_V6 6

// Macros for extracting octets from ip4_addr_t
#define ip4_addr1(addr) (((addr)->addr >> 0) & 0xFF)
#define ip4_addr2(addr) (((addr)->addr >> 8) & 0xFF)
#define ip4_addr3(addr) (((addr)->addr >> 16) & 0xFF)
#define ip4_addr4(addr) (((addr)->addr >> 24) & 0xFF)

// Macro to set IP4 address
#define IP_ADDR4(ipaddr, a, b, c, d) do { \
    (ipaddr)->u_addr.ip4.addr = ((uint32_t)(a) << 0) | \
                                 ((uint32_t)(b) << 8) | \
                                 ((uint32_t)(c) << 16) | \
                                 ((uint32_t)(d) << 24); \
    (ipaddr)->type = IPADDR_TYPE_V4; \
} while(0)

#ifdef __cplusplus
}
#endif

#endif // LWIP_IP_ADDR_H


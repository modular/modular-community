# Note: Additional tests for functions that require complex setups or network access
# (like bind, listen, accept, connect, send, recv, select with sockets, etc.) are
# not included here due to their complexity and potential side effects.

from memory import UnsafePointer
from utils import StringRef
from testing import assert_equal, assert_not_equal, assert_raises
from pathlib import Path, _dir_of_current_file

from libc import (
    AF_INET,
    AI_PASSIVE,
    FD_STDOUT,
    INET_ADDRSTRLEN,
    SOCK_DGRAM,
    SOCK_STREAM,
    addrinfo,
    c_char,
    c_int,
    c_uint,
    c_ushort,
    c_void,
    cftob,
    close,
    external_call,
    gai_strerror,
    getaddrinfo,
    htonl,
    htons,
    in_addr,
    inet_addr,
    inet_ntoa,
    inet_ntop,
    inet_pton,
    ntohl,
    ntohs,
    printf,
    sockaddr,
    socket,
    strlen,
    write,
)


fn test_getaddrinfo() raises:
    var host = String("127.0.0.1")
    var host_ptr = host.unsafe_cstr_ptr()
    var my_addrinfo = addrinfo()
    var servinfo = UnsafePointer[addrinfo]().alloc(1)
    servinfo.init_pointee_move(my_addrinfo)

    var hints = addrinfo()
    hints.ai_family = AF_INET
    hints.ai_socktype = SOCK_STREAM
    hints.ai_flags = AI_PASSIVE

    var error = getaddrinfo(
        host_ptr,
        UnsafePointer[c_char](),
        UnsafePointer.address_of(hints),
        UnsafePointer.address_of(servinfo),
    )
    assert_equal(error, 0, "getaddrinfo failed")

    var addrinfo = servinfo[]

    var ai_addr = addrinfo.ai_addr
    var null_ptr = UnsafePointer[sockaddr]()
    assert_not_equal(ai_addr, null_ptr, "ai_addr is null")


fn test_strlen() raises:
    var s = String("Hello, world!")
    var ptr = s.unsafe_cstr_ptr()
    var length = strlen(ptr)
    assert_equal(length, len(s), "strlen returned incorrect length")


fn test_htonl_ntohl() raises:
    var host_value: c_uint = 0x12345678
    var net_value = htonl(host_value)
    var host_value_converted_back = ntohl(net_value)
    assert_equal(
        host_value_converted_back, host_value, "htonl/ntohl failed to convert correctly"
    )


fn test_cftob() raises:
    assert_equal(cftob(1), True, "cftob failed for value 1")
    assert_equal(cftob(0), False, "cftob failed for value 0")
    assert_equal(cftob(-1), False, "cftob failed for value -1")
    assert_equal(cftob(100), True, "cftob failed for value 100")


fn test_inet_pton() raises:
    var ip_str = String("127.0.0.1")
    var result = inet_pton(AF_INET, ip_str)
    assert_equal(result, 16777343, "inet_pton failed to convert IP address")


fn test_htons_ntohs() raises:
    var host_value: c_ushort = 0x1234
    var net_value = htons(host_value)
    var host_value_converted_back = ntohs(net_value)
    assert_equal(
        host_value_converted_back, host_value, "htons/ntohs failed to convert correctly"
    )


fn test_inet_ntop_pton() raises:
    var ip_str = String("127.0.0.1")
    var ip_str_ptr = ip_str.unsafe_cstr_ptr()
    var addr = UnsafePointer[c_void]().alloc(4)  # IPv4 address is 4 bytes
    var ret = inet_pton(AF_INET, ip_str_ptr, addr)
    assert_equal(ret, 1, "inet_pton failed to convert IP address")

    var buffer = UnsafePointer[c_char]().alloc(INET_ADDRSTRLEN)
    var result = inet_ntop(AF_INET, addr, buffer, INET_ADDRSTRLEN)
    var converted_ip_str = String(buffer)
    assert_equal(
        converted_ip_str, ip_str, "inet_ntop failed to convert IP address back"
    )


fn test_inet_addr_ntoa() raises:
    var ip_str = String("127.0.0.1")
    var ip_str_ptr = ip_str.unsafe_cstr_ptr()
    var addr = inet_addr(ip_str_ptr)
    assert_not_equal(addr, -1, "inet_addr failed to convert IP address")

    var in_addr_struct = in_addr(1000)
    in_addr_struct.s_addr = addr

    var addr_str_ptr = inet_ntoa(in_addr_struct)
    var addr_str = StringRef(ptr=addr_str_ptr)
    assert_equal(addr_str, ip_str, "inet_ntoa failed to convert address back")


fn test_socket_close() raises:
    var fd = socket(AF_INET, SOCK_STREAM, 0)
    assert_not_equal(fd, -1, "socket failed to create socket")
    var close_result = close(fd)
    assert_equal(close_result, 0, "close failed to close socket")


fn test_printf() raises:
    var format_str = String("Hello, %s!\n")
    var format_ptr = format_str.unsafe_cstr_ptr()
    var name = String("world")
    var name_ptr = name.unsafe_cstr_ptr()
    var result = printf(format_ptr, name_ptr)
    assert_equal(result, len("Hello, world!\n"), "printf failed to print correctly")


# TODO: This function cannot be executed in tests because of this error:
# failed to legalize operation 'pop.external_call' that was explicitly marked illegal
# it works if you run it in any Mojo file, though.
# fn test_write() raises:
#     var msg = String("Hello, world!\n")
#     var msg_ptr = msg.unsafe_cstr_ptr()
#     var bytes_written = write(FD_STDOUT, msg_ptr.bitcast[c_void](), len(msg))
#     assert_equal(bytes_written, len(msg), "write failed to write to stdout")

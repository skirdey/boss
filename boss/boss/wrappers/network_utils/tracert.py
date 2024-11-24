import socket
import time


def traceroute(dest_name, max_hops=30, timeout=2.0, port=33434):
    """
    Performs a traceroute to the destination host.

    :param dest_name: The destination hostname or IP address.
    :param max_hops: Maximum number of hops.
    :param timeout: Timeout in seconds for each probe.
    :param port: Destination port number.
    :return: List of hops with (hop number, IP address, RTT in ms).
    """
    dest_addr = socket.gethostbyname(dest_name)
    print(f"Traceroute to {dest_name} ({dest_addr}), {max_hops} hops max.")

    traceroute_result = []
    curr_addr = None

    for ttl in range(1, max_hops + 1):
        # Create a UDP socket
        recv_socket = socket.socket(
            socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP
        )
        send_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        send_socket.setsockopt(socket.SOL_IP, socket.IP_TTL, ttl)
        recv_socket.settimeout(timeout)
        recv_socket.bind(("", port))

        start_time = time.time()
        send_socket.sendto(b"", (dest_name, port))
        curr_addr = None
        curr_name = None

        try:
            data, curr_addr = recv_socket.recvfrom(512)
            rtt = (time.time() - start_time) * 1000  # in ms
            curr_hop_addr = curr_addr[0]  # Extract IP from address tuple
            try:
                curr_name = socket.gethostbyaddr(curr_hop_addr)[0]
            except socket.error:
                curr_name = curr_hop_addr
        except socket.timeout:
            curr_hop_addr = "*"
            rtt = None
        finally:
            send_socket.close()
            recv_socket.close()

        if curr_hop_addr:
            hop_info = (ttl, curr_hop_addr, f"{rtt:.2f} ms" if rtt else "*")
            traceroute_result.append(hop_info)
            print(f"{ttl}\t{curr_hop_addr}\t{hop_info[2]}")
        else:
            hop_info = (ttl, "*", "*")
            traceroute_result.append(hop_info)
            print(f"{ttl}\t*\t*")

        # Only break if we've reached the destination
        if curr_hop_addr == dest_addr:
            break

    return traceroute_result

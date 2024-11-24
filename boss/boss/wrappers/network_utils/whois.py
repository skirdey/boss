import socket


def whois_lookup(domain):
    """
    Performs a WHOIS lookup for the given domain.

    :param domain: The domain to look up.
    :return: The WHOIS information as a string.
    """
    # Determine the WHOIS server based on the TLD
    tld = domain.split(".")[-1]
    if tld in ["com", "net", "org", "info", "biz"]:
        server = "whois.verisign-grs.com"
    elif tld == "io":
        server = "whois.nic.io"
    else:
        # For simplicity, using whois.iana.org for unknown TLDs
        server = "whois.iana.org"

    port = 43
    query = domain + "\r\n"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((server, port))
        sock.sendall(query.encode())
        response = b""
        while True:
            data = sock.recv(4096)
            if not data:
                break
            response += data
        sock.close()
        return response.decode()
    except Exception as e:
        return f"Error performing WHOIS lookup: {e}"

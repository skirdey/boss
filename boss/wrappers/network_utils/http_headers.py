import http.client
from urllib.parse import urlparse


def fetch_http_headers(url):
    """
    Fetches HTTP headers from the specified URL.

    :param url: The target URL (e.g., 'https://www.google.com').
    :return: Dictionary of HTTP headers.
    """
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    host = parsed_url.hostname
    path = parsed_url.path if parsed_url.path else "/"

    if scheme == "http":
        port = 80
        conn = http.client.HTTPConnection(host, port, timeout=10)
    elif scheme == "https":
        port = 443
        conn = http.client.HTTPSConnection(host, port, timeout=10)
    else:
        print(f"Unsupported URL scheme: {scheme}")
        return {}

    try:
        conn.request("HEAD", path)
        response = conn.getresponse()
        headers = dict(response.getheaders())
        conn.close()
        return headers
    except Exception as e:
        print(f"Error fetching HTTP headers: {e}")
        return {}


# Example usage:
if __name__ == "__main__":
    url = "https://developers.inflection.ai"
    headers = fetch_http_headers(url)
    print(f"HTTP Headers for {url}:")
    for key, value in headers.items():
        print(f"{key}: {value}")

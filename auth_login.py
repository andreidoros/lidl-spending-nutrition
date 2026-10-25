#!/usr/bin/env python3
"""
Lidl Plus Authentication CLI (Zero-Friction OAuth2 PKCE)
=========================================================
Automatically intercepts the OAuth2 callback code from Lidl's authentication server
using an interactive browser window, eliminating the need for manual DevTools inspection.
"""

import io
import os
import re
import sys
import urllib.parse
import webbrowser
from pathlib import Path
from lidlplus_api import LidlPlusApi

# Ensure UTF-8 console output on Windows terminals
if sys.platform == "win32":
    try:
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def extract_code(raw_input_str: str) -> str:
    """Extract authorization code from a redirect URL or raw code string."""
    cleaned = raw_input_str.strip()
    match = re.search(r"[?&]code=([A-Za-z0-9._-]+)", cleaned)
    if match:
        return urllib.parse.unquote(match.group(1))
    return cleaned


def save_token_to_env(token: str, country: str) -> None:
    """Atomically write or update LIDL_REFRESH_TOKEN and LIDL_COUNTRY in .env."""
    env_path = Path(".env")
    tmp_path = Path(".env.tmp")

    lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    updated_token = False
    updated_country = False
    new_lines = []

    for line in lines:
        if line.strip().startswith("LIDL_REFRESH_TOKEN="):
            new_lines.append(f"LIDL_REFRESH_TOKEN={token}\n")
            updated_token = True
        elif line.strip().startswith("LIDL_COUNTRY="):
            new_lines.append(f"LIDL_COUNTRY={country.lower()}\n")
            updated_country = True
        else:
            new_lines.append(line)

    if not updated_token:
        new_lines.append(f"LIDL_REFRESH_TOKEN={token}\n")
    if not updated_country:
        new_lines.append(f"LIDL_COUNTRY={country.lower()}\n")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    os.replace(tmp_path, env_path)


def capture_code_with_playwright(login_url: str) -> str | None:
    """Open an interactive browser window and automatically capture the OAuth redirect code."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    auth_code = None

    def check_for_code(text: str) -> str | None:
        if not text:
            return None
        # Strict matching: MUST be the callback redirect URL or location header
        # Avoid matching 'country_code=', 'code_challenge=', etc.
        if "com.lidlplus.app" in text or "connect/authorize/callback" in text:
            match = re.search(r"[?&]code=([A-Za-z0-9._-]+)", text)
            if match:
                candidate = match.group(1)
                # Filter out accidental parameter values
                if len(candidate) > 5 and candidate != "code":
                    return urllib.parse.unquote(candidate)
        return None

    try:
        print("[*] Launching automated login window via Playwright...")
        with sync_playwright() as p:
            # Launch Chromium in visible mode so the user can interact
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()

            def on_response(response):
                nonlocal auth_code
                loc = response.headers.get("location", "")
                c = check_for_code(loc) or check_for_code(response.url)
                if c and not auth_code:
                    auth_code = c

            def on_request(request):
                nonlocal auth_code
                c = check_for_code(request.url)
                if c and not auth_code:
                    auth_code = c

            page.on("response", on_response)
            page.on("request", on_request)

            page.goto(login_url)

            print("[+] Login window opened.")
            print("[*] Please sign in with your phone/email and password in the opened window.")
            print("[*] If prompted for an SMS verification code, enter it in the browser.")
            print("[*] The window will CLOSE AUTOMATICALLY the moment login succeeds!\n")

            while not auth_code:
                if page.is_closed():
                    break
                page.wait_for_timeout(500)

            if auth_code:
                print(f"[+] Successfully captured OAuth authorization code (length: {len(auth_code)}).")

            try:
                browser.close()
            except Exception:
                pass

            return auth_code
    except Exception as e:
        print(f"[!] Playwright interception encountered an issue: {e}")
        return None


def main() -> None:
    print("=" * 70)
    print("         LIDL PLUS AUTHENTICATION WIZARD (OAuth2 PKCE)")
    print("=" * 70)
    print("This wizard will help you authenticate securely with your Lidl Plus account.")
    print("Your password will NEVER be stored or handled by this script.")
    print("-" * 70)

    # 1. Country Selection
    default_country = os.getenv("LIDL_COUNTRY", "ro").strip().lower() or "ro"
    country_prompt = input(f"Enter your account country code (e.g. ro, de, fr, es, it, pl) [{default_country}]: ").strip().lower()
    country = country_prompt if country_prompt else default_country

    print(f"\n[*] Initializing OAuth2 PKCE client for country: {country.upper()}...")
    try:
        api = LidlPlusApi(language=country, country=country.upper())
        login_url = api._register_link
    except Exception as e:
        print(f"[!] Error generating login URL: {e}")
        sys.exit(1)

    # 2. Try Automatic Interception via Playwright
    auth_code = capture_code_with_playwright(login_url)

    # 3. Fallback: Manual Browser + Paste
    if not auth_code:
        print("\n" + "=" * 70)
        print("MANUAL FALLBACK: Log in via your default browser")
        print("=" * 70)
        print(f"Opening: {login_url}\n")
        try:
            webbrowser.open(login_url)
        except Exception:
            pass

        print("1. Log in with your phone and password.")
        print("2. When completed, copy the callback URL and paste it below.")
        user_input = ""
        while not user_input:
            user_input = input("Paste the redirect URL (or authorization code) here: ").strip()
        auth_code = extract_code(user_input)

    # 4. Exchange Code for Refresh Token
    print(f"\n[*] Exchanging authorization code for secure refresh token...")
    try:
        api._authorization_code(auth_code)
    except KeyError as e:
        print(f"\n[!] Token exchange failed: Lidl authentication server did not return tokens (KeyError: {e}).")
        print("    Hints:")
        print("    - The authorization code may have expired or was invalid.")
        print("    - Please re-run 'run.bat' to try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Failed to exchange code for token: {e}")
        sys.exit(1)

    refresh_token = getattr(api, "_refresh_token", None) or getattr(api, "refresh_token", None)
    if not refresh_token:
        print("\n[!] Failed: No refresh token returned by Lidl authentication server.")
        sys.exit(1)

    # 5. Save to .env
    save_token_to_env(refresh_token, country)
    print("\n" + "=" * 70)
    print("[+] SUCCESS! Lidl Plus authentication token saved to '.env'.")
    print("=" * 70)
    print("You can now run:")
    print("  run.bat                      (Windows 1-Click launcher)")
    print("  python sync_incremental.py   (PowerShell / Terminal)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

"""KFPClientManager: authenticate against a Kubeflow Pipelines instance fronted
by Dex (with or without oauth2-proxy in front) and return a usable `kfp.Client`.
"""

import re
import urllib3
from urllib.parse import urlencode, urlsplit

import kfp
import requests


class KFPClientManager:
    def __init__(
        self,
        api_url: str,
        dex_username: str,
        dex_password: str,
        dex_auth_type: str = "local",
        skip_tls_verify: bool = True,
        namespace: str = "csgo2",
    ):
        if dex_auth_type not in ("ldap", "local"):
            raise ValueError(
                f"Invalid dex_auth_type '{dex_auth_type}', must be 'ldap' or 'local'"
            )
        self._api_url = api_url.rstrip("/")
        self._dex_username = dex_username
        self._dex_password = dex_password
        self._dex_auth_type = dex_auth_type
        self._skip_tls_verify = skip_tls_verify
        self._namespace = namespace

        if self._skip_tls_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _get_session_cookies(self) -> str:
        s = requests.Session()
        s.verify = not self._skip_tls_verify

        # Step 1: hit the protected URL. We expect redirects through the
        # auth proxy (oauth2-proxy / authservice) and into Dex.
        resp = s.get(self._api_url, allow_redirects=True)

        # Some clusters return 403 from oauth2-proxy until we explicitly start
        # the oauth2 flow.
        if resp.status_code == 403:
            url_obj = urlsplit(resp.url)
            url_obj = url_obj._replace(
                path=re.sub(r"/$", "", url_obj.path) + "/oauth2/start",
                query=urlencode({"rd": "/"}),
            )
            resp = s.get(url_obj.geturl(), allow_redirects=True)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Initial GET failed: HTTP {resp.status_code} at {resp.url}\n"
                f"Body (truncated): {resp.text[:300]}"
            )

        # Step 2: we should now be on a Dex page. If we're at /dex/auth?req=...
        # (the auth-method selector), follow into the local/ldap variant first.
        url_obj = urlsplit(resp.url)
        path = url_obj.path
        if (
            "/dex/auth" in path
            and not path.rstrip("/").endswith(f"/{self._dex_auth_type}")
            and f"/auth/{self._dex_auth_type}" not in path
        ):
            new_path = re.sub(r"/dex/auth/?", f"/dex/auth/{self._dex_auth_type}/", path)
            new_path = new_path.replace("//", "/")
            target = url_obj._replace(path=new_path.rstrip("/")).geturl()
            resp = s.get(target, allow_redirects=True)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Could not reach Dex {self._dex_auth_type} login: "
                    f"HTTP {resp.status_code} at {resp.url}"
                )

        # Step 3: POST credentials. The form action is the current URL.
        login_url = resp.url
        resp = s.post(
            login_url,
            data={"login": self._dex_username, "password": self._dex_password},
            allow_redirects=True,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Login POST failed: HTTP {resp.status_code} at {resp.url}\n"
                f"Body (truncated): {resp.text[:300]}"
            )

        if len(s.cookies) == 0:
            raise RuntimeError(
                "Login appeared to succeed but no cookies were set. "
                "Verify Dex credentials and auth_type."
            )

        cookie_names = sorted({c.name for c in s.cookies})
        print(f"[KFPClientManager] auth ok, cookies: {cookie_names}")

        return "; ".join(f"{c.name}={c.value}" for c in s.cookies)

    def create_kfp_client(self) -> kfp.Client:
        session_cookies = self._get_session_cookies()
        return kfp.Client(
            host=f"{self._api_url}/pipeline",
            cookies=session_cookies,
            verify_ssl=not self._skip_tls_verify,
            namespace=self._namespace,
        )

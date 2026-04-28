from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient


class _DummyPipeline:
    def __init__(self, client):
        self.client = client

    def run(self, **kwargs):
        return {"audit_verdict": "ok"}


class _DummyClient:
    def healthcheck(self):
        return True, "ok"

    def chat(self, **kwargs):
        return "followup-ok"

    def metadata(self):
        return {"model": "test", "client": "dummy", "base_url": "http://dummy"}


class AuthPrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_file = tempfile.NamedTemporaryFile(prefix="bbe-auth-", suffix=".db", delete=False)
        cls.db_file.close()

        os.environ["SQLITE_DB_PATH"] = cls.db_file.name
        os.environ["AUTH_JWT_SECRET"] = "test-secret-123456789"
        os.environ["AUTH_JWT_ALG"] = "HS256"
        os.environ["AUTH_TOKEN_EXPIRE_MINUTES"] = "60"
        os.environ["ENABLE_ENCRYPTION_UTILS"] = "true"
        os.environ["APP_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")

        from backend.main import app

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(cls.db_file.name)
        except FileNotFoundError:
            pass

    def test_signup_login_me_flow(self):
        signup = self.client.post("/api/auth/signup", json={"username": "userone", "email": "user@example.com", "password": "password123"})
        self.assertEqual(signup.status_code, 201)

        duplicate = self.client.post("/api/auth/signup", json={"username": "anothername", "email": "user@example.com", "password": "password123"})
        self.assertEqual(duplicate.status_code, 409)

        login = self.client.post("/api/auth/login", json={"identifier": "userone", "password": "password123"})
        self.assertEqual(login.status_code, 200)
        token = login.json()["access_token"]

        me = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["email"], "user@example.com")
        self.assertEqual(me.json()["username"], "userone")

        bad_login = self.client.post("/api/auth/login", json={"identifier": "user@example.com", "password": "wrongpass123"})
        self.assertEqual(bad_login.status_code, 401)

    def test_protected_routes_reject_without_token(self):
        explain_trial = self.client.post("/api/explain", headers={"X-Trial-Id": "trial-a"}, json={"question": "q", "model_answer": "a"})
        explain_blocked = self.client.post("/api/explain", headers={"X-Trial-Id": "trial-a"}, json={"question": "q", "model_answer": "a"})
        followup = self.client.post("/api/followup", json={"question": "q", "followup": "f"})
        self.assertEqual(explain_trial.status_code, 200)
        self.assertEqual(explain_blocked.status_code, 401)
        self.assertEqual(followup.status_code, 401)

    def test_protected_routes_accept_valid_token(self):
        self.client.post("/api/auth/signup", json={"username": "authok", "email": "authok@example.com", "password": "password123"})
        login = self.client.post("/api/auth/login", json={"identifier": "authok@example.com", "password": "password123"})
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        with patch("backend.main._build_client", return_value=_DummyClient()), patch("backend.main.ExplainerPipeline", _DummyPipeline):
            explain = self.client.post("/api/explain", headers=headers, json={"question": "q", "model_answer": "a"})
            self.assertEqual(explain.status_code, 200)

            followup = self.client.post(
                "/api/followup",
                headers=headers,
                json={"question": "q", "model_answer": "a", "context": "c", "followup": "f"},
            )
            self.assertEqual(followup.status_code, 200)
            self.assertEqual(followup.json()["reply"], "followup-ok")

    def test_followup_error_does_not_echo_sensitive_content(self):
        self.client.post("/api/auth/signup", json={"username": "erruser", "email": "err@example.com", "password": "password123"})
        login = self.client.post("/api/auth/login", json={"identifier": "erruser", "password": "password123"})
        token = login.json()["access_token"]

        class _FailingClient(_DummyClient):
            def chat(self, **kwargs):
                raise RuntimeError("sensitive-secret-from-provider")

        with patch("backend.main._build_client", return_value=_FailingClient()):
            response = self.client.post(
                "/api/followup",
                headers={"Authorization": f"Bearer {token}"},
                json={"question": "my private question", "model_answer": "private answer", "context": "private context", "followup": "private followup"},
            )
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json()["detail"], "The follow-up request failed.")


class EncryptionUtilTests(unittest.TestCase):
    def test_encryption_roundtrip(self):
        from backend.crypto_utils import FieldEncryptor

        key = Fernet.generate_key().decode("utf-8")
        encryptor = FieldEncryptor(key)
        encrypted = encryptor.encrypt_text("secret-text")
        self.assertNotEqual(encrypted, "secret-text")
        self.assertEqual(encryptor.decrypt_text(encrypted), "secret-text")

    def test_validate_fails_when_enabled_without_key(self):
        from backend import crypto_utils

        old_enabled = os.environ.get("ENABLE_ENCRYPTION_UTILS")
        old_key = os.environ.get("APP_ENCRYPTION_KEY")

        try:
            os.environ["ENABLE_ENCRYPTION_UTILS"] = "true"
            if "APP_ENCRYPTION_KEY" in os.environ:
                del os.environ["APP_ENCRYPTION_KEY"]
            with self.assertRaises(RuntimeError):
                crypto_utils.validate_encryption_setup()
        finally:
            if old_enabled is None:
                os.environ.pop("ENABLE_ENCRYPTION_UTILS", None)
            else:
                os.environ["ENABLE_ENCRYPTION_UTILS"] = old_enabled
            if old_key is None:
                os.environ.pop("APP_ENCRYPTION_KEY", None)
            else:
                os.environ["APP_ENCRYPTION_KEY"] = old_key

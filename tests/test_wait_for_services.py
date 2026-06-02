"""Tests for wait_for_services module."""

import socket
import urllib.error
import pytest
from unittest.mock import patch, MagicMock
from wait_for_services import (
    check_tcp,
    check_http_get,
    check_http_post,
    wait_for_service,
    SERVICES,
)


class TestCheckTcp:
    @patch("wait_for_services.socket.create_connection")
    def test_port_open(self, mock_conn):
        mock_sock = MagicMock()
        mock_conn.return_value = mock_sock
        assert check_tcp("localhost", 5432) is True
        mock_sock.close.assert_called_once()

    @patch("wait_for_services.socket.create_connection")
    def test_port_closed(self, mock_conn):
        mock_conn.side_effect = ConnectionRefusedError()
        assert check_tcp("localhost", 5432) is False

    @patch("wait_for_services.socket.create_connection")
    def test_timeout(self, mock_conn):
        mock_conn.side_effect = socket.timeout()
        assert check_tcp("localhost", 5432) is False


class TestCheckHttpGet:
    @patch("wait_for_services.urllib.request.urlopen")
    def test_responds_200(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        assert check_http_get("http://localhost:8080/v1/models") is True

    @patch("wait_for_services.urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        assert check_http_get("http://localhost:8080/v1/models") is False


class TestCheckHttpPost:
    @patch("wait_for_services.urllib.request.urlopen")
    def test_responds_200(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        assert check_http_post("http://localhost:8081/embed") is True

    @patch("wait_for_services.urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("refused")
        assert check_http_post("http://localhost:8081/embed") is False


class TestWaitForService:
    @patch("wait_for_services.check_tcp")
    def test_immediate_success(self, mock_check):
        mock_check.return_value = True
        config = {"host": "localhost", "port": 5433, "check": "tcp", "description": "PostgreSQL"}
        result = wait_for_service("postgres", config, timeout=5, interval=0.1)
        assert result is True

    @patch("wait_for_services.check_tcp")
    def test_timeout_failure(self, mock_check):
        mock_check.return_value = False
        config = {"host": "localhost", "port": 5433, "check": "tcp", "description": "PostgreSQL"}
        result = wait_for_service("postgres", config, timeout=1, interval=0.1)
        assert result is False


class TestServicesConfig:
    def test_all_services_have_required_fields(self):
        for name, config in SERVICES.items():
            assert "check" in config, f"{name} missing 'check'"
            assert "description" in config, f"{name} missing 'description'"
            assert config["check"] in ("tcp", "http_get", "http_post")

    def test_postgres_has_host_and_port(self):
        assert "host" in SERVICES["postgres"]
        assert "port" in SERVICES["postgres"]

    def test_embeddings_has_url(self):
        assert "url" in SERVICES["embeddings"]

    def test_llm_has_url(self):
        assert "url" in SERVICES["llm"]
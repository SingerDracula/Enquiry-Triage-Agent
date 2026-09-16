from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from enquiry_triage.models import json_schema
from enquiry_triage.providers import (
    GeminiGenerateContentProvider,
    OpenAICompatibleProvider,
    ProviderError,
    StructuredOutputMode,
    provider_from_spec,
)


class ProviderPayloadTests(unittest.TestCase):
    def test_strict_mode_sends_json_schema_with_server_strict_flag(self) -> None:
        provider = OpenAICompatibleProvider(
            model="strict-model",
            base_url="https://provider.example/v1",
            api_key="test-key",
        )
        payload = provider.build_payload(
            system_prompt="Return JSON.",
            email_text="Subject: test",
            response_schema=json_schema(),
        )

        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["messages"][0]["content"], "Return JSON.")

    def test_deepseek_mode_sends_json_object_and_includes_contract_in_prompt(self) -> None:
        provider = OpenAICompatibleProvider(
            model="deepseek-test",
            base_url="https://api.deepseek.com",
            api_key="test-key",
            structured_output_mode=StructuredOutputMode.JSON_OBJECT,
        )
        payload = provider.build_payload(
            system_prompt="Return JSON.",
            email_text="Subject: test",
            response_schema=json_schema(),
        )

        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("json mode requirements", payload["messages"][0]["content"].lower())
        self.assertIn("JSON Schema", payload["messages"][0]["content"])
        self.assertIn("confidence.methodology: exactly one concise sentence, 12-120 characters", payload["messages"][0]["content"])
        self.assertIn("do not treat anger alone as urgent", payload["messages"][0]["content"].lower())
        self.assertIn("priority must be urgent", payload["messages"][0]["content"].lower())
        self.assertIn("priority independently from case_type", payload["messages"][0]["content"])
        self.assertIn("address-change request missing postal details are all NORMAL", payload["messages"][0]["content"])
        self.assertIn("do not use low merely because information is missing", payload["messages"][0]["content"].lower())
        self.assertIn("claim-status request for future reference that says 'No rush' is LOW", payload["messages"][0]["content"])
        self.assertIn("This explicit deferral overrides the NORMAL default", payload["messages"][0]["content"])
        self.assertIn("estimated probability that case_type is correct", payload["messages"][0]["content"])
        self.assertIn("0.90-0.98 for an explicit, unambiguous case_type", payload["messages"][0]["content"])
        self.assertEqual(payload["max_tokens"], 800)

    def test_deepseek_spec_selects_json_object_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                '[providers.deepseek]\nmodel = "deepseek-test"\napi_key = "test-key"\n'
                'base_url = "https://api.deepseek.com"\n',
                encoding="utf-8",
            )
            provider = provider_from_spec("deepseek", config_path=config_path)

        self.assertEqual(provider.structured_output_mode, StructuredOutputMode.JSON_OBJECT)
        self.assertEqual(provider.base_url, "https://api.deepseek.com")

    def test_gpt_spec_selects_openai_strict_schema_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                '[providers.gpt]\nmodel = "test-model"\napi_key = "test-key"\n', encoding="utf-8"
            )
            provider = provider_from_spec("gpt", config_path=config_path)

        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertEqual(provider.base_url, "https://api.openai.com/v1")
        self.assertEqual(provider.structured_output_mode, StructuredOutputMode.JSON_SCHEMA_STRICT)

    def test_gemini_mode_sends_converted_json_schema(self) -> None:
        provider = GeminiGenerateContentProvider(
            model="gemini-test",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
        )
        payload = provider.build_payload(
            system_prompt="Return JSON.",
            email_text="Subject: test",
            response_schema=json_schema(),
        )

        config = payload["generationConfig"]
        self.assertEqual(config["responseMimeType"], "application/json")
        self.assertEqual(config["responseSchema"]["type"], "OBJECT")
        self.assertEqual(config["responseSchema"]["properties"]["confidence"]["type"], "OBJECT")
        self.assertIn(
            "PREMIUM_BILLING",
            config["responseSchema"]["properties"]["case_type"]["enum"],
        )

    def test_gemini_schema_omits_keywords_the_rest_api_rejects(self) -> None:
        provider = GeminiGenerateContentProvider(
            model="gemini-test",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
        )
        payload = provider.build_payload(
            system_prompt="Return JSON.",
            email_text="Subject: test",
            response_schema=json_schema(),
        )

        self.assertNotIn("additionalProperties", json.dumps(payload["generationConfig"]))
        self.assertNotIn("$defs", json.dumps(payload["generationConfig"]))
        self.assertNotIn("$ref", json.dumps(payload["generationConfig"]))

    def test_gemini_schema_rejects_open_objects(self) -> None:
        provider = GeminiGenerateContentProvider(
            model="gemini-test",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
        )

        with self.assertRaises(ProviderError):
            provider.build_payload(
                system_prompt="Return JSON.",
                email_text="Subject: test",
                response_schema={
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            )

    def test_gemini_spec_selects_gemini_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                '[providers.gemini]\nmodel = "gemini-test"\napi_key = "test-key"\n', encoding="utf-8"
            )
            provider = provider_from_spec("gemini", config_path=config_path)

        self.assertIsInstance(provider, GeminiGenerateContentProvider)
        self.assertEqual(provider.base_url, "https://generativelanguage.googleapis.com/v1beta")


class ProviderFailureDiagnosticsTests(unittest.TestCase):
    def _provider(self) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            model="strict-model",
            base_url="https://provider.example/v1",
            api_key="test-key",
        )

    def test_http_rejection_surfaces_status_body_and_latency(self) -> None:
        rejection = urllib.error.HTTPError(
            "https://provider.example/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"message":"This response_format type is unavailable now"}}'),
        )
        with patch("enquiry_triage.providers.urllib.request.urlopen", side_effect=rejection):
            with self.assertRaises(ProviderError) as caught:
                self._provider().generate(
                    system_prompt="Return JSON.",
                    email_text="Subject: test",
                    response_schema=json_schema(),
                )

        self.assertIn("HTTP 400", str(caught.exception))
        self.assertIn("response_format type is unavailable", str(caught.exception))
        self.assertGreaterEqual(caught.exception.latency_ms, 0.0)

    def test_error_body_excerpt_is_bounded_and_single_line(self) -> None:
        rejection = urllib.error.HTTPError(
            "https://provider.example/v1/chat/completions",
            500,
            "Server Error",
            {},
            io.BytesIO(("line one\nline two " + "x" * 1_000).encode("utf-8")),
        )
        with patch("enquiry_triage.providers.urllib.request.urlopen", side_effect=rejection):
            with self.assertRaises(ProviderError) as caught:
                self._provider().generate(
                    system_prompt="Return JSON.",
                    email_text="Subject: test",
                    response_schema=json_schema(),
                )

        detail = str(caught.exception)
        self.assertNotIn("\n", detail)
        self.assertLessEqual(len(detail), 400)

    def test_http_rejection_without_a_body_still_reports_the_status(self) -> None:
        rejection = urllib.error.HTTPError(
            "https://provider.example/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            None,
        )
        with patch("enquiry_triage.providers.urllib.request.urlopen", side_effect=rejection):
            with self.assertRaises(ProviderError) as caught:
                self._provider().generate(
                    system_prompt="Return JSON.",
                    email_text="Subject: test",
                    response_schema=json_schema(),
                )

        self.assertIn("HTTP 429", str(caught.exception))


class GeminiResponseTests(unittest.TestCase):
    def test_gemini_response_is_extracted_and_usage_is_recorded(self) -> None:
        provider = GeminiGenerateContentProvider(
            model="gemini test/model",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            input_usd_per_million=1.0,
            output_usd_per_million=2.0,
        )
        fake_response = io.BytesIO(
            b'{"candidates":[{"finishReason":"STOP","content":{"parts":['
            b'{"text":"{\\\"case_type\\\":\\\"OTHER\\\"}"}]}}],'
            b'"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20}}'
        )

        with patch("enquiry_triage.providers.urllib.request.urlopen", return_value=fake_response) as mocked:
            # BytesIO is a context manager, matching urllib's response contract.
            response = provider.generate(
                system_prompt="Return JSON.", email_text="Subject: test", response_schema=json_schema()
            )

        self.assertEqual(response.content, '{"case_type":"OTHER"}')
        self.assertEqual(response.usage.input_tokens, 10)
        self.assertEqual(response.usage.output_tokens, 20)
        self.assertEqual(response.usage.cost_usd, 0.00005)
        self.assertIn("gemini%20test%2Fmodel:generateContent", mocked.call_args.args[0].full_url)

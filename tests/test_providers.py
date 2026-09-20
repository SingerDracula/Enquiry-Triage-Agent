from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from enquiry_triage.jev import OpenRouterJevClassificationProvider
from enquiry_triage.models import TriageResult, json_schema
from enquiry_triage.providers import (
    GeminiGenerateContentProvider,
    OpenAICompatibleProvider,
    ProviderError,
    StructuredOutputMode,
    provider_from_spec,
)


class FixedDraftProvider:
    name = "draft-model"
    input_usd_per_million = 1.0
    output_usd_per_million = 2.0
    pricing_is_free = False

    def __init__(self, content: str | None = None) -> None:
        self.last_system_prompt: str | None = None
        self.content = content or json.dumps(
            {
                "case_type": "OTHER",
                "priority": "LOW",
                "summary": "Customer asks about a duplicate payment.",
                "draft_reply": "Hello, thank you for contacting us. A team member will review this. Kind regards.",
                "confidence": {
                    "score": 0.55,
                    "methodology": "The original generator selected a broad fallback category.",
                },
                "safety_status": "PENDING_REVIEW",
            }
        )

    def generate(self, *, system_prompt: str, email_text: str, response_schema: dict):
        from enquiry_triage.models import ProviderResponse, Usage

        self.last_system_prompt = system_prompt
        return ProviderResponse(
            model_name=self.name,
            content=self.content,
            latency_ms=10,
            usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.0002),
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
        self.assertEqual(provider.max_tokens, 2048)
        self.assertEqual(provider.thinking_mode, "disabled")

    def test_explicit_free_pricing_requires_zero_rates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                '[providers.deepseek]\nmodel = "deepseek-test"\napi_key = "test-key"\n'
                'base_url = "https://api.deepseek.com"\npricing_is_free = true\n',
                encoding="utf-8",
            )
            self.assertTrue(provider_from_spec("deepseek", config_path=config_path).pricing_is_free)
            with config_path.open("a", encoding="utf-8") as handle:
                handle.write("input_usd_per_million = 1\n")
            with self.assertRaises(ProviderError):
                provider_from_spec("deepseek", config_path=config_path)

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

    def test_glm_spec_uses_official_endpoint_and_json_object_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                '[providers.glm]\nmodel = "glm-4.7-flash"\napi_key = "test-key"\n',
                encoding="utf-8",
            )
            provider = provider_from_spec("glm", config_path=config_path)

        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertEqual(provider.base_url, "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(provider.structured_output_mode, StructuredOutputMode.JSON_OBJECT)
        payload = provider.build_payload(
            system_prompt="Return JSON.", email_text="Subject: test", response_schema=json_schema()
        )
        self.assertEqual(payload["model"], "glm-4.7-flash")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0.01)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertIn("JSON Schema", payload["messages"][0]["content"])

    def test_glm_response_uses_existing_chat_completions_parser(self) -> None:
        provider = OpenAICompatibleProvider(
            model="glm-4.7-flash",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key="test-key",
            structured_output_mode=StructuredOutputMode.JSON_OBJECT,
            temperature=0.01,
            thinking_mode="disabled",
        )
        fake_response = io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"case_type\\":\\"OTHER\\"}"}}],'
            b'"usage":{"prompt_tokens":10,"completion_tokens":20}}'
        )

        with patch("enquiry_triage.providers.urllib.request.urlopen", return_value=fake_response) as mocked:
            response = provider.generate(
                system_prompt="Return JSON.", email_text="Subject: test", response_schema=json_schema()
            )

        self.assertEqual(response.content, '{"case_type":"OTHER"}')
        self.assertEqual(response.usage.input_tokens, 10)
        self.assertEqual(response.usage.output_tokens, 20)
        self.assertEqual(
            mocked.call_args.args[0].full_url,
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        )

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


class OpenRouterJevClassificationTests(unittest.TestCase):
    def _provider(self, content: str | None = None) -> OpenRouterJevClassificationProvider:
        return OpenRouterJevClassificationProvider(
            base_provider=FixedDraftProvider(content),
            api_key="openrouter-test-key",
            jev_input_usd_per_million=0.04,
        )

    def test_payload_uses_two_typed_choice_questions(self) -> None:
        payload = self._provider().build_payload(email_text="Subject: Duplicate payment")

        self.assertEqual(payload["model"], "typesafe/jev-1.13")
        self.assertEqual(payload["questions"]["case_type"]["type"], "choice")
        self.assertEqual(
            set(payload["questions"]["case_type"]["criteria"]),
            {"POLICY_QUERY", "PREMIUM_BILLING", "ADDRESS_CHANGE", "CLAIM", "COMPLAINT", "OTHER"},
        )
        self.assertEqual(
            set(payload["questions"]["priority"]["criteria"]), {"URGENT", "NORMAL", "LOW"}
        )
        self.assertNotIn("providerOptions", payload)
        self.assertIn("untrusted customer data", payload["state"]["trust_boundary"])

    def test_jev_overrides_only_classification_and_its_confidence(self) -> None:
        response_body = {
            "model": "typesafe/jev-1.13-20260917",
            "answers": {
                "case_type": {
                    "type": "choice",
                    "choice": "PREMIUM_BILLING",
                    "probabilities": {"PREMIUM_BILLING": 0.97, "OTHER": 0.03},
                },
                "priority": {
                    "type": "choice",
                    "choice": "NORMAL",
                    "probabilities": {"URGENT": 0.01, "NORMAL": 0.94, "LOW": 0.05},
                },
            },
            "usage": {"input_tokens": 25, "output_tokens": 4, "cost": 0.000001},
        }
        fake_response = io.BytesIO(json.dumps(response_body).encode("utf-8"))
        provider = self._provider()

        with patch("enquiry_triage.jev.urllib.request.urlopen", return_value=fake_response) as mocked:
            response = provider.generate(
                system_prompt="Return JSON.",
                email_text="Subject: Duplicate payment",
                response_schema=json_schema(),
            )

        result = json.loads(response.content)
        validated = TriageResult.model_validate(result)
        self.assertEqual(result["case_type"], "PREMIUM_BILLING")
        self.assertEqual(result["priority"], "NORMAL")
        self.assertEqual(result["summary"], "Customer asks about a duplicate payment.")
        self.assertIn("A team member will review this", result["draft_reply"])
        self.assertEqual(result["safety_status"], "PENDING_REVIEW")
        self.assertEqual(result["confidence"]["score"], 0.97)
        self.assertIn("Jev choice probability", result["confidence"]["methodology"])
        self.assertEqual(validated.case_type.value, "PREMIUM_BILLING")
        self.assertEqual(response.model_name, "typesafe/jev-1.13+draft-model")
        self.assertEqual(response.usage.input_tokens, 125)
        self.assertEqual(response.usage.output_tokens, 54)
        self.assertAlmostEqual(response.usage.cost_usd, 0.000201)
        self.assertIn("case_type: PREMIUM_BILLING", provider.base_provider.last_system_prompt or "")
        self.assertIn("priority: NORMAL", provider.base_provider.last_system_prompt or "")
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(request.headers["Authorization"], "Bearer openrouter-test-key")

    def test_invalid_choice_fails_visibly_instead_of_falling_back(self) -> None:
        response_body = {
            "answers": {
                "case_type": {
                    "type": "choice",
                    "choice": "UNKNOWN",
                    "probabilities": {"UNKNOWN": 1.0},
                },
                "priority": {
                    "type": "choice",
                    "choice": "NORMAL",
                    "probabilities": {"NORMAL": 1.0},
                },
            }
        }
        fake_response = io.BytesIO(json.dumps(response_body).encode("utf-8"))

        with patch("enquiry_triage.jev.urllib.request.urlopen", return_value=fake_response):
            with self.assertRaisesRegex(ProviderError, "invalid 'case_type'"):
                self._provider().generate(
                    system_prompt="Return JSON.",
                    email_text="Subject: test",
                    response_schema=json_schema(),
                )

    def test_malformed_base_output_is_preserved_after_jev_classification(self) -> None:
        response_body = {
            "answers": {
                "case_type": {
                    "type": "choice",
                    "choice": "OTHER",
                    "probabilities": {"OTHER": 0.8},
                },
                "priority": {
                    "type": "choice",
                    "choice": "NORMAL",
                    "probabilities": {"NORMAL": 0.7},
                },
            },
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        fake_response = io.BytesIO(json.dumps(response_body).encode("utf-8"))
        with patch("enquiry_triage.jev.urllib.request.urlopen", return_value=fake_response) as mocked:
            response = self._provider("not-json").generate(
                system_prompt="Return JSON.",
                email_text="Subject: test",
                response_schema=json_schema(),
            )

        self.assertEqual(response.content, "not-json")
        self.assertEqual(response.model_name, "typesafe/jev-1.13+draft-model")
        self.assertEqual(response.usage.input_tokens, 105)
        mocked.assert_called_once()

    def test_provider_factory_wraps_the_configured_base_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                '[providers.deepseek]\nmodel = "deepseek-test"\napi_key = "deepseek-key"\n'
                'base_url = "https://api.deepseek.com"\n\n'
                '[providers.jev]\nmodel = "typesafe/jev-1.13"\napi_key = "openrouter-key"\n'
                'base_provider = "deepseek"\n'
                'input_usd_per_million = 0.04\n',
                encoding="utf-8",
            )
            provider = provider_from_spec("jev", config_path=config_path)

        self.assertIsInstance(provider, OpenRouterJevClassificationProvider)
        self.assertIsInstance(provider.base_provider, OpenAICompatibleProvider)
        self.assertEqual(provider.base_provider.model, "deepseek-test")
        self.assertEqual(provider.name, "typesafe/jev-1.13+deepseek-test")
        self.assertEqual(provider.input_usd_per_million, 0.04)
        self.assertFalse(provider.pricing_is_free)


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

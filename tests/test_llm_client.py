"""LLM HTTP 故障分类和重试，无需访问网络。"""

import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
import requests

from meetmin.summarizer.llm_client import OpenAICompatibleLLM


def response(status, *, headers=None, body=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result._content = json.dumps(body if body is not None else {"error": "test"}).encode()
    return result


@pytest.fixture
def fake_http(monkeypatch):
    results = []
    calls = []
    waits = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("meetmin.summarizer.llm_client.requests.post", post)
    monkeypatch.setattr("meetmin.summarizer.llm_client.time.sleep", waits.append)
    llm = OpenAICompatibleLLM({"base_url": "http://localhost/v1", "model": "test", "retries": 3})
    return llm, results, calls, waits


def success():
    return response(200, body={"choices": [{"message": {"content": "纪要"}}], "usage": {}})


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 418])
def test_non_retryable_http_errors_fail_immediately(fake_http, status):
    llm, results, calls, waits = fake_http
    results.append(response(status))
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as exc:
        llm.chat([{"role": "user", "content": "hello"}])
    assert len(calls) == 1 and not waits
    if status in (401, 403):
        assert "鉴权" in str(exc.value)
    if status in (400, 404, 422):
        assert "模型名" in str(exc.value)


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout("timeout"),
        requests.ConnectionError("disconnected"),
        response(500),
        response(503),
    ],
)
def test_transient_failures_retry_then_succeed(fake_http, error):
    llm, results, calls, waits = fake_http
    results.extend([error, success()])
    text, usage = llm.chat([{"role": "user", "content": "hello"}])
    assert text == "纪要" and usage.calls == 1
    assert len(calls) == 2 and waits == [1.0]


def test_exhaustion_has_no_final_wait(fake_http):
    llm, results, calls, waits = fake_http
    results.extend([response(503), requests.Timeout("timeout"), response(500)])
    with pytest.raises(RuntimeError, match="共 3 次.*HTTP 500"):
        llm.chat([])
    assert len(calls) == 3 and waits == [1.0, 2.0]


def test_single_attempt_never_sleeps_even_with_retry_after(fake_http):
    llm, results, calls, waits = fake_http
    llm.lcfg["retries"] = 1
    results.append(response(429, headers={"Retry-After": "10"}))
    with pytest.raises(RuntimeError, match="共 1 次.*HTTP 429"):
        llm.chat([])
    assert len(calls) == 1 and not waits


@pytest.mark.parametrize("header", ["5", "0005"])
def test_429_obeys_retry_after_seconds(fake_http, header):
    llm, results, calls, waits = fake_http
    results.extend([response(429, headers={"Retry-After": header}), success()])
    llm.chat([])
    assert len(calls) == 2 and waits == [5.0]


def test_429_obeys_retry_after_http_date(fake_http):
    llm, results, calls, waits = fake_http
    date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=20), usegmt=True)
    results.extend([response(429, headers={"Retry-After": date}), success()])
    llm.chat([])
    assert len(calls) == 2 and 18 <= waits[0] <= 20


@pytest.mark.parametrize("header", ["garbage", "-1", "1.5", ""])
def test_invalid_retry_after_uses_exponential_backoff(fake_http, header):
    llm, results, calls, waits = fake_http
    results.extend([response(429, headers={"Retry-After": header}), success()])
    llm.chat([])
    assert len(calls) == 2 and waits == [1.0]


def test_long_retry_after_does_not_retry_early(fake_http):
    llm, results, calls, waits = fake_http
    results.append(response(429, headers={"Retry-After": "999999999999999"}))
    with pytest.raises(RuntimeError, match="超过自动重试上限"):
        llm.chat([])
    assert len(calls) == 1 and not waits


@pytest.mark.parametrize(
    "error", [requests.exceptions.InvalidURL("bad"), requests.exceptions.SSLError("certificate")]
)
def test_other_request_errors_are_not_retried(fake_http, error):
    llm, results, calls, waits = fake_http
    results.append(error)
    with pytest.raises(RuntimeError):
        llm.chat([])
    assert len(calls) == 1 and not waits


def test_invalid_response_is_not_retried(fake_http):
    llm, results, calls, waits = fake_http
    results.append(response(200, body={"unexpected": True}))
    with pytest.raises(RuntimeError, match="格式无效"):
        llm.chat([])
    assert len(calls) == 1 and not waits


def test_malformed_json_is_not_retried(fake_http):
    llm, results, calls, waits = fake_http
    invalid = response(200)
    invalid._content = b"not json"
    results.append(invalid)
    with pytest.raises(RuntimeError, match="格式无效"):
        llm.chat([])
    assert len(calls) == 1 and not waits


def completion(text, reason, prompt=10, output=5):
    return response(
        200,
        body={
            "choices": [{"message": {"content": text}, "finish_reason": reason}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": output},
        },
    )


def test_truncated_response_retries_with_double_limit_and_counts_both_calls(fake_http, caplog):
    llm, results, calls, waits = fake_http
    results.extend([completion("截断", "length"), completion("完整纪要", "stop", 11, 8)])
    text, usage = llm.chat([], max_tokens=100)
    assert text == "完整纪要" and usage.finish_status == "complete"
    assert [call[1]["json"]["max_tokens"] for call in calls] == [100, 200]
    assert (usage.calls, usage.prompt_tokens, usage.completion_tokens) == (2, 21, 13)
    assert not waits and "截断" in caplog.text


def test_twice_truncated_returns_partial_with_warning(fake_http, caplog):
    llm, results, calls, _ = fake_http
    results.extend([completion("短", "length"), completion("仍不完整", "length")])
    text, usage = llm.chat([])
    assert text == "仍不完整" and usage.finish_status == "truncated"
    assert len(calls) == 2 and "仍被截断" in caplog.text


def test_retry_uses_effective_override_limit(fake_http):
    llm, results, calls, _ = fake_http
    llm.lcfg["extra_request"] = {"max_tokens": 150}
    results.extend([completion("短", "length"), completion("完整", "stop")])
    llm.chat([], max_tokens=100)
    assert [call[1]["json"]["max_tokens"] for call in calls] == [150, 300]


@pytest.mark.parametrize("reason", [None, "content_filter", "other"])
def test_unrecognized_finish_reason_is_unknown(fake_http, reason):
    llm, results, calls, _ = fake_http
    results.append(completion("纪要", reason))
    _, usage = llm.chat([])
    assert usage.finish_status == "unknown" and len(calls) == 1


@pytest.mark.parametrize("text", ["", "  \n", None])
def test_empty_output_fails_without_retry(fake_http, text):
    llm, results, calls, _ = fake_http
    results.append(completion(text, "stop"))
    with pytest.raises(RuntimeError, match="空结果"):
        llm.chat([])
    assert len(calls) == 1


def test_empty_truncated_output_retries_then_fails_if_still_empty(fake_http):
    llm, results, calls, _ = fake_http
    results.extend([completion("", "length"), completion(" ", "stop")])
    with pytest.raises(RuntimeError, match="空结果"):
        llm.chat([])
    assert len(calls) == 2


def test_truncated_retry_preserves_http_retry_policy(fake_http):
    llm, results, calls, waits = fake_http
    results.extend([completion("短", "length"), response(503), completion("成功", "stop")])
    text, usage = llm.chat([], max_tokens=100)
    assert text == "成功" and usage.calls == 2
    assert [call[1]["json"]["max_tokens"] for call in calls] == [100, 200, 200]
    assert waits == [1.0]

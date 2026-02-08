"""Tests for format conversion to Qwen2-VL ChatML."""

import json

import pytest
from PIL import Image

from src.data.format_converter import (
    batch_convert_to_chatml,
    format_for_sft_trainer,
    sample_to_chatml,
    sample_to_inference_chatml,
)


class TestSampleToChatML:
    def test_basic_conversion(self, sample_image):
        gt_json = json.dumps({"menu": [{"nm": "Coffee", "price": "3.00"}]})
        messages = sample_to_chatml(sample_image, gt_json)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_user_message_has_image_and_text(self, sample_image):
        messages = sample_to_chatml(sample_image, '{"test": true}')
        user_content = messages[0]["content"]

        assert len(user_content) == 2
        types = [c["type"] for c in user_content]
        assert "image" in types
        assert "text" in types

    def test_assistant_message_has_json(self, sample_image):
        gt_json = '{"total": {"total_price": "100"}}'
        messages = sample_to_chatml(sample_image, gt_json)

        assistant_content = messages[1]["content"]
        assert len(assistant_content) == 1
        assert assistant_content[0]["type"] == "text"
        assert "total_price" in assistant_content[0]["text"]

    def test_custom_instruction(self, sample_image):
        custom = "Extract receipt fields."
        messages = sample_to_chatml(sample_image, "{}", instruction=custom)

        text_content = [c for c in messages[0]["content"] if c["type"] == "text"][0]
        assert text_content["text"] == custom

    def test_with_image_path(self, sample_image):
        messages = sample_to_chatml(
            sample_image, "{}", image_path="/tmp/test.png"
        )
        image_content = [c for c in messages[0]["content"] if c["type"] == "image"][0]
        assert image_content["image"] == "/tmp/test.png"


class TestInferenceChatML:
    def test_inference_has_no_assistant(self, sample_image):
        messages = sample_to_inference_chatml(sample_image)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_inference_has_image_and_text(self, sample_image):
        messages = sample_to_inference_chatml(sample_image)
        content = messages[0]["content"]
        types = [c["type"] for c in content]
        assert "image" in types
        assert "text" in types


class TestBatchConvert:
    def test_batch_conversion(self, sample_image):
        samples = [
            {
                "id": "test_0",
                "image": sample_image,
                "ground_truth_json": '{"menu": []}',
                "metadata": {"dataset": "test", "num_fields": 0},
            },
            {
                "id": "test_1",
                "image": sample_image,
                "ground_truth_json": '{"total": {"total_price": "100"}}',
                "metadata": {"dataset": "test", "num_fields": 1},
            },
        ]

        converted = batch_convert_to_chatml(samples)
        assert len(converted) == 2
        assert all("messages" in c for c in converted)
        assert all("id" in c for c in converted)
        assert converted[0]["id"] == "test_0"

    def test_batch_formats_json(self, sample_image):
        samples = [
            {
                "id": "test",
                "image": sample_image,
                "ground_truth_json": '{"a":"b"}',
                "metadata": {},
            }
        ]
        converted = batch_convert_to_chatml(samples)
        assistant_text = converted[0]["messages"][1]["content"][0]["text"]
        # Should be pretty-formatted
        assert "\n" in assistant_text


class TestSFTFormat:
    def test_sft_format_has_messages(self, sample_image):
        samples = [
            {
                "id": "test",
                "image": sample_image,
                "ground_truth_json": '{"test": true}',
                "metadata": {},
            }
        ]
        formatted = format_for_sft_trainer(samples)
        assert len(formatted) == 1
        assert "messages" in formatted[0]
        assert len(formatted[0].keys()) == 1  # Only 'messages' key for SFTTrainer

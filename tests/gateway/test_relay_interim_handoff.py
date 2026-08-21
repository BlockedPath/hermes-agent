"""Regression for #13: RelayAdapter.send() must re-stamp the gateway-internal
``_interim_send`` marker before forwarding to ``send_for_platform()``.

Stream-final contract invariant 3: an interim send must never trigger
seal-interception on ANY egress door. ``send()`` pops the marker for its own
check; forwarding the already-stripped metadata made an interim send with an
explicit logical platform look turn-final to the second door — sealing a live
stream with commentary text and orphaning the true final into a duplicate.
"""
from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CapabilityDescriptor


class _RecordingTransport:
    """Relay transport that records outbound wire frames."""

    def __init__(self):
        self._identities = [("slack", "bot-1")]
        self.sent = []

    async def send_outbound(self, action, *, platform=None):
        self.sent.append((action, platform))
        return {"success": True, "message_id": "relay-message-1"}


def _make_relay(transport) -> RelayAdapter:
    from gateway.relay.descriptor import CapabilityDescriptor

    return RelayAdapter(
        PlatformConfig(enabled=True),
        CapabilityDescriptor(
            contract_version=1,
            platform="slack",
            label="Slack",
            max_message_length=4000,
            supports_draft_streaming=False,
            supports_edit=True,
            supports_threads=True,
            markdown_dialect="slack",
            len_unit="chars",
        ),
        transport=cast(Any, transport),
    )


@pytest.mark.asyncio
async def test_interim_marker_survives_send_to_send_for_platform_handoff():
    """Interim + explicit logical platform: NO seal-interception on either door."""
    transport = _RecordingTransport()
    relay = _make_relay(transport)

    # Simulate an open native stream: any final-looking send would match it.
    relay._match_open_draft = lambda chat_id, metadata: "chat-1:turn-1"
    relay._seal_open_draft = AsyncMock(
        side_effect=AssertionError("interim send must never trigger seal-interception")
    )

    result = await relay.send(
        "C123",
        "still working~",
        metadata={
            "_interim_send": True,
            "_relay_logical_platform": "slack",
        },
    )

    assert result.success is True
    # Delivered exactly once through the second egress door as a plain send.
    assert len(transport.sent) == 1
    action, platform = transport.sent[0]
    assert action["content"] == "still working~"
    assert platform == "slack"
    # Marker is gateway-internal — stripped before the wire.
    assert "_interim_send" not in (action.get("metadata") or {})
    # And the seal path was never taken.
    relay._seal_open_draft.assert_not_called()


@pytest.mark.asyncio
async def test_final_with_explicit_platform_still_seals():
    """Contract intact: a turn-final through the same lane MUST still seal."""
    transport = _RecordingTransport()
    relay = _make_relay(transport)
    sealed = {}

    async def _fake_seal(chat_id, content, metadata, *, draft_key=None):
        sealed["key"] = draft_key
        from gateway.platforms.base import SendResult

        return SendResult(success=True, message_id="sealed-1")

    relay._match_open_draft = lambda chat_id, metadata: "chat-1:turn-1"
    relay._seal_open_draft = _fake_seal

    result = await relay.send(
        "C123",
        "the real answer",
        metadata={"_relay_logical_platform": "slack"},
    )

    assert result.success is True
    assert sealed["key"] == "chat-1:turn-1"
    # Sealed — nothing hit the plain wire.
    assert transport.sent == []

"""Thin glue between vLLM's KV connector API and kvpilot's own decision logic.

NOT unit-tested here: it requires vLLM (and in practice a GPU) to import and
run at all. All actual decision, codec and bookkeeping logic lives in
kvpilot.live and kvpilot.policy, which ARE unit tested without vLLM or a GPU.
This module only translates vLLM's real KV cache lifecycle events into calls
on that tested logic, and translates the real outcomes back into
kvpilot.live.recorder.LiveRecorder calls.

Status: construction-only scaffold. The vLLM lifecycle methods
(update_state_after_alloc, wait_for_layer_load, wait_for_save,
request_finished, build_connector_meta, start_load_kv, ...) are deliberately
left unimplemented rather than faked -- writing them without a real installed
vLLM to run and check against risks code that looks finished but silently
does nothing. See docs/live-integration.md for what is and is not proven so
far, and the approved plan for the GPU phase that implements them for real.
"""

from __future__ import annotations

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
except ImportError as exc:  # pragma: no cover - exercised only on the GPU box
    raise ImportError(
        "kvpilot.integrations.vllm_connector requires vLLM. Install it on the "
        "rented GPU validation box (see docs/live-integration.md); it is not "
        "installed, and is not expected to be installed, on this dev machine."
    ) from exc

from ..live.recorder import LiveRecorder
from ..policy import CodecProfile, CostModel, HistoricalForecast


class KVPilotConnector(KVConnectorBase_V1):
    """Routes real vLLM pause/resume/cancel events through kvpilot's policy.

    ``profiles`` and ``model`` must come from real measurements
    (kvpilot.live.calibration.measure_codec), never from
    examples/assumed-costs.json -- that file is explicitly illustrative.
    """

    def __init__(self, *args, profiles: list[CodecProfile], model: CostModel,
                 recorder: LiveRecorder, policy: str = "adaptive", **kwargs):
        super().__init__(*args, **kwargs)
        self.profiles = profiles
        self.model = model
        self.recorder = recorder
        self.policy_name = policy
        self.history = HistoricalForecast()

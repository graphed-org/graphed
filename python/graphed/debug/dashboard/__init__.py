"""The M37 live execution dashboard: FINOS Perspective tables served over Tornado, fed by a
websocket network transport.

* :class:`Dashboard` — opt-in convenience: a local server + a loopback client.
* :class:`DashboardServer` — the Perspective/Tornado server (browser viewer + event ingest).
* :class:`NetworkMonitor` — a passive ``Monitor`` that streams a run's events to a server over a
  websocket (local or remote).

The heavy deps (``perspective-python``, ``tornado``, ``websocket-client``) are the ``dashboard``
extra and are imported lazily, so ``import graphed.debug`` works without them.
"""

from __future__ import annotations

from ._client import NetworkMonitor
from ._dashboard import Dashboard
from ._server import DashboardServer

__all__ = ["Dashboard", "DashboardServer", "NetworkMonitor"]

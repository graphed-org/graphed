# M65 frozen suite — graphed-debug slice

Traceability for the debug lane (m65). One section per sub-plan, appended; earlier rows are never
rewritten. Plans live in `graphed-workdir/lanes/debug/`.

## A1 — run control from the dashboard (`freeze-m65a1`, plan-A1.md)

The `control` field of `snapshot()` and `progress()` uses the `RunState` vocabulary: `"running"`
until a command is relayed, then the state that command requests (`"paused"`, `"running"`,
`"cancelled"`); `None` with control off. The page's `#ctl-state` shows the same words.

| Test | Contract | Fails |
|---|---|---|
| `test_m65a1_dashboard_control.py::test_post_control_reaches_a_listening_monitor` | A-5 POST → ingest → `RunControl.apply`; hello; A-6 `control`, `control_listeners`; reader threads are daemons | a monitor that never registers as a listener or never applies a command; a wrong `delivered` count; a non-daemon reader |
| `…::test_resume_reaches_a_monitor_idle_past_its_recv_timeout` | A-5 reader survives `recv()` timeouts | a reader that ends its connection on a timeout (a paused run never hears resume) |
| `…::test_control_route_refusals` | A-5 status codes 400 (unknown cmd, non-JSON, no cmd), 415, 404; A-6 `control is None` when off | a route that accepts a cross-site form post or an unknown command |
| `…::test_monitor_without_control_is_not_a_listener` | A-5 only a `hello` with `control: true` makes a listener | a server that relays commands to every ingest connection |
| `…::test_dashboard_control_is_opt_in` | A-6 `Dashboard.control`, `attach` and its `TypeError` | an attach that sets an attribute nothing reads, or overwrites `control` when off |
| `…::test_dashboard_steers_a_sequential_run` | A-2 on `SequentialRunner` driven from the dashboard; A-7 attach | a run that starts the next task while paused, or does not stop on cancel |
| `…::test_control_monitor_reconnects_while_idle` | A-5 reconnect while idle; `stop()` closes ingest connections | a monitor that reconnects only on its next event; a `stop()` that leaves ingest sockets open |
| `test_m65a1_control_browser.py::test_control_buttons_drive_the_run_control` | A-6 control bar, shown only with control on | buttons that do not reach the `RunControl`; a bar shown with control off; console or page errors |

`m65a1_control_helpers.py`: POST/poll helpers and a TCP relay that lets the reconnect test replace a
server behind one address without rebinding a port.

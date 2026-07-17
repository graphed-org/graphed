# Vendored Perspective (browser bundle)

- **`perspective-dashboard.js`** — a single, self-contained ES-module bundle of FINOS/perspective-dev
  **4.5.1**: the Perspective client + `<perspective-viewer>` + the Datagrid plugin, with the wasm
  **inlined** (no separate fetch). Its default export is the `perspective` client; importing it also
  registers the viewer element and the Datagrid plugin.
- **`themes.css`** — the Perspective viewer themes (`Pro Dark` is used).
- **`d3.min.js`** / **`d3-flamegraph.min.js`** / **`d3-flamegraph.css`** — vendored UMD builds (globals
  `d3` / `flamegraph`) for the **profile flamegraph** panel. The off-thread sampler
  (`graphed_debug._sampler`) ships a `{name,value,children}` inclusive count tree; the server merges
  trees across flushes/workers and serves the result at `/api/flamegraph.json`, which `index.html`
  polls and renders with d3-flame-graph. The profile is **not** a Perspective table.

These are **committed build artifacts** so the dashboard works offline and ships with the wheel
(`pip install graphed-debug[dashboard]` installs them — no Node needed by end users). They are **not**
hand-edited.

## Version pinning (important)

The bundle version **must match `perspective-python`** in `../../../../pyproject.toml`
(`==4.5.1`) — the client and server share a versioned wire protocol, and a mismatch fails only in the
browser (`cannot read properties of null` + red viewer overlays). The browser smoke test
(`tests/frozen/m37/test_dashboard_browser.py`) is the guard.

## Rebuilding / upgrading

The bundle is produced by the committed esbuild recipe in `graphed-debug/tools/dashboard-build/`.
To upgrade, bump the versions in `tools/dashboard-build/package.json` **and** the
`perspective-python` pin together, then:

```bash
cd graphed-debug/tools/dashboard-build
npm ci            # or `npm install`
node build.mjs    # rewrites perspective-dashboard.js + themes.css here
pytest ../../tests/frozen/m37/test_dashboard_browser.py   # confirm it renders
```

Only the Datagrid plugin is bundled: the d3fc plot plugins throw on empty/streaming data, so the
stats + tasks panels use Datagrid. The profile panel is a d3 flamegraph (separate vendored scripts
above), not a Perspective view.

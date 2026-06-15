# Vendored FINOS Perspective (browser client)

These files are the **browser** side of the dashboard, vendored so it works offline and can never
drift from the server. They are pinned to **`@finos/perspective` 3.8.0**, which **must match the
`perspective-python==3.8.0` pin** in `pyproject.toml` — the client and server share a versioned wire
protocol, so a mismatch produces `cannot read properties of null` errors and red viewer overlays in
the browser (this is exactly the bug this layout fixes).

Why FINOS 3.8.0 and not the newer perspective-dev 4.x: the 4.x fork ships bundler-oriented ESM
(bare imports, separate wasm) that cannot be vendored as browser-ready files without a Node build
step. FINOS 3.8.0's `/dist/cdn/` builds are browser-ready (self-registering plugins; wasm fetched
via `new URL("../wasm/<name>.wasm", import.meta.url)`), which is why the `cdn/` + `wasm/` sibling
layout below is load-bearing — don't flatten it.

```
cdn/perspective.js                    @finos/perspective@3.8.0/dist/cdn/perspective.js
cdn/perspective-viewer.js             @finos/perspective-viewer@3.8.0/dist/cdn/perspective-viewer.js
cdn/perspective-viewer-datagrid.js    @finos/perspective-viewer-datagrid@3.8.0/dist/cdn/...
wasm/perspective-server.wasm          @finos/perspective@3.8.0/dist/wasm/perspective-server.wasm
wasm/perspective-viewer.wasm          @finos/perspective-viewer@3.8.0/dist/wasm/perspective-viewer.wasm
css/themes.css                        @finos/perspective-viewer@3.8.0/dist/css/themes.css
```

Only the Datagrid plugin is vendored: the d3fc plot plugins throw on empty/streaming data, so the
dashboard uses Datagrid for all three panels.

## Re-vendoring (if the `perspective-python` pin changes to version `X`)

```bash
B=https://cdn.jsdelivr.net/npm
curl -sSL -o cdn/perspective.js                 "$B/@finos/perspective@X/dist/cdn/perspective.js"
curl -sSL -o cdn/perspective-viewer.js          "$B/@finos/perspective-viewer@X/dist/cdn/perspective-viewer.js"
curl -sSL -o cdn/perspective-viewer-datagrid.js "$B/@finos/perspective-viewer-datagrid@X/dist/cdn/perspective-viewer-datagrid.js"
curl -sSL -o wasm/perspective-server.wasm       "$B/@finos/perspective@X/dist/wasm/perspective-server.wasm"
curl -sSL -o wasm/perspective-viewer.wasm       "$B/@finos/perspective-viewer@X/dist/wasm/perspective-viewer.wasm"
curl -sSL -o css/themes.css                     "$B/@finos/perspective-viewer@X/dist/css/themes.css"
```

Then run the browser smoke test (`tests/frozen/m37/test_dashboard_browser.py`) to confirm the page
still renders with no console errors.

# Aries Image Studio web

React + TypeScript + Tailwind single-page app: timeline, composer, gallery, lightbox and
status panel. It talks to the manager's HTTP API (`/jobs`, `/status`, `/stage_runs`).

History, favorites and finished images live in the browser (localStorage for records,
IndexedDB for image blobs). Moving them server-side is planned; see the root README.

```bash
npm ci
npm run dev    # dev server on :5173, proxies the API to VITE_BACKEND_URL (default http://localhost:8000)
npm run build  # type-check + production build into dist/
npm run lint
```

If the manager runs on another host, put `VITE_BACKEND_URL=http://<manager-host>:8000`
in `web/.env.local` (gitignored) or the environment.

Behavior is specified in `../docs/SPEC_UI.md`; visuals follow `../mockups/`.

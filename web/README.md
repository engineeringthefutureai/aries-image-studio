# Aries Image Studio web

React + TypeScript + Tailwind single-page app: timeline, composer, gallery, lightbox and
status panel. Studio serves it and its API (`/api/*`, ARCHITECTURE_V3.md §8).

Nothing is kept in browser storage: jobs, images, favorites and preferences all load from
Studio, so every browser shows the same history.

```bash
npm ci
npm run dev    # dev server on :5173, proxies /api to VITE_BACKEND_URL (default http://localhost:8080)
npm run build  # type-check + production build into dist/
npm run lint
```

If Studio runs on another host, put `VITE_BACKEND_URL=http://<studio-host>:8080` in
`web/.env.local` (gitignored) or the environment.

Behavior is specified in `../docs/SPEC_UI.md`; visuals follow `../mockups/`.

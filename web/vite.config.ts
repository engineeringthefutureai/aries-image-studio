import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { readFileSync } from 'node:fs'
import { defineConfig, loadEnv } from 'vite'

const { version } = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf-8'))

export default defineConfig(({ mode }) => {
  // The dev server proxies API calls to Studio. Set VITE_BACKEND_URL in the
  // environment or in web/.env.local (gitignored) when it runs on another host.
  const env = loadEnv(mode, process.cwd(), 'VITE_')
  const backendTarget = env.VITE_BACKEND_URL || 'http://localhost:8080'
  const proxy = Object.fromEntries(
    ['/api', '/metrics'].map((path) => [
      path,
      { target: backendTarget, changeOrigin: true },
    ])
  )

  return {
    plugins: [react(), tailwindcss()],
    define: {
      __APP_VERSION__: JSON.stringify(version),
    },
    server: {
      host: true,
      port: 5173,
      proxy,
    },
  }
})

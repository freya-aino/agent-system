import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// The dev server proxies to S1 so the UI is developed against the real API. A mock
// would let the UI drift from the service, and the one thing this interface must
// never do is imply a permission the server would refuse.
export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: 'dist',
    // S1 serves these from disk; no hashed subdirectories to configure on the
    // server side.
    assetsDir: 'assets',
    sourcemap: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/experiments': 'http://127.0.0.1:8080',
      '/deployments': 'http://127.0.0.1:8080',
      '/promotions': 'http://127.0.0.1:8080',
      '/policy': 'http://127.0.0.1:8080',
      '/budgets': 'http://127.0.0.1:8080',
      '/events': 'http://127.0.0.1:8080',
      '/jobs': 'http://127.0.0.1:8080',
      '/killswitch': 'http://127.0.0.1:8080',
      '/healthz': 'http://127.0.0.1:8080',
      '/openapi.json': 'http://127.0.0.1:8080',
    },
  },
})

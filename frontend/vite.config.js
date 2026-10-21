import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.VITE_API_TARGET || 'http://127.0.0.1:8000';
  return {
    server: {
      port: 5173,
      strictPort: true,
      proxy: Object.fromEntries(
        ['/health', '/labels', '/predict', '/docs', '/openapi.json'].map(
          (path) => [path, { target, changeOrigin: true }],
        ),
      ),
    },
  };
});

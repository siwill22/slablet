import { defineConfig } from 'vite';

// Local only: `npm run dev` on localhost:5180 (5173-5179 are left to other viewers).
export default defineConfig({
  base: process.env.VITE_BASE ?? '/',
  server: { port: 5180 },
  build: { target: 'es2022', chunkSizeWarningLimit: 700 },
});

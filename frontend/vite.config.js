import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/hotels': 'http://localhost:5000',
      '/upload': 'http://localhost:5000',
      '/ask': 'http://localhost:5000',
      '/delete': 'http://localhost:5000',
      '/browse': 'http://localhost:5000',
      '/health': 'http://localhost:5000'
    }
  }
})

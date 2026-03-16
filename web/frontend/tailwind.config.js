/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: '#00ff88',
        dark: { DEFAULT: '#0a0a0a', 1: '#111', 2: '#1a1a1a', 3: '#222' },
      },
      fontFamily: { mono: ['JetBrains Mono', 'Fira Code', 'monospace'] },
    },
  },
  plugins: [],
}

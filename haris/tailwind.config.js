/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bg-deep': 'var(--color-bg-deep)',
        'bg-surface': 'var(--color-bg-surface)',
        'bg-raised': 'var(--color-bg-raised)',
        border: 'var(--color-border)',
        muted: 'var(--color-muted)',
        'grey-mid': 'var(--color-grey-mid)',
        'grey-light': 'var(--color-grey-light)',
        'white-soft': 'var(--color-white-soft)',
        'red-hot': 'var(--color-red-hot)',
        'red-mid': 'var(--color-red-mid)',
        'red-dim': 'var(--color-red-dim)',
      },
      fontFamily: {
        rajdhani: ['Rajdhani', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
}

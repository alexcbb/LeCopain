// tailwind.config.js (CORRIGÉ)
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: { // Garder les couleurs
        'crt-green': '#33ff33',
        'crt-bg': '#0a1e0a',
        'tv-frame': '#2a2a2a',
        'tv-border': '#1e1e1e'
      },
       boxShadow: { // Garder les ombres
         'inset-tv-frame': 'inset 0 0 15px rgba(0,0,0,0.6)',
         'inset-screen': 'inset 0 0 40px 10px rgba(0,0,0,0.7)',
       }
    },
  },
  plugins: [],
}
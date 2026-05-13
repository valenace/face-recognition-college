/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/**/*.js",
    "./apps/**/templates/**/*.html",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'brand-yellow': {
          DEFAULT: '#FFCC00',
          pastel: '#FFF9E6',
          dark: '#CCA300',
        },
        'brand-blue': {
          DEFAULT: '#00247D',
          pastel: '#E6ECF9',
          dark: '#001A5A',
        },
        'pastel-bg': {
          light: '#F8F9FA',
          dark: '#121212',
        }
      }
    },
  },
  plugins: [],
}

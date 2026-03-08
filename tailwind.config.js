/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cream: '#F6F7EB',
        coral: '#E94F37',
        charcoal: '#393E41',
        'coral-dark': '#D43E2A',
      },
    },
  },
  plugins: [],
}

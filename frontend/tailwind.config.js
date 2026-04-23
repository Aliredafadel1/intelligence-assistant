/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: '#090d1f',
        panel: '#121935',
        borderGlow: 'rgba(122, 122, 255, 0.28)',
        accentBlue: '#4f8cff',
        accentPurple: '#a855f7',
      },
      boxShadow: {
        glass: '0 10px 40px rgba(8, 16, 48, 0.55)',
        neon: '0 0 0 1px rgba(122, 122, 255, 0.25), 0 0 24px rgba(88, 108, 255, 0.35)',
      },
      backgroundImage: {
        aurora:
          'radial-gradient(circle at 20% 20%, rgba(79,140,255,0.28), transparent 36%), radial-gradient(circle at 85% 0%, rgba(168,85,247,0.18), transparent 30%), linear-gradient(180deg, #060913 0%, #0b1021 65%, #0c1125 100%)',
      },
    },
  },
  plugins: [],
}


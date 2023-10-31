/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bkg: "#FFFFFF",
        content: "#151515",
        description: "#808080",
        red: "#D63F37",
        yellow: "#F1BE3F",
        green: "#2FA769",
        blue: "#1f75ff",
      },
      fontFamily: {
        Onest: ["Onest", "sans-serif"],
      },
    },
  },

  plugins: [],
}
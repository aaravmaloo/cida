import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cida: {
          bg: "#eff2f8",
          panel: "#f8faff",
          card: "#ffffff",
          line: "#d9e1ef",
          text: "#171f33",
          mute: "#6d7690",
          accent: "#1d5fff",
          accentSoft: "#d7e3ff",
        },
      },
      boxShadow: {
        panel: "0 8px 30px rgba(25, 43, 92, 0.08)",
      },
      fontFamily: {
        heading: ["var(--font-sora)", "sans-serif"],
        body: ["var(--font-manrope)", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;


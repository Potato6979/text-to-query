/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          ink: "#17202a",
          soft: "#405063",
          muted: "#738092",
          line: "#dce2dc",
          pine: "#25483e",
          blue: "#385a72",
          paper: "#fbfbf8",
          page: "#f4f6f2",
        }
      },
      fontFamily: {
        display: ["Segoe UI", "sans-serif"],
        body: ["Segoe UI", "sans-serif"]
      },
      boxShadow: {
        panel: "0 24px 60px rgba(28, 42, 53, 0.10)"
      }
    },
  },
  plugins: [],
};

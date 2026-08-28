/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        "bg-primary": "#0d1117",
        "bg-secondary": "#161b22",
        "bg-tertiary": "#1c2128",
        "text-primary": "#e6edf3",
        "text-secondary": "#8b949e",
        "text-tertiary": "#6e7681",
        "border-color": "#30363d",
        "border-light": "#21262d",
        "accent-blue": "#58a6ff",
        "accent-green": "#3fb950",
        "accent-orange": "#d29922",
        "accent-purple": "#bc8cff",
        "accent-red": "#f85149",
        "accent-cyan": "#39c5cf",
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', '"PingFang SC"', '"Noto Sans SC"', '"Microsoft YaHei"', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'Consolas', '"Courier New"', 'monospace'],
      },
    },
  },
  plugins: [],
};

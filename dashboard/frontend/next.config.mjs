const API = process.env.API_URL || "http://127.0.0.1:8000";
/** @type {import('next').NextConfig} */
export default {
  reactStrictMode: false,          // the canvas overlay owns a rAF loop
  async rewrites() {
    // Proxy the API through Next so the browser sees one origin: no CORS
    // preflight on every chunk request, and video range requests stay simple.
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

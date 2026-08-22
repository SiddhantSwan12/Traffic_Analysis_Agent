import "./globals.css";

export const metadata = {
  title: "Traffic Insight",
  description: "Drone traffic detection, tracking and network analysis",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

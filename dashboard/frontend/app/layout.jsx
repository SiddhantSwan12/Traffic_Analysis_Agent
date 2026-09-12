import "./globals.css";

export const metadata = {
  title: "FlytBase Traffic Intelligence",
  description: "Professional traffic operations and network analytics dashboard",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

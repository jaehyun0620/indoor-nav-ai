import "./globals.css";

export const metadata = {
  title: "실내 길 안내 — Indoor Nav AI",
  description: "시각장애인을 위한 AI 실내 길 안내 시스템",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export default function RootLayout({ children }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}

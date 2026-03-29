import "./globals.css";

export const metadata = {
  title: "실내 길 안내 — Indoor Nav AI",
  description: "시각장애인을 위한 AI 실내 길 안내 시스템",
  viewport: "width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no",
};

export default function RootLayout({ children }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

// 백엔드 URL 설정
// 백엔드는 기본적으로 HTTPS로 실행됨 (ENABLE_SSL=true)
// HTTP로 실행 중이면 http://localhost:8081로 변경
// 환경 변수로 설정 가능: VITE_BACKEND_URL=http://localhost:8081
const BACKEND_URL = "https://localhost:8081"; // HTTPS로 변경 (백엔드가 HTTPS로 실행 중)
const BACKEND_WS_URL = "wss://localhost:8081"; // WebSocket도 HTTPS로 변경

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    basicSsl(), // HTTPS 활성화 (모바일 카메라 접근을 위해 필요)
  ],
  server: {
    host: "0.0.0.0", // 모든 네트워크 인터페이스에서 접근 가능
    port: 5173,
    https: true, // HTTPS 활성화
    strictPort: true, // 포트 고정
    open: false, // 자동 브라우저 열기 비활성화
    hmr: {
      protocol: "wss", // WebSocket Secure
      // host를 제거하여 자동 감지 (다양한 네트워크 환경에서 작동)
      // clientPort도 제거하여 자동 감지
      // HMR WebSocket 오류 무시 (정상적인 연결 종료)
      client: {
        overlay: {
          warnings: false,
          errors: true,
        },
      },
    },
    // 백엔드 API 프록시 설정
    proxy: {
      "/api": {
        // 백엔드 서버는 기본적으로 HTTPS로 실행됨 (ENABLE_SSL=true)
        // HTTP로 실행 중이면 http://localhost:8081로 변경
        // 환경 변수 VITE_BACKEND_URL로 설정 가능 (예: http://localhost:8081)
        target: BACKEND_URL,
        changeOrigin: true,
        secure: false, // 자체 서명 인증서 허용
        timeout: 60000, // 60초 타임아웃
        proxyTimeout: 60000,
        configure: (proxy, _options) => {
          proxy.on("error", (err, req, res) => {
            // EPIPE, ECONNRESET은 일반적인 연결 종료 오류이므로 로그 레벨 낮춤
            if (err.code !== "EPIPE" && err.code !== "ECONNRESET") {
              console.error("[프록시 오류]", err.code || err.message, req?.url);
            }
            // 응답이 아직 시작되지 않았으면 502 에러 반환
            if (!res.headersSent) {
              res.writeHead(502, {
                "Content-Type": "text/plain",
              });
              res.end(
                "백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
              );
            }
          });
          proxy.on("proxyReq", (proxyReq, req) => {
            // 스트림 요청인 경우 특별 처리
            if (req.url.includes("/stream")) {
              proxyReq.setTimeout(0); // 타임아웃 없음
              proxyReq.setHeader("Connection", "keep-alive");
              proxyReq.setHeader("Accept", "*/*");
            } else if (req.url.includes("capture-stream")) {
              // 큰 파일 업로드 시 타임아웃 증가
              proxyReq.setTimeout(60000);
              // Expect: 100-continue 헤더 제거 (일부 서버에서 문제 발생)
              proxyReq.removeHeader("expect");
              // Connection 헤더 명시적 설정
              proxyReq.setHeader("Connection", "keep-alive");
              // Content-Length가 있으면 유지
              if (req.headers["content-length"]) {
                proxyReq.setHeader(
                  "Content-Length",
                  req.headers["content-length"]
                );
              }
            } else {
              // 일반 API 요청은 60초 타임아웃
              proxyReq.setTimeout(60000);
            }
          });
          proxy.on("proxyRes", (proxyRes, req) => {
            // 스트림 응답인 경우 Connection 헤더 유지
            if (req.url.includes("/stream")) {
              // 스트림 응답은 그대로 전달
            } else if (req.url.includes("capture-stream")) {
              // 응답 수신 시 로깅
              console.log(`[프록시 응답] ${proxyRes.statusCode} ${req.url}`);
            } else {
              // 일반 API 응답 로깅 (디버깅용)
              if (proxyRes.statusCode >= 400) {
                console.warn(`[프록시 응답] ${proxyRes.statusCode} ${req.url}`);
              }
            }
          });
        },
      },
      "/images": {
        target: BACKEND_URL,
        changeOrigin: true,
        secure: false, // 자체 서명 인증서 허용
        timeout: 30000,
      },
      "/uploads": {
        target: BACKEND_URL,
        changeOrigin: true,
        secure: false, // 자체 서명 인증서 허용
        timeout: 30000,
      },
      // WebSocket 프록시
      "/ws": {
        target: BACKEND_WS_URL,
        ws: true,
        changeOrigin: true,
        secure: false,
        configure: (proxy, _options) => {
          proxy.on("error", (err, _req, _res) => {
            // 일반적인 연결 종료 오류는 무시 (스트리밍 연결에서 정상적인 현상)
            if (
              err.code !== "EPIPE" &&
              err.code !== "ECONNRESET" &&
              err.code !== "ECONNREFUSED" &&
              !err.message.includes("Parse Error")
            ) {
              console.log("[WebSocket 프록시] 오류:", err.code || err.message);
            }
          });
          proxy.on("proxyReqWs", (proxyReq, _req, _socket) => {
            // WebSocket 연결 요청 시 로깅 (선택적)
            // console.log("[WebSocket 프록시] 연결 요청:", _req.url);
          });
        },
      },
      // MJPEG 스트림 프록시 (스트리밍 최적화) - /api/stream은 /api 프록시로 처리됨
    },
  },
});

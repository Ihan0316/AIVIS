/**
 * 프론트엔드 로그를 서버로 전송하는 유틸리티
 * 콘솔 로그를 서버의 로그 파일에 저장
 */

const API_BASE_URL = "/api";
const LOG_BUFFER_SIZE = 10; // 버퍼 크기 (배치 전송)
const LOG_SEND_INTERVAL = 2000; // 2초마다 전송

// 로그 버퍼
let logBuffer = [];
let sendTimer = null;

// 원본 console 메서드 저장
const originalConsole = {
  log: console.log.bind(console),
  error: console.error.bind(console),
  warn: console.warn.bind(console),
  info: console.info.bind(console),
  debug: console.debug.bind(console),
};

/**
 * 로그를 서버로 전송
 */
async function sendLogsToServer() {
  if (logBuffer.length === 0) return;

  const logsToSend = [...logBuffer];
  logBuffer = [];

  try {
    await fetch(`${API_BASE_URL}/frontend-logs`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        logs: logsToSend,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
      }),
    }).catch(() => {
      // 전송 실패 시 버퍼에 다시 추가 (최대 100개까지만)
      if (logBuffer.length < 100) {
        logBuffer.unshift(...logsToSend);
      }
    });
  } catch (error) {
    // 전송 실패 시 원본 console로만 출력
    originalConsole.error("[Logger] 로그 전송 실패:", error);
  }
}

/**
 * 로그를 버퍼에 추가하고 주기적으로 전송
 */
function addToBuffer(level, args) {
  const logEntry = {
    level,
    message: args
      .map((arg) => {
        if (typeof arg === "object") {
          try {
            return JSON.stringify(arg);
          } catch {
            return String(arg);
          }
        }
        return String(arg);
      })
      .join(" "),
    timestamp: new Date().toISOString(),
    stack: new Error().stack,
  };

  logBuffer.push(logEntry);

  // 버퍼가 가득 차면 즉시 전송
  if (logBuffer.length >= LOG_BUFFER_SIZE) {
    sendLogsToServer();
  } else {
    // 타이머가 없으면 설정
    if (!sendTimer) {
      sendTimer = setTimeout(() => {
        sendLogsToServer();
        sendTimer = null;
      }, LOG_SEND_INTERVAL);
    }
  }
}

/**
 * console 메서드 오버라이드
 */
function overrideConsole() {
  console.log = function (...args) {
    originalConsole.log(...args);
    addToBuffer("LOG", args);
  };

  console.error = function (...args) {
    originalConsole.error(...args);
    addToBuffer("ERROR", args);
  };

  console.warn = function (...args) {
    originalConsole.warn(...args);
    addToBuffer("WARN", args);
  };

  console.info = function (...args) {
    originalConsole.info(...args);
    addToBuffer("INFO", args);
  };

  console.debug = function (...args) {
    originalConsole.debug(...args);
    addToBuffer("DEBUG", args);
  };
}

/**
 * 페이지 언로드 시 남은 로그 전송
 */
function setupUnloadHandler() {
  window.addEventListener("beforeunload", () => {
    if (logBuffer.length > 0) {
      // 동기적으로 전송 (navigator.sendBeacon 사용)
      const logsJson = JSON.stringify({
        logs: logBuffer,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
      });

      navigator.sendBeacon(
        `${API_BASE_URL}/frontend-logs`,
        new Blob([logsJson], { type: "application/json" })
      );
    }
  });
}

/**
 * 로거 초기화
 */
export function initLogger() {
  overrideConsole();
  setupUnloadHandler();
  originalConsole.log("[Logger] 프론트엔드 로거 초기화 완료");
}

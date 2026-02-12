import { createRoot } from "react-dom/client";
import "./index.css";
import AIVISApp from "./App.jsx";
import { initLogger } from "./utils/logger";

// 프론트엔드 로거 초기화 (콘솔 로그를 서버로 전송)
initLogger();

// StrictMode 제거: WebSocket 및 카메라 중복 연결 방지
// 개발 중에는 useEffect가 두 번 실행되어 WebSocket과 카메라 접근이 중복됨
createRoot(document.getElementById("root")).render(<AIVISApp />);

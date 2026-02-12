import React, { useEffect, useRef, useState } from 'react';

/**
 * 카메라 뷰 컴포넌트
 * 실시간 CCTV 스트림 및 바운딩박스 표시
 */
export function CameraView({ 
  camId, 
  label, 
  wsUrl, 
  onViolation, 
  isFocused = false,
  onClick 
}) {
  const canvasRef = useRef(null);
  const [fps, setFps] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  useEffect(() => {
    if (!wsUrl) return;

    const connect = () => {
      try {
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          console.log(`[CAM-${camId}] WebSocket connected`);
          setIsConnected(true);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            // 프레임 렌더링
            if (data.frame && canvasRef.current) {
              const img = new Image();
              img.onload = () => {
                const ctx = canvasRef.current.getContext('2d');
                ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
              };
              img.src = `data:image/jpeg;base64,${data.frame}`;
            }

            // FPS 계산
            frameCountRef.current++;
            const now = Date.now();
            if (now - lastFpsUpdateRef.current >= 1000) {
              setFps(frameCountRef.current);
              frameCountRef.current = 0;
              lastFpsUpdateRef.current = now;
            }

            // 위반 콜백
            if (data.violations && data.violations.length > 0 && onViolation) {
              onViolation(camId, data.violations);
            }
          } catch (err) {
            console.error(`[CAM-${camId}] Parse error:`, err);
          }
        };

        ws.onclose = () => {
          console.log(`[CAM-${camId}] WebSocket disconnected`);
          setIsConnected(false);
          // 재연결 시도
          setTimeout(connect, 3000);
        };

        ws.onerror = (err) => {
          console.error(`[CAM-${camId}] WebSocket error:`, err);
        };

        wsRef.current = ws;
      } catch (err) {
        console.error(`[CAM-${camId}] Connection error:`, err);
      }
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [wsUrl, camId, onViolation]);

  return (
    <div 
      className={`camera-view ${isFocused ? 'focused' : ''}`}
      onClick={onClick}
      style={{
        position: 'relative',
        backgroundColor: '#1a1a2e',
        borderRadius: '8px',
        overflow: 'hidden',
        cursor: 'pointer'
      }}
    >
      {/* 헤더 */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        padding: '8px 12px',
        background: 'linear-gradient(to bottom, rgba(0,0,0,0.7), transparent)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        zIndex: 10
      }}>
        <span style={{ color: '#fff', fontWeight: 'bold' }}>{label}</span>
        <span style={{ 
          color: isConnected ? '#4ade80' : '#ef4444',
          fontSize: '12px'
        }}>
          {fps.toFixed(1)} FPS
        </span>
      </div>

      {/* 캔버스 */}
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover'
        }}
      />

      {/* 연결 상태 */}
      {!isConnected && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: '#fff',
          textAlign: 'center'
        }}>
          <div>연결 중...</div>
        </div>
      )}
    </div>
  );
}

export default CameraView;


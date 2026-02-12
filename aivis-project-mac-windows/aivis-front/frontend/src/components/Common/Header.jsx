import React from 'react';

/**
 * 헤더 컴포넌트
 * 시스템 제목, 날씨, 시간 표시
 */
export function Header({ 
  weather, 
  currentTime,
  onSearchClick,
  onNotificationClick,
  onSettingsClick 
}) {
  // 날씨 아이콘 매핑
  const getWeatherIcon = (condition) => {
    if (!condition) return '🌤️';
    const c = condition.toLowerCase();
    if (c.includes('clear') || c.includes('sunny')) return '☀️';
    if (c.includes('cloud')) return '☁️';
    if (c.includes('rain')) return '🌧️';
    if (c.includes('snow')) return '❄️';
    if (c.includes('thunder')) return '⛈️';
    return '🌤️';
  };

  // 시간 포맷
  const formatTime = (date) => {
    if (!date) return '--:--:--';
    return date.toLocaleTimeString('ko-KR', { hour12: false });
  };

  // 날짜 포맷
  const formatDate = (date) => {
    if (!date) return '';
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    return `${date.getFullYear()}년 ${date.getMonth() + 1}월 ${date.getDate()}일 (${days[date.getDay()]})`;
  };

  return (
    <header style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '12px 24px',
      backgroundColor: '#1e1e2e',
      borderBottom: '1px solid #2d2d3d'
    }}>
      {/* 로고 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <span style={{ color: '#fff', fontWeight: 'bold' }}>AI</span>
        </div>
        <span style={{ 
          color: '#fff', 
          fontSize: '20px', 
          fontWeight: 'bold',
          letterSpacing: '2px'
        }}>
          AIVIS
        </span>
      </div>

      {/* 날씨 정보 */}
      {weather && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
          color: '#9ca3af'
        }}>
          <span style={{ fontSize: '20px' }}>
            {getWeatherIcon(weather.description)}
          </span>
          <span>{weather.temp?.toFixed(0) || '--'}°C</span>
          <span>💧 {weather.humidity || '--'}%</span>
        </div>
      )}

      {/* 날짜 및 시간 */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '24px'
      }}>
        <span style={{ color: '#9ca3af' }}>
          {formatDate(currentTime)}
        </span>
        <span style={{ 
          color: '#fff', 
          fontSize: '24px', 
          fontWeight: 'bold',
          fontFamily: 'monospace'
        }}>
          {formatTime(currentTime)}
        </span>
      </div>

      {/* 액션 버튼 */}
      <div style={{ display: 'flex', gap: '12px' }}>
        <button
          onClick={onSearchClick}
          style={{
            padding: '8px',
            backgroundColor: 'transparent',
            border: 'none',
            color: '#9ca3af',
            cursor: 'pointer',
            fontSize: '18px'
          }}
          title="검색"
        >
          🔍
        </button>
        <button
          onClick={onNotificationClick}
          style={{
            padding: '8px',
            backgroundColor: 'transparent',
            border: 'none',
            color: '#9ca3af',
            cursor: 'pointer',
            fontSize: '18px'
          }}
          title="알림"
        >
          🔔
        </button>
        <button
          onClick={onSettingsClick}
          style={{
            padding: '8px',
            backgroundColor: 'transparent',
            border: 'none',
            color: '#9ca3af',
            cursor: 'pointer',
            fontSize: '18px'
          }}
          title="설정"
        >
          ⚙️
        </button>
      </div>
    </header>
  );
}

export default Header;


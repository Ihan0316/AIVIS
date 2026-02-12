import React, { useState, useMemo } from 'react';

/**
 * 이벤트 로그 컴포넌트
 * 위반 사항 목록 표시 및 필터링
 */
export function EventLog({ 
  events = [], 
  onEventClick,
  maxItems = 50 
}) {
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  // 필터링된 이벤트
  const filteredEvents = useMemo(() => {
    return events
      .filter(event => {
        // 타입 필터
        if (filter !== 'all') {
          const eventType = event.violation_type || event.type || '';
          if (filter === 'fall' && !eventType.includes('넘어짐') && !eventType.includes('fall')) {
            return false;
          }
          if (filter === 'ppe' && (eventType.includes('넘어짐') || eventType.includes('fall'))) {
            return false;
          }
        }
        
        // 검색 필터
        if (searchQuery) {
          const searchLower = searchQuery.toLowerCase();
          const name = (event.worker_name || event.name || '').toLowerCase();
          const type = (event.violation_type || event.type || '').toLowerCase();
          return name.includes(searchLower) || type.includes(searchLower);
        }
        
        return true;
      })
      .slice(0, maxItems);
  }, [events, filter, searchQuery, maxItems]);

  // 위반 타입에 따른 색상
  const getTypeColor = (type) => {
    if (!type) return '#6b7280';
    if (type.includes('넘어짐') || type.includes('fall')) return '#ef4444';
    if (type.includes('안전모') || type.includes('helmet')) return '#f59e0b';
    if (type.includes('마스크') || type.includes('mask')) return '#3b82f6';
    if (type.includes('조끼') || type.includes('vest')) return '#8b5cf6';
    return '#6b7280';
  };

  // 시간 포맷
  const formatTime = (timestamp) => {
    if (!timestamp) return '--:--:--';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('ko-KR', { hour12: false });
  };

  return (
    <div className="event-log" style={{
      backgroundColor: '#1e1e2e',
      borderRadius: '8px',
      padding: '16px',
      height: '100%',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* 헤더 */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '12px'
      }}>
        <h3 style={{ color: '#fff', margin: 0 }}>이벤트 로그</h3>
        <span style={{ color: '#9ca3af', fontSize: '12px' }}>
          {filteredEvents.length}건
        </span>
      </div>

      {/* 필터 */}
      <div style={{
        display: 'flex',
        gap: '8px',
        marginBottom: '12px'
      }}>
        {['all', 'fall', 'ppe'].map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              padding: '4px 12px',
              borderRadius: '4px',
              border: 'none',
              backgroundColor: filter === f ? '#3b82f6' : '#374151',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            {f === 'all' ? '전체' : f === 'fall' ? '넘어짐' : 'PPE'}
          </button>
        ))}
      </div>

      {/* 검색 */}
      <input
        type="text"
        placeholder="이름 또는 위반 유형 검색..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        style={{
          padding: '8px 12px',
          borderRadius: '4px',
          border: '1px solid #374151',
          backgroundColor: '#111827',
          color: '#fff',
          marginBottom: '12px',
          fontSize: '14px'
        }}
      />

      {/* 이벤트 목록 */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px'
      }}>
        {filteredEvents.length === 0 ? (
          <div style={{ 
            color: '#6b7280', 
            textAlign: 'center', 
            padding: '20px' 
          }}>
            이벤트가 없습니다
          </div>
        ) : (
          filteredEvents.map((event, index) => (
            <div
              key={event.id || index}
              onClick={() => onEventClick?.(event)}
              style={{
                padding: '12px',
                backgroundColor: '#111827',
                borderRadius: '6px',
                borderLeft: `4px solid ${getTypeColor(event.violation_type || event.type)}`,
                cursor: 'pointer',
                transition: 'background-color 0.2s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#1f2937'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#111827'}
            >
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start'
              }}>
                <div>
                  <div style={{ color: '#fff', fontWeight: 'bold', marginBottom: '4px' }}>
                    {event.worker_name || event.name || '알 수 없음'}
                  </div>
                  <div style={{ 
                    color: getTypeColor(event.violation_type || event.type),
                    fontSize: '12px'
                  }}>
                    {event.violation_type || event.type || '위반'}
                  </div>
                </div>
                <div style={{ color: '#9ca3af', fontSize: '11px' }}>
                  {formatTime(event.timestamp || event.created_at)}
                </div>
              </div>
              {event.area && (
                <div style={{ color: '#6b7280', fontSize: '11px', marginTop: '4px' }}>
                  구역: {event.area}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default EventLog;


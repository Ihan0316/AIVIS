import React from 'react';

/**
 * KPI 카드 컴포넌트
 * 주요 지표를 시각화하여 표시
 */
export function KPICard({ title, value, icon, color, trend, trendValue }) {
  return (
    <div className="kpi-card" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="kpi-header">
        <span className="kpi-icon" style={{ color }}>
          {icon}
        </span>
        <h3 className="kpi-title">{title}</h3>
      </div>

      <div className="kpi-body">
        <div className="kpi-value">{value}</div>

        {trend && (
          <div className={`kpi-trend ${trend}`}>
            <span className="trend-icon">
              {trend === 'up' ? '▲' : trend === 'down' ? '▼' : '━'}
            </span>
            <span className="trend-value">{trendValue}</span>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * KPI 그리드 컴포넌트
 * 여러 KPI 카드를 그리드 형태로 표시
 */
export function KPIGrid({ children }) {
  return <div className="kpi-grid">{children}</div>;
}

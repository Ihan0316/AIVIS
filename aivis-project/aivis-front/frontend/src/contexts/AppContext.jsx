import React, { createContext, useContext, useState, useCallback } from 'react';

/**
 * 전역 애플리케이션 상태 Context
 * 페이지 네비게이션, UI 상태, 설정 관리
 */
const AppContext = createContext();

export function AppProvider({ children }) {
  // 페이지 네비게이션
  const urlParams = new URLSearchParams(window.location.search);
  const isCameraMode = urlParams.get('camera') === 'true';
  const [activePage, setActivePage] = useState(isCameraMode ? 'access-camera' : 'dashboard');

  // UI 상태
  const [showSearch, setShowSearch] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [focusCam, setFocusCam] = useState(null);
  const [showBottomRightPopup, setShowBottomRightPopup] = useState(null);

  // 설정
  const [language, setLanguage] = useState('ko');
  const [theme, setTheme] = useState('dark');
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [dense, setDense] = useState(true);

  // 검색
  const [searchQuery, setSearchQuery] = useState('');
  const [searchHistory, setSearchHistory] = useState([]);
  const [showSearchHistory, setShowSearchHistory] = useState(false);

  // 검색 기록 추가
  const addToSearchHistory = useCallback((query) => {
    if (!query || query.trim() === '') return;

    const trimmedQuery = query.trim();
    setSearchHistory((prev) => {
      const filtered = prev.filter((item) => item !== trimmedQuery);
      const updated = [trimmedQuery, ...filtered].slice(0, 10);
      return updated;
    });
  }, []);

  // 검색 기록에서 선택
  const selectFromHistory = useCallback((query) => {
    setSearchQuery(query);
    setShowSearchHistory(false);
  }, []);

  // 텍스트 하이라이트
  const highlightText = useCallback((text, query) => {
    if (!query || !text || query.trim() === '') return text;

    const searchQuery = query.trim();
    const regex = new RegExp(
      `(${searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`,
      'gi'
    );
    const parts = String(text).split(regex);

    return parts.map((part, index) =>
      regex.test(part) ? (
        <mark
          key={index}
          style={{
            background: 'var(--accent-blue)',
            color: 'white',
            padding: '2px 4px',
            borderRadius: '3px',
            fontWeight: '600',
          }}
        >
          {part}
        </mark>
      ) : (
        part
      )
    );
  }, []);

  const value = {
    // 페이지 네비게이션
    activePage,
    setActivePage,

    // UI 상태
    showSearch,
    setShowSearch,
    showNotifications,
    setShowNotifications,
    showSettings,
    setShowSettings,
    focusCam,
    setFocusCam,
    showBottomRightPopup,
    setShowBottomRightPopup,

    // 설정
    language,
    setLanguage,
    theme,
    setTheme,
    notificationsEnabled,
    setNotificationsEnabled,
    dense,
    setDense,

    // 검색
    searchQuery,
    setSearchQuery,
    searchHistory,
    showSearchHistory,
    setShowSearchHistory,
    addToSearchHistory,
    selectFromHistory,
    highlightText,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
}

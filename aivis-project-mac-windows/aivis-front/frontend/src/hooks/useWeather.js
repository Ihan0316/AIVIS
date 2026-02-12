import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

/**
 * 날씨 정보 Hook
 * 사용자 위치 기반 날씨 정보 조회
 */
export function useWeather() {
  const [weather, setWeather] = useState(null);
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [error, setError] = useState(null);

  // 위치 정보 가져오기
  const getUserLocation = useCallback(() => {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation is not supported'));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          const location = {
            lat: position.coords.latitude,
            lon: position.coords.longitude,
          };
          setUserLocation(location);
          resolve(location);
        },
        (error) => {
          console.error('Failed to get location:', error);
          reject(error);
        }
      );
    });
  }, []);

  // 날씨 정보 가져오기
  const fetchWeather = useCallback(async (location) => {
    setWeatherLoading(true);
    setError(null);

    try {
      const response = await api.getWeather(location.lat, location.lon);

      if (response.success && response.weather) {
        setWeather(response.weather);
      } else {
        setError('Failed to fetch weather data');
      }
    } catch (err) {
      console.error('Weather fetch error:', err);
      setError(err.message);
    } finally {
      setWeatherLoading(false);
    }
  }, []);

  // 날씨 정보 갱신
  const refreshWeather = useCallback(async () => {
    try {
      const location = userLocation || await getUserLocation();
      await fetchWeather(location);
    } catch (err) {
      console.error('Failed to refresh weather:', err);
      setError(err.message);
    }
  }, [userLocation, getUserLocation, fetchWeather]);

  // 초기 로드
  useEffect(() => {
    refreshWeather();

    // 30분마다 자동 갱신
    const interval = setInterval(refreshWeather, 30 * 60 * 1000);

    return () => clearInterval(interval);
  }, [refreshWeather]);

  return {
    weather,
    weatherLoading,
    userLocation,
    error,
    refreshWeather,
  };
}

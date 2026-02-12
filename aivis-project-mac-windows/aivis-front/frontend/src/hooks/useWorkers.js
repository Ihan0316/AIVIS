import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

/**
 * 작업자 관리 Hook
 * 작업자 목록 로드, 생성, 수정, 삭제 기능 제공
 */
export function useWorkers() {
  const [workersList, setWorkersList] = useState([]);
  const [loadingWorkers, setLoadingWorkers] = useState(true);
  const [totalWorkers, setTotalWorkers] = useState(0);

  // 작업자 목록 로드
  const loadWorkers = useCallback(async () => {
    setLoadingWorkers(true);
    try {
      const response = await api.getWorkers();

      if (response.success && response.workers) {
        setWorkersList(response.workers);
        setTotalWorkers(response.workers.filter((w) => w.role === 'worker').length);
      } else {
        setWorkersList([]);
      }
    } catch (error) {
      console.error('작업자 목록 로드 오류:', error);
      setWorkersList([]);
    } finally {
      setLoadingWorkers(false);
    }
  }, []);

  // 작업자 생성
  const createWorker = useCallback(async (workerData) => {
    try {
      const response = await api.createWorker(workerData);
      if (response.success) {
        await loadWorkers(); // 목록 갱신
        return { success: true };
      }
      return { success: false, error: response.error };
    } catch (error) {
      console.error('작업자 생성 오류:', error);
      return { success: false, error: error.message };
    }
  }, [loadWorkers]);

  // 작업자 수정
  const updateWorker = useCallback(async (workerId, workerData) => {
    try {
      const response = await api.updateWorker(workerId, workerData);
      if (response.success) {
        await loadWorkers(); // 목록 갱신
        return { success: true };
      }
      return { success: false, error: response.error };
    } catch (error) {
      console.error('작업자 수정 오류:', error);
      return { success: false, error: error.message };
    }
  }, [loadWorkers]);

  // 작업자 삭제
  const deleteWorker = useCallback(async (workerId) => {
    try {
      const response = await api.deleteWorker(workerId);
      if (response.success) {
        await loadWorkers(); // 목록 갱신
        return { success: true };
      }
      return { success: false, error: response.error };
    } catch (error) {
      console.error('작업자 삭제 오류:', error);
      return { success: false, error: error.message };
    }
  }, [loadWorkers]);

  // 초기 로드
  useEffect(() => {
    loadWorkers();
  }, [loadWorkers]);

  return {
    workersList,
    loadingWorkers,
    totalWorkers,
    loadWorkers,
    createWorker,
    updateWorker,
    deleteWorker,
  };
}

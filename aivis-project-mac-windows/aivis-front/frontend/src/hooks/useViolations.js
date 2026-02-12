import { useState, useEffect, useCallback } from "react";
import { api } from "../services/api";
import { mapWorkerName } from "../utils/utils";

/**
 * 위반 사항 관리 Hook
 * KPI, 이벤트 로그, 통계 데이터 관리
 */
export function useViolations() {
  const [kpiHelmet, setKpiHelmet] = useState(0);
  const [kpiVest, setKpiVest] = useState(0);
  const [kpiFall, setKpiFall] = useState(0);
  const [totalAlerts, setTotalAlerts] = useState(0);
  const [pendingCount, setPendingCount] = useState(0);
  const [eventRows, setEventRows] = useState([]);
  const [fullLogs, setFullLogs] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [completedActions, setCompletedActions] = useState(0);
  const [safetyScore, setSafetyScore] = useState(0);
  const [weeklyTotalDetections, setWeeklyTotalDetections] = useState(0);
  const [weeklyHelmetViolations, setWeeklyHelmetViolations] = useState(0);
  const [weeklyFallDetections, setWeeklyFallDetections] = useState(0);

  // 위반 사항 통계 로드
  const loadViolationStats = useCallback(async () => {
    try {
      const response = await api.getViolationStats();

      if (response.success) {
        // DB 응답 형식: {success: true, kpi: {...}, chart_data: [...]}
        const kpi = response.kpi || {};
        const chart_data = response.chart_data || [];

        // KPI 업데이트
        setKpiHelmet(kpi.helmet || 0);
        setKpiVest(kpi.vest || 0);
        setKpiFall(kpi.fall || 0);
        setTotalAlerts(kpi.total || 0);

        // chart_data에서 주간 통계 계산
        const totalWeekly = chart_data.reduce(
          (sum, week) => sum + (week.violations || 0),
          0
        );
        const totalHelmet = chart_data.reduce(
          (sum, week) => sum + (week.helmet_violations || 0),
          0
        );
        const totalFall = chart_data.reduce(
          (sum, week) => sum + (week.fall_detections || 0),
          0
        );

        setWeeklyTotalDetections(totalWeekly);
        setWeeklyHelmetViolations(totalHelmet || kpi.helmet || 0);
        setWeeklyFallDetections(totalFall || kpi.fall || 0);

        // chart_data 설정
        setChartData(chart_data);

        // 안전 점수 계산 (위반 건수가 적을수록 높음)
        const totalViolations = kpi.total || 0;
        // 최소 0점, 최대 100점 제한
        const score = Math.min(100, Math.max(0, 100 - totalViolations * 2));
        setSafetyScore(score);
      }
    } catch (error) {
      console.error("위반 통계 로드 오류:", error);
    }
  }, []);

  // 위반 사항 목록 로드
  const loadViolations = useCallback(async (filters = {}) => {
    try {
      const response = await api.getViolations(filters);

      if (response.success && response.violations) {
        setFullLogs(response.violations);

        // pendingCount와 completedActions 계산
        const pending = response.violations.filter(
          (v) => v.status === "new" || v.status === "pending" || !v.status
        ).length;
        const completed = response.violations.filter(
          (v) => v.status === "done"
        ).length;
        setPendingCount(pending);
        setCompletedActions(completed);

        // 이벤트 행 업데이트 (최근 10개)
        // DB 필드명 매핑: cam_id 또는 camera_id, type 또는 violation_type
        const recentEvents = response.violations.slice(0, 10).map((v) => {
          const cam_id =
            v.cam_id !== undefined
              ? v.cam_id
              : v.camera_id !== undefined
              ? v.camera_id
              : 0;
          const zone =
            v.work_zone ||
            (cam_id !== undefined
              ? ["A", "B", "C", "D"][cam_id] || "N/A"
              : "N/A");

          return {
            id: v._id || v.id,
            worker: mapWorkerName(v.worker_name || v.worker_id || "Unknown"),
            zone: zone,
            risk: v.type || v.violation_type || "Unknown",
            status: v.status || "pending",
            timestamp: v.timestamp,
          };
        });

        setEventRows(recentEvents);
      }
    } catch (error) {
      console.error("위반 목록 로드 오류:", error);
      setFullLogs([]);
      setEventRows([]);
    }
  }, []);

  // 위반 상태 업데이트 (개별 업데이트 최적화)
  const updateViolationStatus = useCallback(
    async (violationId, newStatus) => {
      try {
        const response = await api.updateViolationStatus(
          violationId,
          newStatus
        );

        if (response.success) {
          // 로컬 상태 업데이트 (개별 항목만 업데이트)
          setFullLogs((prev) =>
            prev.map((v) =>
              v._id === violationId || v.id === violationId
                ? { ...v, status: newStatus }
                : v
            )
          );

          // 통계는 배치 업데이트로 최적화 (즉시 업데이트하지 않고 지연)
          // 여러 상태 변경을 한 번에 처리하기 위해 디바운싱
          if (!updateViolationStatus.statsUpdateTimeout) {
            updateViolationStatus.statsUpdateTimeout = setTimeout(() => {
              loadViolationStats();
              updateViolationStatus.statsUpdateTimeout = null;
            }, 500); // 500ms 내 여러 업데이트가 있으면 마지막 것만 실행
          } else {
            // 이미 타이머가 있으면 기존 타이머 취소하고 새로 설정
            clearTimeout(updateViolationStatus.statsUpdateTimeout);
            updateViolationStatus.statsUpdateTimeout = setTimeout(() => {
              loadViolationStats();
              updateViolationStatus.statsUpdateTimeout = null;
            }, 500);
          }

          return { success: true };
        }

        return { success: false, error: response.error };
      } catch (error) {
        console.error("위반 상태 업데이트 오류:", error);
        return { success: false, error: error.message };
      }
    },
    [loadViolationStats]
  );

  // 차트 데이터 생성
  const generateChartData = useCallback(() => {
    if (!fullLogs.length) return;

    // 날짜별 위반 건수 집계
    // DB 필드명: type 또는 violation_type 사용
    const groupedByDate = fullLogs.reduce((acc, log) => {
      // timestamp 처리: 숫자(밀리초) 또는 문자열
      let dateStr = "";
      try {
        if (log.timestamp) {
          const ts =
            typeof log.timestamp === "number"
              ? log.timestamp > 1e12
                ? log.timestamp
                : log.timestamp * 1000
              : parseInt(log.timestamp);
          dateStr = new Date(ts).toLocaleDateString("ko-KR");
        } else if (log.violation_datetime) {
          dateStr = new Date(log.violation_datetime).toLocaleDateString(
            "ko-KR"
          );
        }
      } catch (e) {
        console.warn("날짜 파싱 실패:", log, e);
        return acc;
      }

      if (!dateStr) return acc;

      if (!acc[dateStr]) {
        acc[dateStr] = { helmet: 0, vest: 0, fall: 0 };
      }

      const violationType = (
        log.type ||
        log.violation_type ||
        ""
      ).toLowerCase();

      if (
        violationType.includes("안전모") ||
        violationType.includes("helmet") ||
        violationType.includes("헬멧")
      ) {
        acc[dateStr].helmet++;
      }
      if (
        violationType.includes("조끼") ||
        violationType.includes("vest") ||
        violationType.includes("안전조끼")
      ) {
        acc[dateStr].vest++;
      }
      if (
        violationType.includes("낙상") ||
        violationType.includes("fall") ||
        violationType.includes("넘어짐")
      ) {
        acc[dateStr].fall++;
      }

      return acc;
    }, {});

    const chartDataArray = Object.entries(groupedByDate).map(
      ([date, counts]) => ({
        date,
        ...counts,
      })
    );

    setChartData(chartDataArray);
  }, [fullLogs]);

  // 초기 로드 및 자동 갱신
  useEffect(() => {
    loadViolationStats();
    loadViolations();

    // 30초마다 자동 갱신
    const interval = setInterval(() => {
      loadViolationStats();
      loadViolations();
    }, 30000);

    return () => clearInterval(interval);
  }, [loadViolationStats, loadViolations]);

  // 차트 데이터 자동 생성
  useEffect(() => {
    generateChartData();
  }, [generateChartData]);

  return {
    // KPI
    kpiHelmet,
    kpiVest,
    kpiFall,
    totalAlerts,
    pendingCount,
    completedActions,
    safetyScore,

    // 주간 통계
    weeklyTotalDetections,
    weeklyHelmetViolations,
    weeklyFallDetections,

    // 데이터
    eventRows,
    fullLogs,
    chartData,

    // 함수
    loadViolationStats,
    loadViolations,
    updateViolationStatus,
  };
}

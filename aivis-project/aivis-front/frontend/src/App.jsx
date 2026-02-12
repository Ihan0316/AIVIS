import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
} from "react";
import Chart from "chart.js/auto";
import * as XLSX from "xlsx";
import { translations } from "./utils/translations";
import {
  translateRiskType,
  getWeekLabel,
  translateWeatherDescription,
  mapWorkerName,
} from "./utils/utils";
import { api } from "./services/api";
import { generateSummaryPdf } from "./utils/reportPdf";
import "./App.css";

export default function AIVISApp() {
  const urlParams = new URLSearchParams(window.location.search);
  const isCameraMode = urlParams.get("camera") === "true";

  const [activePage, setActivePage] = useState(
    isCameraMode ? "access-camera" : "dashboard"
  );

  const [kpiHelmet, setKpiHelmet] = useState(0);
  const [kpiVest, setKpiVest] = useState(0);
  const [kpiFall, setKpiFall] = useState(0);
  const [totalAlerts, setTotalAlerts] = useState(0);
  const [pendingCount, setPendingCount] = useState(0);
  const [eventRows, setEventRows] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [fullLogs, setFullLogs] = useState([]);
  const [logTab, setLogTab] = useState("table");
  const [tableFilters, setTableFilters] = useState({
    worker: [],
    zone: [],
    status: [],
  });
  const [showFilterDropdown, setShowFilterDropdown] = useState(null);
  const filterDropdownRef = useRef(null);

  const [totalWorkers, setTotalWorkers] = useState(0);
  const [completedActions, setCompletedActions] = useState(0);
  const [safetyScore, setSafetyScore] = useState(0);
  const [weeklyTotalDetections, setWeeklyTotalDetections] = useState(0);
  const [weeklyHelmetViolations, setWeeklyHelmetViolations] = useState(0);
  const [weeklyFallDetections, setWeeklyFallDetections] = useState(0);

  const [facialRecognitionAccuracy] = useState(97.4);
  const [equipmentDetectionAccuracy] = useState(95.2);
  const [behaviorDetectionAccuracy] = useState(91.5);
  const [controlTeamCount] = useState(1);

  const [weather, setWeather] = useState(null);
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [currentTime, setCurrentTime] = useState(new Date());

  const [dense] = useState(true);
  const [focusCam, setFocusCam] = useState(null);
  const [showBottomRightPopup, setShowBottomRightPopup] = useState(null);
  const alarmRef = useRef(null);
  const [showSearch, setShowSearch] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [language] = useState("ko");
  const [theme] = useState("dark");
  const [notificationsEnabled] = useState(true);

  const [workersList, setWorkersList] = useState([]);
  const [loadingWorkers, setLoadingWorkers] = useState(true);
  const [showWorkerModal, setShowWorkerModal] = useState(false);
  const [editingWorker, setEditingWorker] = useState(null);
  // IME 조합 상태 추적 (한글 입력용)
  const isComposingRef = useRef(false);

  const [workerFormData, setWorkerFormData] = useState({
    worker_id: "",
    name: "",
    contact: "",
    team: "",
    role: "worker",
    blood_type: "",
  });
  const [workerIdError, setWorkerIdError] = useState("");
  const [selectedTeamFilter, setSelectedTeamFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchHistory, setSearchHistory] = useState([]);
  const [showSearchHistory, setShowSearchHistory] = useState(false);
  const [todayImages, setTodayImages] = useState([]);
  const [loadingTodayImages, setLoadingTodayImages] = useState(false);
  const [showEditButtons, setShowEditButtons] = useState(false);

  // 커스텀 알림 상태
  const [toast, setToast] = useState(null);
  const [toastExiting, setToastExiting] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [confirmDialogData, setConfirmDialogData] = useState(null);

  // 이미지 모달 상태
  const [showImageModal, setShowImageModal] = useState(false);
  const [selectedImagePath, setSelectedImagePath] = useState(null);

  // 커스텀 알림 함수
  const showToast = useCallback((message, type = "info") => {
    setToastExiting(false);
    setToast({ message, type });
    setTimeout(() => {
      setToastExiting(true);
    }, 3000);
  }, []);

  // 애니메이션 종료 시 toast 제거
  const handleToastAnimationEnd = useCallback(() => {
    if (toastExiting) {
      setToast(null);
      setToastExiting(false);
    }
  }, [toastExiting]);

  // 커스텀 확인 다이얼로그 함수
  const showConfirm = useCallback((message, onConfirm, onCancel = null) => {
    setConfirmDialogData({ message, onConfirm, onCancel });
    setShowConfirmDialog(true);
  }, []);

  // 확인 다이얼로그 확인 처리
  const handleConfirm = useCallback(() => {
    if (confirmDialogData && confirmDialogData.onConfirm) {
      confirmDialogData.onConfirm();
    }
    setShowConfirmDialog(false);
    setConfirmDialogData(null);
  }, [confirmDialogData]);

  // 확인 다이얼로그 취소 처리
  const handleCancel = useCallback(() => {
    if (confirmDialogData && confirmDialogData.onCancel) {
      confirmDialogData.onCancel();
    }
    setShowConfirmDialog(false);
    setConfirmDialogData(null);
  }, [confirmDialogData]);

  useEffect(() => {
    const loadWorkers = async () => {
      setLoadingWorkers(true);
      try {
        const response = await api.getWorkers();

        if (response.success && response.workers) {
          setWorkersList(response.workers);
          setTotalWorkers(
            response.workers.filter((w) => w.role === "worker").length
          );
        } else {
          setWorkersList([]);
        }
      } catch (error) {
        console.error("작업자 목록 로드 오류:", error);
        setWorkersList([]);
      } finally {
        setLoadingWorkers(false);
      }
    };

    loadWorkers();

    // 작업자 목록 실시간 업데이트 (30초마다)
    const workersInterval = setInterval(loadWorkers, 30000);
    return () => clearInterval(workersInterval);
  }, []);

  const highlightText = useCallback((text, query) => {
    if (!query || !text || query.trim() === "") return text;

    const searchQuery = query.trim();
    const regex = new RegExp(
      `(${searchQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`,
      "gi"
    );
    const parts = String(text).split(regex);

    return parts.map((part, index) =>
      regex.test(part) ? (
        <mark
          key={index}
          style={{
            background: "var(--accent-blue)",
            color: "white",
            padding: "2px 4px",
            borderRadius: "3px",
            fontWeight: "600",
          }}
        >
          {part}
        </mark>
      ) : (
        part
      )
    );
  }, []);

  const addToSearchHistory = useCallback((query) => {
    if (!query || query.trim() === "") return;

    const trimmedQuery = query.trim();
    setSearchHistory((prev) => {
      const filtered = prev.filter((item) => item !== trimmedQuery);
      const updated = [trimmedQuery, ...filtered].slice(0, 10);
      return updated;
    });
  }, []);

  const selectFromHistory = useCallback((query) => {
    setSearchQuery(query);
    setShowSearchHistory(false);
  }, []);

  const filteredLogs = useMemo(() => {
    if (!fullLogs.length) return [];

    return fullLogs.filter((row) => {
      if (
        tableFilters.worker.length > 0 &&
        !tableFilters.worker.includes(row.worker)
      ) {
        return false;
      }
      if (
        tableFilters.zone.length > 0 &&
        !tableFilters.zone.includes(row.zone)
      ) {
        return false;
      }
      if (
        tableFilters.status.length > 0 &&
        !tableFilters.status.includes(row.status)
      ) {
        return false;
      }
      return true;
    });
  }, [fullLogs, tableFilters]);

  const filterOptions = useMemo(() => {
    if (!fullLogs.length) return { workers: [], zones: [], statuses: [] };

    const workers = [
      ...new Set(fullLogs.map((row) => row.worker).filter(Boolean)),
    ].sort();
    const zones = [
      ...new Set(fullLogs.map((row) => row.zone).filter(Boolean)),
    ].sort();
    const statuses = ["critical", "normal"];

    return { workers, zones, statuses };
  }, [fullLogs]);

  const toggleFilter = (column, value) => {
    setTableFilters((prev) => {
      const currentFilters = prev[column] || [];
      const newFilters = currentFilters.includes(value)
        ? currentFilters.filter((f) => f !== value)
        : [...currentFilters, value];
      return { ...prev, [column]: newFilters };
    });
  };

  const clearFilter = (column) => {
    setTableFilters((prev) => ({ ...prev, [column]: [] }));
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        filterDropdownRef.current &&
        !filterDropdownRef.current.contains(event.target)
      ) {
        setShowFilterDropdown(null);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const generateWorkerIdByTeam = useCallback(
    (team) => {
      if (!team || !team.trim()) return "";

      const teamLetter = team.replace("팀", "").trim().charAt(0).toUpperCase();

      const teamBaseMap = {
        A: 301,
        B: 401,
        C: 501,
        D: 601,
      };

      const baseNumber = teamBaseMap[teamLetter] || 700;

      const teamWorkers = workersList.filter(
        (w) =>
          w.team === team &&
          (w.workerId || w.worker_id) &&
          /^\d+$/.test(w.workerId || w.worker_id || "")
      );

      let maxNumber = baseNumber - 1;
      teamWorkers.forEach((w) => {
        const num = parseInt(w.workerId || w.worker_id || "0", 10);
        if (num >= baseNumber && num < baseNumber + 100 && num > maxNumber) {
          maxNumber = num;
        }
      });

      const nextNumber = maxNumber + 1;

      if (nextNumber >= baseNumber + 100) {
        return String(baseNumber + 99);
      }

      return String(nextNumber);
    },
    [workersList]
  );

  const formatViolationEvent = useCallback(
    (v) => {
      if (!v) return null;

      // DB 필드명 매핑: cam_id 또는 camera_id 사용
      const cam_id =
        v.cam_id !== undefined
          ? v.cam_id
          : v.camera_id !== undefined
          ? v.camera_id
          : 0;

      // zone 추출 (work_zone 우선, 없으면 cam_id 기반)
      let zone = v.work_zone || "";
      if (!zone) {
        const area_map = { 0: "A", 1: "B", 2: "C", 3: "D" };
        zone = area_map[cam_id] || `A-${cam_id + 1}`;
      }
      // zone에서 첫 글자만 추출 (A-1 -> A)
      const zoneLetter = zone.charAt(0).toUpperCase();
      const teamName = `${zoneLetter}팀`;

      // 관리자 찾기 (team 우선, 없으면 zone 기반)
      const team = v.worker_team || teamName;
      let manager = (workersList || []).find(
        (w) => w && w.team === team && w.role === "manager"
      );

      // team으로 찾지 못했으면 zone 기반으로 다시 찾기
      if (!manager) {
        manager = (workersList || []).find((w) => {
          if (!w || w.role !== "manager") return false;
          const wTeam = w.team || "";
          const wZone = wTeam.replace("팀", "").trim();
          return wZone === zoneLetter;
        });
      }

      const managerName = manager
        ? manager.workerName || manager.name || ""
        : "";
      const managerZone =
        manager && manager.team
          ? manager.team.replace("팀", "").trim()
          : zoneLetter;

      // timestamp 처리: DB에서 숫자(밀리초)로 오는 경우 처리
      const formatTimestamp = (ts) => {
        if (!ts) return "";
        try {
          if (typeof ts === "number") {
            // 밀리초인지 초인지 판단
            const timestamp = ts > 1e12 ? ts : ts * 1000;
            return new Date(timestamp)
              .toISOString()
              .replace("T", " ")
              .slice(0, 19);
          } else if (typeof ts === "string") {
            // 문자열인 경우 숫자로 변환 시도
            const num = parseInt(ts);
            if (!isNaN(num)) {
              const timestamp = num > 1e12 ? num : num * 1000;
              return new Date(timestamp)
                .toISOString()
                .replace("T", " ")
                .slice(0, 19);
            }
            // 이미 날짜 문자열인 경우
            return ts.replace("T", " ").slice(0, 19);
          }
        } catch (e) {
          console.warn("timestamp 포맷팅 실패:", ts, e);
        }
        return "";
      };

      // 원본 violation_datetime 보존 (API 호출 시 사용)
      // 백엔드에서 반환한 원본 값을 그대로 사용해야 DB 조회가 성공함
      const originalViolationDatetime = v.violation_datetime
        ? String(v.violation_datetime).trim()
        : "";

      // 표시용 datetime 변환
      let datetimeStr = "";
      if (originalViolationDatetime) {
        // 이미 올바른 형식인지 확인
        if (
          /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(
            originalViolationDatetime
          )
        ) {
          // 이미 올바른 형식
          datetimeStr = originalViolationDatetime;
        } else {
          // ISO 형식이나 다른 형식인 경우 변환 (표시용)
          datetimeStr = originalViolationDatetime
            .replace("T", " ")
            .slice(0, 19);
        }
      } else if (v.timestamp) {
        datetimeStr = formatTimestamp(v.timestamp);
      }

      // 필수 필드 검증 및 정규화
      // worker_id는 문자열로 변환 (백엔드에서 문자열로 비교)
      const workerId = v.worker_id ? String(v.worker_id).trim() : "";

      if (!workerId || !originalViolationDatetime) {
        console.warn("[formatViolationEvent] 필수 필드 누락:", {
          worker_id: workerId,
          violation_datetime: originalViolationDatetime,
          original_data: v,
        });
      }

      return {
        worker: mapWorkerName(v.worker_name || v.worker_id || ""),
        risk: v.type || v.violation_type || "",
        manager: managerName || teamName,
        managerTeam: manager ? manager.team : teamName,
        status: v.status || "new",
        datetime: datetimeStr, // 표시용
        zone: managerZone || zoneLetter,
        worker_id: workerId,
        violation_datetime: originalViolationDatetime, // API 호출용 원본 값 (백엔드에서 반환한 그대로)
      };
    },
    [workersList]
  );

  // 리포트 페이지에서 사용할 수 있도록 loadViolationsAndStats 함수를 외부로 분리
  const loadViolationsAndStats = useCallback(
    async (limit = 100, days = null, statsDays = null, page = null) => {
      // page 파라미터가 없으면 현재 activePage 사용
      const currentPage = page || activePage;
      // limit이 null이면 모든 데이터 로드
      // days가 null이면 기본값 사용 (백엔드에서 7일)
      // statsDays가 null이면 기본값 사용 (백엔드에서 7일)
      try {
        console.log(
          "[데이터 로드] API 호출 시작, limit:",
          limit,
          "days:",
          days,
          "statsDays:",
          statsDays
        );
        const violationsResponse = await api.getViolations(limit, null, days);
        console.log("[데이터 로드] API 응답:", {
          success: violationsResponse.success,
          violationsCount: violationsResponse.violations?.length || 0,
          violations: violationsResponse.violations?.slice(0, 3) || [],
        });

        if (violationsResponse.success && violationsResponse.violations) {
          // 실시간 위험 이벤트 로그: 모든 "new" 상태 이벤트 표시 (제한 없음)
          const formattedEvents = violationsResponse.violations
            .filter((v) => v.status === "new" || !v.status)
            .map((v) => {
              // 원본 데이터 먼저 보존 (formatViolationEvent 호출 전)
              const originalData = {
                worker_id: v.worker_id,
                violation_datetime: v.violation_datetime,
              };

              // 디버깅: 원본 데이터 확인
              if (v.status === "new" || !v.status) {
                console.log("[데이터 로드] 원본 위반 데이터:", {
                  worker_id: originalData.worker_id,
                  violation_datetime: originalData.violation_datetime,
                  violation_datetime_type:
                    typeof originalData.violation_datetime,
                  violation_datetime_length:
                    originalData.violation_datetime?.length,
                });
              }

              const formatted = formatViolationEvent(v);
              if (formatted) {
                // 원본 데이터 보존 (API 호출 시 DB에 저장된 형식 그대로 사용)
                formatted._original = originalData;
              }
              return formatted;
            })
            .filter((event) => event !== null);
          setEventRows(formattedEvents);
          // pendingCount는 일일 안전 점수 계산 부분에서 KPI 항목 기준으로 설정됨
          // 여기서는 임시로 설정하지 않음 (아래 일일 안전 점수 계산에서 설정)

          const formattedLogs = violationsResponse.violations
            .map((v) => {
              let processingTimeStr = null;
              let completedTimeStr = null;

              // violation_datetime 또는 timestamp에서 날짜 문자열 추출
              const violationDateStr = v.violation_datetime
                ? v.violation_datetime.replace("T", " ").trim()
                : v.timestamp
                ? new Date(v.timestamp)
                    .toISOString()
                    .replace("T", " ")
                    .slice(0, 19)
                : "";

              if (
                v.processing_time !== null &&
                v.processing_time !== undefined &&
                v.processing_time > 0 &&
                violationDateStr
              ) {
                const totalSeconds = parseInt(v.processing_time);
                const hours = Math.floor(totalSeconds / 3600);
                const minutes = Math.floor((totalSeconds % 3600) / 60);
                const seconds = totalSeconds % 60;

                if (hours > 0) {
                  processingTimeStr = `${hours}시간 ${minutes}분`;
                } else if (minutes > 0) {
                  processingTimeStr = `${minutes}분 ${seconds}초`;
                } else {
                  processingTimeStr = `${seconds}초`;
                }

                try {
                  const [datePart, timePart] = violationDateStr.split(" ");

                  if (datePart && timePart) {
                    const [year, month, day] = datePart.split("-").map(Number);
                    const [hours, minutes, seconds] = timePart
                      .split(":")
                      .map(Number);

                    const violationDate = new Date(
                      year,
                      month - 1,
                      day,
                      hours,
                      minutes,
                      seconds || 0
                    );

                    if (!isNaN(violationDate.getTime())) {
                      const completedDate = new Date(
                        violationDate.getTime() + totalSeconds * 1000
                      );

                      const completedYear = completedDate.getFullYear();
                      const completedMonth = String(
                        completedDate.getMonth() + 1
                      ).padStart(2, "0");
                      const completedDay = String(
                        completedDate.getDate()
                      ).padStart(2, "0");
                      const completedHours = String(
                        completedDate.getHours()
                      ).padStart(2, "0");
                      const completedMinutes = String(
                        completedDate.getMinutes()
                      ).padStart(2, "0");
                      const completedSeconds = String(
                        completedDate.getSeconds()
                      ).padStart(2, "0");

                      completedTimeStr = `${completedYear}-${completedMonth}-${completedDay} ${completedHours}:${completedMinutes}:${completedSeconds}`;
                    }
                  }
                } catch {
                  // 처리 완료 시간 계산 실패 시 무시
                }
              }

              // zone 추출: work_zone 우선, 없으면 cam_id 또는 camera_id 기반
              const cam_id =
                v.cam_id !== undefined
                  ? v.cam_id
                  : v.camera_id !== undefined
                  ? v.camera_id
                  : 0;
              const zone =
                v.work_zone ||
                (cam_id !== undefined
                  ? ["A", "B", "C", "D"][cam_id] || ""
                  : "");

              // time 필드가 없으면 timestamp나 violation_datetime에서 생성
              let finalTime = violationDateStr;
              if (!finalTime || finalTime === "") {
                // timestamp가 있으면 사용
                if (v.timestamp) {
                  try {
                    const ts =
                      typeof v.timestamp === "number"
                        ? v.timestamp
                        : parseInt(v.timestamp);
                    const timestamp = ts > 1e12 ? ts : ts * 1000;
                    finalTime = new Date(timestamp)
                      .toISOString()
                      .replace("T", " ")
                      .slice(0, 19);
                  } catch (e) {
                    console.warn(
                      "[데이터 로드] timestamp 변환 실패:",
                      v.timestamp,
                      e
                    );
                    finalTime = new Date()
                      .toISOString()
                      .replace("T", " ")
                      .slice(0, 19);
                  }
                } else {
                  // 모든 필드가 없으면 현재 시간 사용 (데이터 손실 방지)
                  finalTime = new Date()
                    .toISOString()
                    .replace("T", " ")
                    .slice(0, 19);
                }
              }

              return {
                worker: mapWorkerName(v.worker_name || v.worker_id || ""),
                risk: v.type || v.violation_type || "",
                time: finalTime,
                zone: zone || "N/A",
                processing_time: completedTimeStr || processingTimeStr || "",
                status: v.status === "done" ? "normal" : "critical",
                image_path: v.image_path || "", // 이미지 경로 추가
                _original: v, // 원본 데이터 보존
              };
            })
            // time 필터링 제거 - 모든 데이터 포함 (time이 없어도 기본값 사용)
            .sort((a, b) => {
              // time으로 정렬 (최신순)
              if (!a.time || !b.time) return 0;
              return a.time < b.time ? 1 : -1;
            });

          console.log("[데이터 로드] 포맷팅된 로그:", {
            formattedLogsCount: formattedLogs.length,
            originalCount: violationsResponse.violations?.length || 0,
            sample: formattedLogs.slice(0, 3),
          });
          setFullLogs(formattedLogs);
        } else {
          console.warn(
            "[데이터 로드] API 응답에 violations가 없음:",
            violationsResponse
          );
        }

        console.log("[데이터 로드] 통계 API 호출 시작");
        // 대시보드인 경우: 실시간 현장 요약 KPI는 하루치, 주간 위험 통계는 7주(49일) 데이터
        // 리포트/로그: 전체 통계 (statsDays=null, 백엔드 기본값 7일)
        if (currentPage === "dashboard") {
          // 실시간 현장 요약 KPI: 하루치 통계
          const todayStatsResponse = await api.getViolationStats(1);
          console.log("[데이터 로드] 오늘 통계 API 응답:", {
            success: todayStatsResponse.success,
            kpi: todayStatsResponse.kpi,
          });
          if (todayStatsResponse.success && todayStatsResponse.kpi) {
            setKpiHelmet(todayStatsResponse.kpi.helmet || 0);
            setKpiVest(todayStatsResponse.kpi.vest || 0);
            setKpiFall(todayStatsResponse.kpi.fall || 0);

            // 금일 총 알림: 일일 안전 점수 계산 부분에서 KPI 항목 기준으로 설정됨
            // 여기서는 임시로 설정하지 않음 (아래 일일 안전 점수 계산에서 설정)
          }

          // 주간 위험 통계 차트: 7주(49일) 데이터
          const weeklyStatsResponse = await api.getViolationStats(49);
          console.log("[데이터 로드] 주간 통계 API 응답:", {
            success: weeklyStatsResponse.success,
            chartDataCount: weeklyStatsResponse.chart_data?.length || 0,
          });
          if (weeklyStatsResponse.success && weeklyStatsResponse.chart_data) {
            setChartData(weeklyStatsResponse.chart_data);

            // 대시보드: 주간 통계 계산
            const totalWeekly = weeklyStatsResponse.chart_data.reduce(
              (sum, week) => sum + (week.violations || 0),
              0
            );
            setWeeklyTotalDetections(totalWeekly);

            const totalHelmet = weeklyStatsResponse.chart_data.reduce(
              (sum, week) => sum + (week.helmet_violations || 0),
              0
            );
            const totalFall = weeklyStatsResponse.chart_data.reduce(
              (sum, week) => sum + (week.fall_detections || 0),
              0
            );
            setWeeklyHelmetViolations(
              totalHelmet || weeklyStatsResponse.kpi?.helmet || 0
            );
            setWeeklyFallDetections(
              totalFall || weeklyStatsResponse.kpi?.fall || 0
            );
          }

          // 일일 안전 점수 계산: 금일 총 알림(KPI 합계) 사용
          if (violationsResponse.success && violationsResponse.violations) {
            // KPI에 해당하는 위반 사항만 필터링 (안전모, 안전조끼, 넘어짐)
            const kpiViolations = violationsResponse.violations.filter((v) => {
              const type = (v.type || v.violation_type || "").toLowerCase();
              return (
                type.includes("안전모") ||
                type.includes("헬멧") ||
                type.includes("helmet") ||
                type.includes("hardhat") ||
                type.includes("안전조끼") ||
                type.includes("조끼") ||
                type.includes("vest") ||
                type.includes("reflective") ||
                type.includes("넘어짐") ||
                type.includes("낙상") ||
                type.includes("fall")
              );
            });

            const completed = kpiViolations.filter(
              (v) => v.status === "done"
            ).length;
            setCompletedActions(completed);

            // 금일 총 알림(KPI 합계)을 total로 사용
            const kpiTotal =
              (todayStatsResponse.kpi?.helmet || 0) +
              (todayStatsResponse.kpi?.vest || 0) +
              (todayStatsResponse.kpi?.fall || 0);
            const total = kpiTotal;
            const score =
              total > 0 ? Math.round((completed / total) * 100) : 100;
            setSafetyScore(score);

            // 확인 필요: KPI 항목 중 status !== "done"인 항목 수 (일일 안전 점수 계산 방식과 일치)
            const pendingKpiItems = kpiTotal - completed;
            setPendingCount(pendingKpiItems);
            // 금일 총 알림도 확인 필요한 KPI 항목 수로 설정
            setTotalAlerts(pendingKpiItems);
          }
        } else {
          // 리포트/로그: 전체 통계
          const statsResponse = await api.getViolationStats(statsDays);
          console.log("[데이터 로드] 통계 API 응답:", {
            success: statsResponse.success,
            kpi: statsResponse.kpi,
            chartDataCount: statsResponse.chart_data?.length || 0,
          });
          if (statsResponse.success && statsResponse.kpi) {
            setKpiHelmet(statsResponse.kpi.helmet || 0);
            setKpiVest(statsResponse.kpi.vest || 0);
            setKpiFall(statsResponse.kpi.fall || 0);
          }

          if (statsResponse.chart_data) {
            setChartData(statsResponse.chart_data);

            // 리포트/로그: 주간 통계 계산
            const totalWeekly = statsResponse.chart_data.reduce(
              (sum, week) => sum + (week.violations || 0),
              0
            );
            setWeeklyTotalDetections(totalWeekly);

            const totalHelmet = statsResponse.chart_data.reduce(
              (sum, week) => sum + (week.helmet_violations || 0),
              0
            );
            const totalFall = statsResponse.chart_data.reduce(
              (sum, week) => sum + (week.fall_detections || 0),
              0
            );
            setWeeklyHelmetViolations(
              totalHelmet || statsResponse.kpi.helmet || 0
            );
            setWeeklyFallDetections(totalFall || statsResponse.kpi.fall || 0);
          }

          // 금일 총 알림: 다른 페이지에서는 기존 통계의 total 사용
          const kpiTotal = statsResponse.kpi.total || 0;
          setTotalAlerts(kpiTotal);

          // 일일 안전 점수 계산: 금일 총 알림(KPI 합계) 사용
          if (violationsResponse.success && violationsResponse.violations) {
            // KPI에 해당하는 위반 사항만 필터링 (안전모, 안전조끼, 넘어짐)
            const kpiViolations = violationsResponse.violations.filter((v) => {
              const type = (v.type || v.violation_type || "").toLowerCase();
              return (
                type.includes("안전모") ||
                type.includes("헬멧") ||
                type.includes("helmet") ||
                type.includes("hardhat") ||
                type.includes("안전조끼") ||
                type.includes("조끼") ||
                type.includes("vest") ||
                type.includes("reflective") ||
                type.includes("넘어짐") ||
                type.includes("낙상") ||
                type.includes("fall")
              );
            });

            const completed = kpiViolations.filter(
              (v) => v.status === "done"
            ).length;
            setCompletedActions(completed);

            // 금일 총 알림(KPI 합계)을 total로 사용
            const total = kpiTotal;
            const score =
              total > 0 ? Math.round((completed / total) * 100) : 100;
            setSafetyScore(score);

            // 확인 필요: KPI 항목 중 status !== "done"인 항목 수 (일일 안전 점수 계산 방식과 일치)
            const pendingKpiItems = kpiTotal - completed;
            setPendingCount(pendingKpiItems);
            // 금일 총 알림도 확인 필요한 KPI 항목 수로 설정
            setTotalAlerts(pendingKpiItems);
          }
        }
      } catch (error) {
        console.error("[데이터 로드] 위반 사항 및 통계 로드 오류:", error);
        console.error("[데이터 로드] 오류 상세:", {
          message: error.message,
          stack: error.stack,
        });
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [formatViolationEvent] // activePage는 파라미터로 전달하므로 의존성에서 제외
  );

  // 초기 로드 및 activePage 변경 시 데이터 로드 (대시보드: 하루치 데이터)
  useEffect(() => {
    if (activePage === "dashboard") {
      // 대시보드: 하루치 데이터 전체 (limit=null, days=1), 주간 통계는 7주(49일)
      loadViolationsAndStats(null, 1, 49, "dashboard");
    } else {
      // 기본값 (7일)
      loadViolationsAndStats(100, null, null, activePage);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePage]); // loadViolationsAndStats는 useCallback으로 안정화되어 의존성에서 제외 (중복 실행 방지)

  // 리포트 페이지 및 이벤트 로그 탭 활성화 시 데이터 다시 로드
  useEffect(() => {
    if (activePage === "calendar") {
      console.log(
        `[데이터 로드] ${activePage} 페이지 활성화, 데이터 로드 시작`
      );
      // 리포트 페이지: 3개월치 데이터 (limit=null, days=90), 통계는 기본값
      loadViolationsAndStats(null, 90, null, "calendar")
        .then(() => {
          console.log(`[데이터 로드] ${activePage} 페이지 데이터 로드 완료`);
        })
        .catch((error) => {
          console.error(
            `[데이터 로드] ${activePage} 페이지 데이터 로드 실패:`,
            error
          );
        });
    } else if (activePage === "logs") {
      console.log(
        `[데이터 로드] ${activePage} 페이지 활성화, 데이터 로드 시작`
      );
      // 이벤트 로그 탭: 지난 30일 (days=30), 통계는 기본값
      loadViolationsAndStats(null, 30, null, "logs")
        .then(() => {
          console.log(`[데이터 로드] ${activePage} 페이지 데이터 로드 완료`);
        })
        .catch((error) => {
          console.error(
            `[데이터 로드] ${activePage} 페이지 데이터 로드 실패:`,
            error
          );
        });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePage]); // loadViolationsAndStats는 useCallback으로 안정화되어 의존성에서 제외 (중복 실행 방지)

  // 이벤트 로그 페이지 실시간 업데이트 (15초마다)
  useEffect(() => {
    if (activePage !== "logs") return;

    const loadLogsData = async () => {
      try {
        await loadViolationsAndStats(null, 30, null, "logs");
      } catch (error) {
        console.error("[이벤트 로그 실시간 업데이트] 오류:", error);
      }
    };

    // 초기 로드는 위의 useEffect에서 처리되므로 여기서는 주기적 갱신만
    const logsInterval = setInterval(loadLogsData, 15000);
    return () => clearInterval(logsInterval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePage]); // loadViolationsAndStats는 useCallback으로 안정화되어 의존성에서 제외 (중복 실행 방지)

  // 리포트 페이지 실시간 업데이트 (30초마다)
  useEffect(() => {
    if (activePage !== "calendar") return;

    const loadCalendarData = async () => {
      try {
        await loadViolationsAndStats(null, 90, null, "calendar");
      } catch (error) {
        console.error("[리포트 실시간 업데이트] 오류:", error);
      }
    };

    // 초기 로드는 위의 useEffect에서 처리되므로 여기서는 주기적 갱신만
    const calendarInterval = setInterval(loadCalendarData, 30000);
    return () => clearInterval(calendarInterval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePage]); // loadViolationsAndStats는 useCallback으로 안정화되어 의존성에서 제외 (중복 실행 방지)

  const handleStartSession = async (overrideWorkerCode = null) => {
    const codeToUse = overrideWorkerCode || workerCode;
    const codeString = codeToUse ? String(codeToUse).trim() : "";

    // 에러 메시지 초기화
    setWorkerCodeError("");

    if (!codeString) {
      setWorkerCodeError("작업자 ID를 입력해주세요.");
      return;
    }

    // 작업자 ID가 등록되어 있는지 확인
    const workerExists = workersList.some(
      (w) => (w.workerId || w.worker_id) === codeString
    );
    if (!workerExists) {
      setWorkerCodeError("등록 후 다시 시도해주세요.");
      return;
    }

    setWorkerCode(codeString);
    workerCodeRef.current = codeString;

    setCaptureStep("front");
    capturedStepsRef.current.clear();
    setFaceInGuide(false);
    setCountdown(0);
    countdownCountRef.current = 0;
    lastMessageRef.current = "";
    setCaptureMessage("얼굴을 가이드에 맞춰주세요");
    countdownWaitingRef.current = false;
    lastCaptureTimeRef.current = 0; // 첫 촬영 딜레이 제거

    // UUID v7 형식의 세션 ID 생성 (간단한 구현)
    const generateSessionId = () => {
      const now = Date.now();
      const random = Math.random().toString(36).substring(2, 15);
      return `${now.toString(36)}-${random}`;
    };

    const newSessionId = generateSessionId();
    setSessionId(newSessionId);

    // 작업자 입력 UI 숨김
    setShowWorkerInput(false);

    // 카메라 시작 (showAccessCamera를 true로 설정하면 useEffect가 자동으로 카메라를 시작함)
    setShowAccessCamera(true);

    // 카메라가 준비될 때까지 기다림
    const waitForCamera = async () => {
      let retries = 0;
      const maxRetries = 10; // 최대 1초 대기 (100ms * 10)

      while (retries < maxRetries) {
        const video = accessCameraVideoRef.current;

        // 비디오가 준비되었는지 확인
        if (video && video.srcObject) {
          const stream = video.srcObject;
          const videoTracks = stream.getVideoTracks();

          // 비디오와 스트림이 모두 준비되었는지 확인
          if (
            videoTracks.length > 0 &&
            videoTracks[0].readyState === "live" &&
            video.readyState >= video.HAVE_CURRENT_DATA &&
            video.videoWidth > 0 &&
            video.videoHeight > 0 &&
            !video.paused
          ) {
            // 추가로 실제 프레임이 렌더링되는지 확인
            await new Promise((resolve) => requestAnimationFrame(resolve));
            if (video.videoWidth > 0 && video.videoHeight > 0) {
              // 카메라 준비 완료
              cameraReadyRef.current = true;
              // 작업자 입력 UI 숨김 (이제 스트림이 안정적으로 유지됨)
              setShowWorkerInput(false);
              return;
            }
          }
        }

        await new Promise((resolve) => setTimeout(resolve, 100));
        retries++;
      }

      // 1초 내에 준비되지 않아도 작업자 입력 UI 숨김 (카메라는 백그라운드에서 계속 준비됨)
      cameraReadyRef.current = true;
      setShowWorkerInput(false);
    };

    await waitForCamera();
  };

  // 상세보기 Canvas ref (바운딩 박스 그리기용)
  const focusCanvasRef = useRef(null);
  const focusCanvasRef2 = useRef(null);
  const focusImageRef = useRef(null);
  const focusImageRef2 = useRef(null);

  // 바운딩 박스 데이터 (WebSocket에서 받아옴)
  const [boundingBoxData, setBoundingBoxData] = useState({
    0: {
      violations: [],
      recognized_faces: [],
      normal_detections: [],
      original_width: 1920,
      original_height: 1080,
    },
    1: {
      violations: [],
      recognized_faces: [],
      normal_detections: [],
      original_width: 1920,
      original_height: 1080,
    },
  });

  // WebSocket 연결
  const [wsConnections, setWsConnections] = useState({
    0: null,
    1: null,
    2: null,
    3: null,
  });

  // 대시보드 서버 스트림용 상태
  const [cameraRefreshKey, setCameraRefreshKey] = useState(0);
  const [streamErrors, setStreamErrors] = useState({
    0: { hasError: false, retryCount: 0, isSslError: false },
    1: { hasError: false, retryCount: 0, isSslError: false },
    2: { hasError: false, retryCount: 0, isSslError: false },
    3: { hasError: false, retryCount: 0, isSslError: false },
  });
  const streamRetryTimers = useRef({ 0: null, 1: null, 2: null, 3: null });

  // 카메라 목록
  const [selectedCamera1, setSelectedCamera1] = useState(null);
  const [selectedCamera2, setSelectedCamera2] = useState(null);

  // 카메라 상태 (출입용 - 새 창 모드)
  const [accessCameraStream, setAccessCameraStream] = useState(null);
  const accessCameraVideoRef = useRef(null);
  const accessCameraCanvasRef = useRef(null);
  const [showAccessCamera, setShowAccessCamera] = useState(false);
  const [faceInGuide, setFaceInGuide] = useState(false);
  const [captureMessage, setCaptureMessage] = useState("");
  const [isAutoCapturing, setIsAutoCapturing] = useState(false);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [faceApiLoaded, setFaceApiLoaded] = useState(false);
  const [countdown, setCountdown] = useState(0); // 정면 촬영 카운트다운 타이머
  const faceDetectionIntervalRef = useRef(null);
  const lastCaptureTimeRef = useRef(0); // 마지막 촬영 시간 (useRef로 관리)
  const lastMessageRef = useRef(""); // 마지막 메시지 (깜빡임 방지용)
  const pauseFaceDetectionRef = useRef(false); // 얼굴 감지 일시 중지 플래그 (촬영 후 딜레이용)
  const capturedStepsRef = useRef(new Set()); // 촬영 완료된 단계 추적 (반복 촬영 방지)
  const cameraReadyRef = useRef(false); // 비디오 메타데이터 준비 여부
  const captureMessageRef = useRef("");
  const countdownTimerRef = useRef(null); // 카운트다운 타이머 ref
  const countdownCountRef = useRef(0); // 카운트다운 카운터 ref (클로저 문제 해결)
  const captureStepRef = useRef("front"); // 현재 촬영 단계 ref (클로저 문제 해결)
  const startFaceDetectionRef = useRef(null); // startFaceDetection 함수 ref (외부 호출용)
  const debugInfoRef = useRef({ logs: [], lastUpdate: Date.now() }); // 디버그 정보 저장
  const countdownWaitingRef = useRef(false);
  const warmupCheckIntervalRef = useRef(null); // 준비 상태 확인 인터벌

  // 다중 촬영 상태 (정면, 왼쪽, 오른쪽)
  const [captureStep, setCaptureStep] = useState("front"); // 'front', 'left', 'right', 'complete'
  const debugLogsRef = useRef([]); // 디버그 로그 배열 (최근 10개 유지)

  // 작업자 코드 및 세션 관리
  const [workerCode, setWorkerCode] = useState("");
  const workerCodeRef = useRef(""); // workerCode의 최신 값을 보장하기 위한 ref
  const [workerName, setWorkerName] = useState(""); // 선택한 작업자 이름
  const [workerCodeError, setWorkerCodeError] = useState(""); // 작업자 ID 에러 메시지
  const [sessionId, setSessionId] = useState(null);
  const [showWorkerInput, setShowWorkerInput] = useState(true); // 작업자 입력 UI 표시 여부

  // workerCode가 변경될 때마다 ref 업데이트
  useEffect(() => {
    // workerCode가 문자열인지 확인하고, 객체인 경우 workerId 추출
    if (typeof workerCode === "string") {
      workerCodeRef.current = workerCode;
    } else if (
      workerCode &&
      typeof workerCode === "object" &&
      workerCode.workerId
    ) {
      workerCodeRef.current = String(workerCode.workerId);
    } else if (workerCode && typeof workerCode === "object") {
      // 객체이지만 workerId가 없는 경우 경고
      workerCodeRef.current = "";
    } else {
      workerCodeRef.current = "";
    }
  }, [workerCode]);

  // 구역별 색상 매핑
  const getZoneColor = (zone) => {
    if (!zone) return "var(--text-secondary)";
    const zoneLetter = zone.charAt(0).toUpperCase();
    const zoneColorMap = {
      A: "#3B82F6", // 파란색
      B: "#EF4444", // 빨간색
      C: "#F59E0B", // 노란색
    };
    return zoneColorMap[zoneLetter] || "var(--text-secondary)";
  };

  // 구역 글자 추출
  const getZoneLetter = (zone) => {
    if (!zone) return "?";
    const zoneLetter = zone.charAt(0).toUpperCase();
    return zoneLetter;
  };

  // 위험 유형별 색상 매핑
  const getRiskTypeColor = (risk) => {
    if (!risk) return "transparent";
    // 넘어짐 감지: 빨간색
    if (risk === "넘어짐 감지" || risk === "Fall Detection") {
      return "#EF4444"; // 빨간색
    }
    // PPE 미착용 (안전모 미착용, 안전조끼 미착용): 노란색
    if (
      risk === "안전모 미착용" ||
      risk === "안전조끼 미착용" ||
      risk === "Unworn Safety Helmet" ||
      risk === "Unworn Safety Vest"
    ) {
      return "#F59E0B"; // 노란색
    }
    return "transparent";
  };

  // workers, riskTypes, zones는 이제 DB에서 로드됨

  // 활성화된 위험 구역 추출 (실시간 이벤트 로그에서)
  const activeAlertZones = useMemo(() => {
    const zones = new Set();
    eventRows.forEach((event) => {
      if (event.zone) {
        // zone에서 첫 글자만 추출 (예: "A", "A-1 구역" → "A")
        const zoneLetter = event.zone.charAt(0).toUpperCase();
        if (["A", "B", "C"].includes(zoneLetter)) {
          zones.add(zoneLetter);
        }
      }
    });
    return Array.from(zones);
  }, [eventRows]);

  const t = translations[language];

  // Charts
  const gaugeRef = useRef(null);
  const barRef = useRef(null);
  const gaugeChartRef = useRef(null);
  const barChartRef = useRef(null);

  // 프론트엔드에서 카메라 직접 연결 코드 제거 (백엔드 MJPEG 스트림 사용)
  // startWebcam, startWebcam2 관련 useEffect 제거됨

  // WebSocket 연결 및 바운딩 박스 데이터 수신
  useEffect(() => {
    if (
      !focusCam ||
      (focusCam.type !== "webcam" &&
        focusCam.type !== "webcam2" &&
        focusCam.type !== "webcam3" &&
        focusCam.type !== "webcam4")
    ) {
      // WebSocket 연결 해제 (연결이 있는 경우만)
      // 상태 업데이트를 조건부로 하여 무한 루프 방지
      setWsConnections((prev) => {
        let hasActiveConnection = false;
        Object.values(prev).forEach((ws) => {
          if (ws) {
            hasActiveConnection = true;
            ws.close();
          }
        });
        return hasActiveConnection
          ? { 0: null, 1: null, 2: null, 3: null }
          : prev;
      });
      return;
    }

    const camId =
      focusCam.type === "webcam"
        ? 0
        : focusCam.type === "webcam2"
        ? 1
        : focusCam.type === "webcam3"
        ? 2
        : 3;

    // 이미 연결된 상태면 건너뛰기 (무한 재연결 방지)
    if (wsConnections[camId]) return;

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    // 백엔드 포트(8081)로 고정 (Vite 포트 5173 아님)
    const wsUrl = `${wsProtocol}//${window.location.hostname}:8081/ws?cam_id=${camId}`;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log(`WebSocket 연결 성공 (CAM-${camId})`);
        // 연결 성공 시에만 상태 업데이트
        setWsConnections((prev) => ({ ...prev, [camId]: ws }));
      };

      ws.onmessage = (event) => {
        try {
          const rawData = JSON.parse(event.data);

          // 메시지 타입에 따라 데이터 추출
          let data = rawData;
          if (rawData.type === "model_results" && rawData.result) {
            // main.py의 broadcast_worker_results 형식
            data = rawData.result;
          } else if (rawData.type === "ai_result") {
            // camera_worker.py의 broadcast_to_websockets 형식
            data = rawData;
            // faces를 recognized_faces로 변환
            if (data.faces && !data.recognized_faces) {
              data.recognized_faces = data.faces;
            }
          }

          // violations 데이터 처리: person_box를 bbox로 변환
          const violations = (data.violations || []).map((v) => ({
            ...v,
            bbox: v.bbox || v.person_box || v.box, // person_box를 bbox로 매핑
            box: v.box || v.person_box || v.bbox,
          }));

          const recognized_faces = data.recognized_faces || [];
          const normal_detections =
            data.normal_detections || data.detected_workers || [];

          // frame_width/frame_height 또는 original_width/original_height 사용
          const original_width =
            data.frame_width || data.original_width || 1920;
          const original_height =
            data.frame_height || data.original_height || 1080;

          // 디버깅: 바운딩 박스 데이터 수신 확인
          if (violations.length > 0 || recognized_faces.length > 0) {
            console.log(`[바운딩박스] CAM-${camId} 데이터 수신:`, {
              violations: violations.length,
              recognized_faces: recognized_faces.length,
              normal_detections: normal_detections.length,
              original_width,
              original_height,
            });
          }

          setBoundingBoxData((prev) => ({
            ...prev,
            [camId]: {
              violations,
              recognized_faces,
              normal_detections: Array.isArray(normal_detections)
                ? normal_detections
                : [],
              original_width,
              original_height,
            },
          }));
        } catch (err) {
          console.error("WebSocket 데이터 파싱 오류:", err);
        }
      };

      ws.onerror = (error) => {
        console.error(`WebSocket 오류 (CAM-${camId}):`, error);
      };

      ws.onclose = () => {
        console.log(`WebSocket 연결 종료 (CAM-${camId})`);
        setWsConnections((prev) => ({ ...prev, [camId]: null }));
      };

      return () => {
        // 언마운트 시 연결 해제
        if (ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      };
    } catch (err) {
      console.error(`WebSocket 연결 실패 (CAM-${camId}):`, err);
    }
    // wsConnections를 의존성 배열에서 제거하여 무한 재실행 방지
    // focusCam이 변경될 때만 실행되도록 함
  }, [focusCam]);

  // 바운딩 박스 그리기 (Canvas) - 비활성화됨: 백엔드에서 이미 그려진 프레임을 사용
  // 백엔드의 render_frame_results에서 바운딩 박스를 그린 프레임을 MJPEG 스트림으로 전송하므로
  // 프론트엔드에서 별도로 그릴 필요가 없음
  useEffect(() => {
    // 백엔드에서 이미 바운딩 박스를 그린 프레임을 사용하므로 프론트엔드 Canvas 렌더링 비활성화
    return;

    // 아래 코드는 비활성화됨 (백업용으로 유지)
    /*
    if (
      !focusCam ||
      (focusCam.type !== "webcam" &&
        focusCam.type !== "webcam2" &&
        focusCam.type !== "webcam3" &&
        focusCam.type !== "webcam4")
    ) {
      return;
    }

    const camId =
      focusCam.type === "webcam"
        ? 0
        : focusCam.type === "webcam2"
        ? 1
        : focusCam.type === "webcam3"
        ? 2
        : 3;
    const canvas =
      camId === 0 ? focusCanvasRef.current : focusCanvasRef2.current;
    const img = camId === 0 ? focusImageRef.current : focusImageRef2.current;

    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId;
    let pulsePhase = 0;

    const drawBoundingBoxes = () => {
      // Canvas 크기 조정
      const container = canvas.parentElement;
      if (container) {
        const rect = container.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
      }

      // Canvas 초기화
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 이미지가 로드되지 않았으면 리턴
      if (!img.complete || img.naturalWidth === 0) {
        animationFrameId = requestAnimationFrame(drawBoundingBoxes);
        return;
      }

      // 이미지 크기 및 위치 계산 (img 태그와 동일하게)
      const imgRect = img.getBoundingClientRect();
      const containerRect = canvas.getBoundingClientRect();

      // 이미지가 실제로 렌더링된 크기와 위치
      const imgAspect = img.naturalWidth / img.naturalHeight;
      const containerAspect = containerRect.width / containerRect.height;

      let drawWidth, drawHeight, drawX, drawY;
      if (imgAspect > containerAspect) {
        drawWidth = containerRect.width;
        drawHeight = containerRect.width / imgAspect;
        drawX = 0;
        drawY = (containerRect.height - drawHeight) / 2;
      } else {
        drawWidth = containerRect.height * imgAspect;
        drawHeight = containerRect.height;
        drawX = (containerRect.width - drawWidth) / 2;
        drawY = 0;
      }

      // 좌표 변환 스케일 계산
      const scaleX = drawWidth / boundingBoxData[camId].original_width;
      const scaleY = drawHeight / boundingBoxData[camId].original_height;

      // Pulse 애니메이션 (위반 박스용)
      pulsePhase += 0.05;
      const pulseOpacity = 0.5 + 0.5 * Math.sin(pulsePhase);

      // Corner Bracket 그리기 함수 (세련된 스타일 V2)
      const drawCornerBracket = (
        x,
        y,
        width,
        height,
        color,
        isViolation = false
      ) => {
        const cornerLength = Math.min(width, height) * 0.25;
        const cornerThickness = 2.5; // 두께 조정 (3 -> 2.5)
        const lineOpacity = 0.15; // 배경선 약간 진하게

        ctx.save();

        // 위반 박스는 Pulse 효과 + 강력한 글로우
        if (isViolation) {
          ctx.globalAlpha = pulseOpacity;
          ctx.shadowColor = color;
          ctx.shadowBlur = 20; // 글로우 강화
        } else {
          ctx.globalAlpha = 0.9;
          ctx.shadowColor = color;
          ctx.shadowBlur = 8;
        }

        ctx.strokeStyle = color;
        ctx.lineCap = "square"; // 끝부분을 각지게 (더 기계적인 느낌)
        ctx.lineJoin = "miter";

        // 1. 내부 반투명 채우기 (영역감 부여)
        ctx.fillStyle = color;
        ctx.globalAlpha = isViolation ? 0.15 : 0.05;
        ctx.fillRect(x, y, width, height);

        // 2. 아주 옅은 전체 테두리 (가이드라인)
        ctx.lineWidth = 1;
        ctx.globalAlpha = lineOpacity;
        ctx.strokeRect(x, y, width, height);

        // 3. 모서리 4군데 강조 (코너 브라켓)
        ctx.globalAlpha = isViolation ? pulseOpacity : 1.0;
        ctx.lineWidth = cornerThickness;

        // 상단 좌측 (⌜)
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLength);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLength, y);
        ctx.stroke();

        // 상단 우측 (⌝)
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLength, y);
        ctx.lineTo(x + width, y);
        ctx.lineTo(x + width, y + cornerLength);
        ctx.stroke();

        // 하단 우측 (⌟)
        ctx.beginPath();
        ctx.moveTo(x + width, y + height - cornerLength);
        ctx.lineTo(x + width, y + height);
        ctx.lineTo(x + width - cornerLength, y + height);
        ctx.stroke();

        // 하단 좌측 (⌞)
        ctx.beginPath();
        ctx.moveTo(x + cornerLength, y + height);
        ctx.lineTo(x, y + height);
        ctx.lineTo(x, y + height - cornerLength);
        ctx.stroke();

        ctx.restore();
      };

      // 라벨 그리기 함수 (가독성 개선 V2)
      const drawLabel = (x, y, width, text, color) => {
        const fontSize = 15; // 폰트 키움
        const padding = 8;
        // 산세리프 폰트, 가독성 최우선
        ctx.font = `600 ${fontSize}px "Pretendard", "Segoe UI", sans-serif`;
        const metrics = ctx.measureText(text);
        const textWidth = metrics.width;
        const labelHeight = fontSize + padding;
        const labelWidth = textWidth + padding * 2;

        const lineLength = 25; // 지시선 길이 연장

        ctx.save();
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        // 지시선 시작점 (박스 우측 상단 모서리)
        const startX = x + width;
        const startY = y;

        // 지시선 끝점 (사선 위로)
        const endX = startX + lineLength;
        const endY = startY - lineLength;

        // 지시선 그리기 (/ 형태)
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        // 텍스트 받침 선 (__ 형태)
        ctx.lineTo(endX + labelWidth + 10, endY); // 텍스트보다 약간 더 길게
        ctx.stroke();

        // 텍스트 그리기
        ctx.globalAlpha = 1.0;
        ctx.textBaseline = "bottom";

        // 텍스트 그림자 (검은색 테두리 효과로 가독성 극대화)
        ctx.shadowColor = "rgba(0,0,0,1.0)";
        ctx.shadowBlur = 4;
        ctx.lineWidth = 3;
        ctx.strokeText(text, endX + padding, endY - 4); // 텍스트 외곽선

        ctx.shadowBlur = 0; // 그림자 제거 후 채우기
        ctx.fillStyle = color; // 네온 색상으로 채우기
        ctx.fillText(text, endX + padding, endY - 4);

        ctx.restore();
      };

      const data = boundingBoxData[camId];

      // 데이터가 없으면 리턴
      if (!data || !data.original_width || !data.original_height) {
        animationFrameId = requestAnimationFrame(drawBoundingBoxes);
        return;
      }

      // bbox 파싱 헬퍼 함수
      const parseBbox = (bbox) => {
        if (!bbox) return { x: 0, y: 0, width: 0, height: 0 };

        // 배열 형식 [x1, y1, x2, y2] 또는 [x, y, width, height]
        if (Array.isArray(bbox)) {
          if (bbox.length >= 4) {
            // [x1, y1, x2, y2] 형식
            if (bbox[2] > bbox[0] && bbox[3] > bbox[1]) {
              return {
                x: bbox[0],
                y: bbox[1],
                width: bbox[2] - bbox[0],
                height: bbox[3] - bbox[1],
              };
            } else {
              // [x, y, width, height] 형식
              return {
                x: bbox[0],
                y: bbox[1],
                width: bbox[2],
                height: bbox[3],
              };
            }
          }
        }

        // 객체 형식 {x, y, width, height} 또는 {x1, y1, x2, y2}
        if (typeof bbox === "object") {
          if (bbox.x2 !== undefined && bbox.y2 !== undefined) {
            return {
              x: bbox.x1 || bbox.x || 0,
              y: bbox.y1 || bbox.y || 0,
              width: (bbox.x2 || 0) - (bbox.x1 || bbox.x || 0),
              height: (bbox.y2 || 0) - (bbox.y1 || bbox.y || 0),
            };
          }
          return {
            x: bbox.x || bbox.x1 || 0,
            y: bbox.y || bbox.y1 || 0,
            width: bbox.width || bbox.w || 0,
            height: bbox.height || bbox.h || 0,
          };
        }

        return { x: 0, y: 0, width: 0, height: 0 };
      };

      // 통합 박스 데이터 생성 (강력한 Grouping)
      const groupBoxes = () => {
        const mergedBoxes = [];

        // 모든 감지된 박스 수집
        const allDetections = [
          ...data.violations.map((d) => ({
            ...d,
            isViolation: true,
            type: d.type || d.violation_type || "위반",
          })),
          ...data.recognized_faces.map((d) => ({
            ...d,
            isFace: true,
            type: mapWorkerName(d.worker || d.name || "얼굴"),
          })),
          ...(Array.isArray(data.normal_detections)
            ? data.normal_detections
            : []
          ).map((d) => ({
            ...d,
            isNormal: true,
            type: mapWorkerName(d.worker || d.name || "정상"),
          })),
        ];

        if (allDetections.length === 0) return [];

        // 박스 병합 (IoU 기반 - 더 넓은 범위로 병합)
        const usedIndices = new Set();

        allDetections.forEach((det1, i) => {
          if (usedIndices.has(i)) return;

          const bbox1 = parseBbox(det1.bbox || det1.box);
          if (bbox1.width === 0) return;

          const group = {
            bbox: bbox1,
            items: [det1],
            mainType: det1.type,
            maxConf: det1.confidence || 0,
            isViolation: det1.isViolation,
            isFace: det1.isFace,
            isNormal: det1.isNormal,
          };

          usedIndices.add(i);

          // 다른 박스들과 비교하여 병합
          allDetections.forEach((det2, j) => {
            if (i === j || usedIndices.has(j)) return;

            const bbox2 = parseBbox(det2.bbox || det2.box);
            if (bbox2.width === 0) return;

            // IoU 계산
            const x1 = Math.max(bbox1.x, bbox2.x);
            const y1 = Math.max(bbox1.y, bbox2.y);
            const x2 = Math.min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
            const y2 = Math.min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);

            if (x2 > x1 && y2 > y1) {
              const intersection = (x2 - x1) * (y2 - y1);
              const area1 = bbox1.width * bbox1.height;
              const area2 = bbox2.width * bbox2.height;
              const union = area1 + area2 - intersection;
              const iou = intersection / union;

              // 겹침이 10% 이상이면 같은 그룹으로 간주 (더 적극적으로 병합)
              if (iou > 0.1) {
                group.items.push(det2);
                usedIndices.add(j);

                if (det2.isViolation) group.isViolation = true;
                if (det2.isFace) group.isFace = true;

                // 가장 큰 박스로 영역 확장
                const minX = Math.min(group.bbox.x, bbox2.x);
                const minY = Math.min(group.bbox.y, bbox2.y);
                const maxX = Math.max(
                  group.bbox.x + group.bbox.width,
                  bbox2.x + bbox2.width
                );
                const maxY = Math.max(
                  group.bbox.y + group.bbox.height,
                  bbox2.y + bbox2.height
                );

                group.bbox = {
                  x: minX,
                  y: minY,
                  width: maxX - minX,
                  height: maxY - minY,
                };
              }
            }
          });

          mergedBoxes.push(group);
        });

        return mergedBoxes;
      };

      const groupedBoxes = groupBoxes();

      // 통합된 박스 그리기
      groupedBoxes.forEach((group) => {
        const bbox = group.bbox;
        const x = bbox.x * scaleX + drawX;
        const y = bbox.y * scaleY + drawY;
        const width = bbox.width * scaleX;
        const height = bbox.height * scaleY;

        // 라벨 생성
        let labelText = "";
        let color = "#10B981"; // 기본: Green

        // 1. 위반 사항 수집
        const violations = group.items
          .filter((item) => item.isViolation)
          .map((item) => {
            const type = item.type || "";
            if (type.includes("Hardhat")) return "안전모";
            if (type.includes("Mask")) return "마스크";
            if (type.includes("Vest")) return "조끼";
            return type;
          });

        // 2. 작업자 이름 확인
        const workerName = group.items.find((item) => item.isFace)?.type;
        const nameDisplay =
          workerName && workerName !== "Unknown" && workerName !== "알 수 없음"
            ? workerName
            : "";

        // 3. 최종 라벨 및 색상 결정
        if (group.isViolation) {
          color = "#FF0044"; // Red
          const uniqueViolations = [...new Set(violations)];
          // 너무 길면 줄임
          const violationText =
            uniqueViolations.length > 2
              ? `${uniqueViolations[0]} 외 ${uniqueViolations.length - 1}건`
              : uniqueViolations.join(", ");

          labelText = `${
            nameDisplay ? `[${nameDisplay}] ` : ""
          }${violationText} 미착용`;
        } else if (nameDisplay) {
          color = "#00CCFF"; // Blue
          labelText = `${nameDisplay}`;
        } else {
          color = "#10B981"; // Green
          labelText = "작업자";
        }

        // 미식별이면서 위반이 없는 경우
        if (!group.isViolation && !nameDisplay && group.isFace) {
          color = "#FFAA00"; // Orange
          labelText = "미식별";
        }

        drawCornerBracket(x, y, width, height, color, group.isViolation);
        drawLabel(x, y, width, labelText, color);
      });

      // 기존 개별 박스 그리기 코드 제거됨 (Grouping 결과만 그림)

      animationFrameId = requestAnimationFrame(drawBoundingBoxes);
    };

    drawBoundingBoxes();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
    */
  }, [focusCam, boundingBoxData]);

  // Face-api.js 모델 로드 (앱 시작 시 미리 로드)
  useEffect(() => {
    const loadModels = async () => {
      // face-api.js가 아직 로드되지 않았으면 대기
      if (typeof window.faceapi === "undefined") {
        const checkInterval = setInterval(() => {
          if (typeof window.faceapi !== "undefined") {
            clearInterval(checkInterval);
            loadModels();
          }
        }, 100);
        return;
      }

      try {
        const MODEL_URL =
          "https://justadudewhohacks.github.io/face-api.js/models";
        await Promise.all([
          window.faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          window.faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        ]);
        setModelsLoaded(true);
        setFaceApiLoaded(true);
      } catch (error) {
        console.error("Face-api.js 모델 로딩 실패:", error);
        setCaptureMessage("AI 모델 로딩에 실패했습니다.");
      }
    };

    // 앱 시작 시 즉시 모델 로드 (카메라 열기 전에 미리 로드)
    loadModels();
  }, []);

  // captureStep 변경 시 ref 동기화 및 얼굴 감지 재시작
  // 디버그 정보 업데이트 함수
  const updateDebugInfo = (info) => {
    const newInfo = {
      step: captureStep,
      message: captureMessageRef.current,
      isAutoCapturing,
      timestamp: new Date().toISOString(),
      ...info,
    };
    debugInfoRef.current = newInfo;
  };

  useEffect(() => {
    captureMessageRef.current = captureMessage;
  }, [captureMessage]);

  useEffect(() => {
    const prevStep = captureStepRef.current;
    captureStepRef.current = captureStep;

    // 디버그 정보 업데이트
    updateDebugInfo({
      step: captureStep,
      prevStep,
      message: captureMessageRef.current,
      isAutoCapturing,
      log: `captureStep 변경: ${prevStep} -> ${captureStep}`,
    });

    // captureStep이 변경되고, 촬영 중이 아니고, 얼굴 감지가 일시 중지되지 않았고, 카메라가 켜져있을 때 얼굴 감지 재시작
    if (
      (isCameraMode || showAccessCamera) &&
      !isAutoCapturing &&
      !pauseFaceDetectionRef.current && // 얼굴 감지 일시 중지 확인
      modelsLoaded &&
      faceApiLoaded &&
      accessCameraVideoRef.current &&
      captureStep !== "complete" &&
      !capturedStepsRef.current.has(captureStep) &&
      prevStep !== captureStep
    ) {
      updateDebugInfo({
        log: `얼굴 감지 재시작 시작: ${prevStep} -> ${captureStep}`,
        message: captureMessageRef.current,
      });

      // 기존 interval 정리
      if (faceDetectionIntervalRef.current) {
        clearInterval(faceDetectionIntervalRef.current);
        faceDetectionIntervalRef.current = null;
      }

      // 얼굴 감지 재시작 (더 적극적으로 여러 번 시도)
      const restartWithRetry = (attempt = 1) => {
        setTimeout(() => {
          if (startFaceDetectionRef.current && !isAutoCapturing) {
            updateDebugInfo({ log: `얼굴 감지 재시작 시도 ${attempt}` });

            try {
              startFaceDetectionRef.current();

              // 재시작 확인 (더 긴 시간 대기)
              setTimeout(() => {
                if (!faceDetectionIntervalRef.current && attempt < 10) {
                  updateDebugInfo({
                    log: `얼굴 감지 재시작 실패, 재시도 ${attempt + 1}`,
                  });
                  restartWithRetry(attempt + 1);
                } else if (faceDetectionIntervalRef.current) {
                  updateDebugInfo({
                    log: `얼굴 감지 재시작 성공 시도 ${attempt}`,
                  });
                }
              }, 500); // 200ms -> 500ms로 증가
            } catch (err) {
              updateDebugInfo({ log: `얼굴 감지 재시작 오류: ${err.message}` });
              if (attempt < 10) {
                restartWithRetry(attempt + 1);
              }
            }
          } else if (attempt < 10) {
            updateDebugInfo({
              log: `얼굴 감지 재시작 조건 불만족, 재시도 ${attempt + 1}`,
            });
            restartWithRetry(attempt + 1);
          }
        }, attempt * 200); // 100ms -> 200ms로 증가
      };

      restartWithRetry(1);
    }
  }, [
    captureStep,
    isAutoCapturing,
    modelsLoaded,
    faceApiLoaded,
    isCameraMode,
    showAccessCamera,
  ]);

  // 출입 카메라 설정 (새 창 모드)
  useEffect(() => {
    let stream = null;

    // 카메라 시작 조건: 카메라 모드이고 작업자 입력 UI가 숨겨진 경우
    const shouldStartCamera =
      (isCameraMode || showAccessCamera) && !showWorkerInput;

    if (!shouldStartCamera) {
      // 작업자 입력 UI가 표시된 경우 스트림 중지
      cameraReadyRef.current = false;
      if (accessCameraStream) {
        accessCameraStream.getTracks().forEach((track) => track.stop());
        setAccessCameraStream(null);
      }
      if (faceDetectionIntervalRef.current) {
        clearInterval(faceDetectionIntervalRef.current);
        faceDetectionIntervalRef.current = null;
      }
      // 카운트다운 타이머 정리
      if (countdownTimerRef.current) {
        clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
      // 카메라가 닫힐 때 상태 초기화
      setCaptureStep("front");
      capturedStepsRef.current.clear(); // 촬영 완료 단계 초기화
      setCountdown(0); // 카운트다운 초기화
      countdownCountRef.current = 0; // 카운트다운 카운터 초기화
      setFaceInGuide(false);
      lastMessageRef.current = "";
      setCaptureMessage("얼굴을 가이드에 맞춰주세요");
      countdownWaitingRef.current = false;
      if (warmupCheckIntervalRef.current) {
        clearInterval(warmupCheckIntervalRef.current);
        warmupCheckIntervalRef.current = null;
      }
      return;
    }

    // 카메라가 시작될 때 마지막 촬영 시간 초기화 (첫 촬영 딜레이 제거)
    if (
      shouldStartCamera &&
      captureStep === "front" &&
      !capturedStepsRef.current.has("front")
    ) {
      lastCaptureTimeRef.current = 0;
      capturedStepsRef.current.clear(); // 촬영 완료 단계 초기화
      setFaceInGuide(false);
      lastMessageRef.current = "";
      setCaptureMessage("얼굴을 가이드에 맞춰주세요");
      countdownWaitingRef.current = false;
    }

    const startAccessCamera = async () => {
      try {
        // 모바일 환경 감지
        const isMobile =
          /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
            navigator.userAgent
          ) || window.innerWidth <= 768;

        // 모바일에서는 ideal 또는 max 사용, 데스크톱에서는 고정 해상도
        const videoConstraints = isMobile
          ? {
              facingMode: "user",
              width: { ideal: 1280, max: 1920 },
              height: { ideal: 720, max: 1080 },
            }
          : {
              width: 1280,
              height: 720,
              facingMode: "user",
            };

        stream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraints,
          audio: false,
        });
        setAccessCameraStream(stream);

        if (accessCameraVideoRef.current) {
          const video = accessCameraVideoRef.current;
          video.srcObject = stream;

          // 모바일에서 비디오 재생 보장 (playsInline, muted 속성으로 자동 재생 정책 우회)
          video.setAttribute("playsinline", "true");
          video.setAttribute("webkit-playsinline", "true");
          video.muted = true;

          // 비디오가 실제로 재생되도록 보장 (모바일 대응)
          const playVideo = async () => {
            try {
              await video.play();
            } catch (err) {
              console.error("비디오 재생 오류:", err);
              // 모바일에서 재생 실패 시 재시도
              setTimeout(() => {
                video
                  .play()
                  .catch((e) => console.error("비디오 재생 재시도 실패:", e));
              }, 300);
            }
          };
          playVideo();

          // 비디오가 로드된 후 얼굴 감지 시작
          const handleLoadedMetadata = () => {
            // 비디오가 실제로 프레임을 렌더링할 때까지 대기
            const checkVideoReady = () => {
              if (
                video.readyState >= video.HAVE_CURRENT_DATA &&
                video.videoWidth > 0 &&
                video.videoHeight > 0
              ) {
                cameraReadyRef.current = true;
                // 실제 준비 상태 확인 (카메라 + 모델)
                const isModelsReady =
                  modelsLoaded &&
                  faceApiLoaded &&
                  typeof window.faceapi !== "undefined";
                if (isModelsReady) {
                  const stepMessages = {
                    front: "정면을 향해주세요",
                    left: "왼쪽을 향해주세요",
                    right: "오른쪽을 향해주세요",
                  };
                  setCaptureMessage(
                    stepMessages[captureStep] || "얼굴을 가이드에 맞춰주세요"
                  );
                  startFaceDetection();
                } else {
                  setCaptureMessage("AI 모델 준비 중... 잠시만 기다려주세요");
                }
              } else {
                // 아직 준비되지 않았으면 다시 시도
                setTimeout(checkVideoReady, 100);
              }
            };

            checkVideoReady();
          };

          video.addEventListener("loadedmetadata", () => {
            cameraReadyRef.current = true;
            handleLoadedMetadata();
          });

          // 이미 로드된 경우 즉시 실행
          if (video.readyState >= video.HAVE_METADATA) {
            cameraReadyRef.current = true;
            handleLoadedMetadata();
          }
        }
      } catch (err) {
        console.error("카메라 접근 오류:", err);
        alert("카메라에 접근할 수 없습니다. 권한을 확인해주세요.");
        if (isCameraMode) {
          window.close();
        } else {
          setShowAccessCamera(false);
        }
      }
    };

    const startFaceDetection = () => {
      // 기존 interval 정리
      if (faceDetectionIntervalRef.current) {
        clearInterval(faceDetectionIntervalRef.current);
        faceDetectionIntervalRef.current = null;
      }

      if (!modelsLoaded || typeof window.faceapi === "undefined") {
        return;
      }

      const CAPTURE_COOLDOWN = 1500; // 1.5초 쿨다운
      const FACE_SIZE_THRESHOLD = 150; // 얼굴 최소 크기 (px)

      let stableCount = 0; // 연속으로 조건을 만족한 횟수

      const cancelCountdown = (message) => {
        if (countdownTimerRef.current) {
          clearInterval(countdownTimerRef.current);
          countdownTimerRef.current = null;
        }
        if (countdownCountRef.current !== 0) {
          countdownCountRef.current = 0;
          setCountdown(0);
        }
        if (message && message !== lastMessageRef.current) {
          setCaptureMessage(message);
          lastMessageRef.current = message;
        }
      };

      // 실제 준비 상태 확인 함수
      const checkAllReady = () => {
        const video = accessCameraVideoRef.current;
        const isVideoReady =
          video &&
          video.readyState >= video.HAVE_ENOUGH_DATA &&
          video.videoWidth > 0 &&
          video.videoHeight > 0;

        const isModelsReady =
          modelsLoaded &&
          faceApiLoaded &&
          typeof window.faceapi !== "undefined";

        return isVideoReady && isModelsReady;
      };

      const startCountdownAndCapture = (step, duration = 3) => {
        // 실제 준비 상태 확인 (고정 10초 대기 제거)
        if (!checkAllReady()) {
          if (!warmupCheckIntervalRef.current) {
            setCaptureMessage("장비 준비 중입니다...");

            const startTime = Date.now();
            const maxWaitMs = 10000; // 최대 10초 대기

            warmupCheckIntervalRef.current = setInterval(() => {
              if (checkAllReady()) {
                // 준비 완료 시 즉시 시작
                clearInterval(warmupCheckIntervalRef.current);
                warmupCheckIntervalRef.current = null;

                const readyMessage =
                  "장비 준비가 완료되었습니다. 촬영을 시작합니다.";
                setCaptureMessage(readyMessage);
                lastMessageRef.current = readyMessage;

                if (cameraReadyRef.current && !isAutoCapturing) {
                  // 짧은 딜레이 후 다시 촬영 시도
                  setTimeout(() => {
                    startCountdownAndCapture(step, duration);
                  }, 200);
                }
              } else if (Date.now() - startTime > maxWaitMs) {
                // 타임아웃 (10초 경과)
                clearInterval(warmupCheckIntervalRef.current);
                warmupCheckIntervalRef.current = null;
                setCaptureMessage(
                  "장비 준비 시간이 초과되었습니다. 다시 시도해주세요."
                );
              } else {
                // 준비 중 메시지 업데이트
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                setCaptureMessage(`장비 준비 중입니다... (${elapsed}초)`);
              }
            }, 200); // 200ms마다 체크
          }
          return false;
        }

        if (!cameraReadyRef.current) {
          if (!countdownWaitingRef.current) {
            countdownWaitingRef.current = true;
            const waitMsg =
              "카메라 준비 중입니다. 잠시 후 자동으로 촬영됩니다.";
            if (waitMsg !== lastMessageRef.current) {
              setCaptureMessage(waitMsg);
              lastMessageRef.current = waitMsg;
            }

            setTimeout(() => {
              countdownWaitingRef.current = false;
              if (cameraReadyRef.current && !isAutoCapturing) {
                startCountdownAndCapture(step, duration);
              }
            }, 250);
          }
          return false;
        }

        // 이미 촬영된 단계는 건너뛰기 (재촬영 모드에서도 동일하게 적용)
        if (capturedStepsRef.current.has(step)) {
          return false;
        }

        if (countdownCountRef.current !== 0 || countdownTimerRef.current) {
          return false;
        }

        captureStepRef.current = step;
        countdownCountRef.current = duration;
        setCountdown(duration);

        const stepLabel =
          step === "front" ? "정면" : step === "left" ? "왼쪽" : "오른쪽";
        const countdownMessage = `${stepLabel} 촬영 준비 중...`;
        if (countdownMessage !== lastMessageRef.current) {
          setCaptureMessage(countdownMessage);
          lastMessageRef.current = countdownMessage;
        }

        countdownTimerRef.current = setInterval(() => {
          countdownCountRef.current -= 1;
          setCountdown(Math.max(countdownCountRef.current, 0));

          if (countdownCountRef.current <= 0) {
            clearInterval(countdownTimerRef.current);
            countdownTimerRef.current = null;
            countdownCountRef.current = 0;
            setCountdown(0);

            if (faceDetectionIntervalRef.current) {
              clearInterval(faceDetectionIntervalRef.current);
              faceDetectionIntervalRef.current = null;
            }

            const currentStep = captureStepRef.current;
            // 이미 촬영된 단계는 건너뛰기
            if (capturedStepsRef.current.has(currentStep)) {
              setIsAutoCapturing(false);
              return;
            }

            setIsAutoCapturing(true);
            lastCaptureTimeRef.current = Date.now();
            stableCount = 0;

            setTimeout(() => {
              try {
                const capturePromise = captureFromAccessCamera();

                if (
                  capturePromise &&
                  typeof capturePromise.then === "function"
                ) {
                  capturePromise.catch((err) => {
                    console.error(`촬영 오류:`, err);
                    setIsAutoCapturing(false);

                    if (
                      accessCameraVideoRef.current &&
                      !faceDetectionIntervalRef.current
                    ) {
                      setTimeout(() => {
                        if (startFaceDetectionRef.current) {
                          startFaceDetectionRef.current();
                        }
                      }, 500);
                    }
                  });
                }
              } catch (err) {
                console.error("촬영 중 예외 발생:", err);
                setIsAutoCapturing(false);
              }
            }, 100);
          }
        }, 1000);

        return true;
      };

      // 500ms마다 얼굴 감지 수행
      // 촬영 중일 때는 얼굴 감지 일시 중지 (깜빡임 방지)
      if (isAutoCapturing) {
        return;
      }

      faceDetectionIntervalRef.current = setInterval(async () => {
        // 촬영 중이면 얼굴 감지 완전히 건너뛰기 (깜빡임 방지)
        if (isAutoCapturing || !accessCameraVideoRef.current) {
          stableCount = 0; // 촬영 중이면 카운터 리셋
          return;
        }

        // 카운트다운 중에는 얼굴 감지 건너뛰기 (깜빡임 방지)
        // 단, 카운트다운 취소를 위한 최소한의 체크만 수행
        const isCountdownActive =
          countdownCountRef.current > 0 || countdownTimerRef.current;

        if (isCountdownActive) {
          return;
        }

        const video = accessCameraVideoRef.current;

        if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

        // 카운트다운 중 가이드 벗어남 체크는 카운트다운 타이머 내부에서 별도로 수행
        if (pauseFaceDetectionRef.current) {
          return; // 일시 중지된 경우 얼굴 감지 건너뛰기
        }

        // 현재 단계가 이미 촬영되었는지 확인 (반복 촬영 방지)
        if (capturedStepsRef.current.has(captureStep)) {
          stableCount = 0;
          return;
        }

        // 쿨다운 시간 확인 및 첫 촬영 여부 확인
        const now = Date.now();
        const isFirstCapture = lastCaptureTimeRef.current === 0;
        // 측면 촬영 시 안정화 횟수 증가 (가이드에 정확히 맞추도록)
        const isSideCapture = captureStep === "left" || captureStep === "right";
        // 안정성 검증 횟수: 측면은 5번 (1.5초), 정면은 4번 (1.2초) - 더 엄격하게
        const requiredStableCount = isSideCapture ? 5 : 4;
        // 촬영 완료 후 쿨다운 시간 증가 (반복 촬영 방지)
        const cooldownTime = capturedStepsRef.current.has(captureStep)
          ? 10000
          : CAPTURE_COOLDOWN; // 촬영 완료된 단계는 10초 쿨다운
        if (
          !isFirstCapture &&
          now - lastCaptureTimeRef.current < cooldownTime
        ) {
          const remainingTime = Math.ceil(
            (cooldownTime - (now - lastCaptureTimeRef.current)) / 1000
          );
          const newMessage = `다음 촬영까지 ${remainingTime}초 대기 중...`;
          // 메시지가 변경될 때만 업데이트 (깜빡임 방지)
          if (newMessage !== lastMessageRef.current) {
            setCaptureMessage(newMessage);
            lastMessageRef.current = newMessage;
          }
          return;
        }

        try {
          // face-api.js로 얼굴 감지 (landmarks 포함)
          const detections = await window.faceapi
            .detectAllFaces(video, new window.faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks();

          if (detections.length > 0) {
            const detection = detections[0];
            const face = detection.detection.box;
            const landmarks = detection.landmarks;

            // 가이드 영역 계산 (화면 중앙 기준)
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            // 모바일 환경 감지
            const isMobile =
              /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
                navigator.userAgent
              ) || window.innerWidth <= 768;

            // 측면 촬영 시 가이드 영역 크기 조정 (정확한 위치 확인)
            const isSideCapture =
              captureStep === "left" || captureStep === "right";
            // 모바일에서는 화면 크기에 비례하여 가이드 크기 조정 (더 엄격하게)
            const baseGuideWidth = isMobile
              ? Math.min(videoWidth * 0.5, 350)
              : isSideCapture
              ? 450
              : 350;
            const baseGuideHeight = isMobile
              ? Math.min(videoHeight * 0.5, 450)
              : isSideCapture
              ? 550
              : 450;
            const guideWidth = isSideCapture
              ? isMobile
                ? baseGuideWidth * 1.1
                : baseGuideWidth * 1.15
              : baseGuideWidth;
            const guideHeight = isSideCapture
              ? isMobile
                ? baseGuideHeight * 1.1
                : baseGuideHeight * 1.15
              : baseGuideHeight;
            const guideCenterX = videoWidth / 2;
            const guideCenterY = videoHeight / 2;
            const guideLeft = guideCenterX - guideWidth / 2;
            const guideRight = guideCenterX + guideWidth / 2;
            const guideTop = guideCenterY - guideHeight / 2;
            const guideBottom = guideCenterY + guideHeight / 2;

            // 얼굴 중심점 계산
            const faceCenterX = face.x + face.width / 2;
            const faceCenterY = face.y + face.height / 2;

            // 얼굴 각도 계산 (측면 촬영용)
            let faceAngleOK = true;
            if (captureStep === "left" || captureStep === "right") {
              // 측면 촬영일 때는 얼굴 각도 조건을 거의 체크하지 않음 (인식 개선)
              // 측면 촬영은 인식이 어려우므로 각도 조건을 최소화
              const nose = landmarks.getNose();
              const leftEye = landmarks.getLeftEye();
              const rightEye = landmarks.getRightEye();

              // 랜드마크가 있으면 매우 완화된 조건으로 체크
              if (nose && leftEye && rightEye) {
                const noseX = nose[0].x;
                const leftEyeX =
                  leftEye.reduce((sum, p) => sum + p.x, 0) / leftEye.length;
                const rightEyeX =
                  rightEye.reduce((sum, p) => sum + p.x, 0) / rightEye.length;
                const eyeCenterX = (leftEyeX + rightEyeX) / 2;

                // 얼굴이 옆을 향하는 정도 계산
                const faceDirection = noseX - eyeCenterX;

                if (captureStep === "left") {
                  // 왼쪽 측면: 거의 모든 각도 허용 (매우 완화)
                  // 음수면 좋지만, 양수여도 허용 (거의 모든 경우 허용)
                  faceAngleOK = faceDirection < 10; // -2 → 10으로 매우 완화 (거의 모든 각도 허용)
                } else if (captureStep === "right") {
                  // 오른쪽 측면: 거의 모든 각도 허용 (매우 완화)
                  faceAngleOK = faceDirection > -10; // 2 → -10으로 매우 완화 (거의 모든 각도 허용)
                }
              }
              // 랜드마크가 없으면 각도 체크를 건너뛰고 허용
            } else {
              // 정면 촬영: 얼굴이 정면을 향하고 있는지 확인
              const nose = landmarks.getNose();
              const leftEye = landmarks.getLeftEye();
              const rightEye = landmarks.getRightEye();

              if (nose && leftEye && rightEye) {
                const noseX = nose[0].x;
                const leftEyeX =
                  leftEye.reduce((sum, p) => sum + p.x, 0) / leftEye.length;
                const rightEyeX =
                  rightEye.reduce((sum, p) => sum + p.x, 0) / rightEye.length;
                const eyeCenterX = (leftEyeX + rightEyeX) / 2;

                // 정면일 때는 코가 눈 중앙에 가까워야 함 (조건 완화)
                const faceDirection = Math.abs(noseX - eyeCenterX);
                faceAngleOK = faceDirection < 40; // 40px 이내로 완화 (20px → 40px)
              }
            }

            // 얼굴이 가이드 중앙에 얼마나 가까운지 계산 (측면 촬영 시 더 완화)
            const distanceFromCenterX = Math.abs(faceCenterX - guideCenterX);
            const distanceFromCenterY = Math.abs(faceCenterY - guideCenterY);
            // 가이드에 정확히 맞추도록 조건 강화 (더 엄격하게)
            const maxDistanceX = isSideCapture
              ? guideWidth * 0.15 // 측면: 가이드 너비의 15% 이내 (매우 엄격)
              : guideWidth * 0.12; // 정면: 가이드 너비의 12% 이내 (매우 엄격)
            const maxDistanceY = isSideCapture
              ? guideHeight * 0.15 // 측면: 가이드 높이의 15% 이내 (매우 엄격)
              : guideHeight * 0.12; // 정면: 가이드 높이의 12% 이내 (매우 엄격)

            // 얼굴이 가이드 안에 있고 중앙에 가까운지 확인 (조건 완화)
            const faceInGuideArea =
              faceCenterX >= guideLeft &&
              faceCenterX <= guideRight &&
              faceCenterY >= guideTop &&
              faceCenterY <= guideBottom &&
              distanceFromCenterX <= maxDistanceX &&
              distanceFromCenterY <= maxDistanceY;

            // 얼굴 크기 확인 (매우 엄격하게)
            const faceSizeOK = isSideCapture
              ? // 측면 촬영: 매우 엄격한 조건
                face.width >= FACE_SIZE_THRESHOLD * 0.8 && // 최소 크기 80%
                face.width <= guideWidth * 0.85 && // 가이드 너비의 85% 이하 (매우 엄격)
                face.height >= FACE_SIZE_THRESHOLD * 0.8 && // 최소 크기 80%
                face.height <= guideHeight * 0.8 // 가이드 높이의 80% 이하 (매우 엄격)
              : // 정면 촬영: 매우 엄격한 조건
                face.width >= FACE_SIZE_THRESHOLD * 0.6 && // 최소 크기 60%
                face.width <= guideWidth * 0.95 && // 가이드 너비의 95% 이하
                face.height >= FACE_SIZE_THRESHOLD * 0.6 && // 최소 크기 60%
                face.height <= guideHeight * 0.9; // 가이드 높이의 90% 이하

            const isPerfectPosition =
              faceInGuideArea && faceSizeOK && faceAngleOK;

            // 상태 업데이트 최소화 (깜빡임 방지)
            // 촬영 중이 아니고, 카운트다운 중이 아니고, 얼굴 감지 interval이 활성화된 상태에서만 업데이트
            const isCountdownActive =
              countdownCountRef.current > 0 || countdownTimerRef.current;
            if (
              !isAutoCapturing &&
              faceDetectionIntervalRef.current &&
              isPerfectPosition !== faceInGuide
            ) {
              setFaceInGuide(isPerfectPosition);
            }

            if (!cameraReadyRef.current) {
              stableCount = 0;
              const waitMsg = "카메라 초기화 중입니다. 잠시만 기다려주세요.";
              if (
                !isAutoCapturing &&
                faceDetectionIntervalRef.current &&
                waitMsg !== lastMessageRef.current
              ) {
                setCaptureMessage(waitMsg);
                lastMessageRef.current = waitMsg;
              }
              return;
            }

            if (isPerfectPosition) {
              if (isAutoCapturing) {
                stableCount = 0;
                return;
              }

              if (isCountdownActive) {
                stableCount = 0;
                return;
              }

              if (capturedStepsRef.current.has(captureStep)) {
                stableCount = 0;
                return;
              }

              if (captureStep === "front") {
                const started = startCountdownAndCapture(captureStep, 3);
                if (started) {
                  stableCount = 0;
                  return;
                }
              } else {
                stableCount++;

                if (stableCount >= requiredStableCount) {
                  const started = startCountdownAndCapture(captureStep, 3);
                  if (started) {
                    stableCount = 0;
                    return;
                  }
                } else {
                  if (
                    !isAutoCapturing &&
                    faceDetectionIntervalRef.current &&
                    stableCount % 5 === 0 &&
                    !isCountdownActive
                  ) {
                    const progress = Math.min(
                      (stableCount / requiredStableCount) * 100,
                      100
                    );
                    const progressBar =
                      "█".repeat(Math.floor(progress / 10)) +
                      "░".repeat(10 - Math.floor(progress / 10));
                    const newMessage = `위치 확인 중... ${progressBar} ${Math.round(
                      progress
                    )}%`;
                    if (newMessage !== lastMessageRef.current) {
                      setCaptureMessage(newMessage);
                      lastMessageRef.current = newMessage;
                    }
                  }
                }
              }
            } else {
              const stepMessages = {
                front: "정면을 향해주세요",
                left: "왼쪽을 향해주세요",
                right: "오른쪽을 향해주세요",
              };

              let guidanceMessage = null;

              if (!faceAngleOK) {
                guidanceMessage =
                  stepMessages[captureStep] || "얼굴 각도를 맞춰주세요";
              } else if (!faceInGuideArea) {
                guidanceMessage =
                  distanceFromCenterX > maxDistanceX ||
                  distanceFromCenterY > maxDistanceY
                    ? "얼굴을 중앙에 맞춰주세요"
                    : "얼굴을 가이드 안에 맞춰주세요";
              } else if (!faceSizeOK) {
                guidanceMessage =
                  face.width < FACE_SIZE_THRESHOLD ||
                  face.height < FACE_SIZE_THRESHOLD
                    ? "조금 더 가까이 다가가세요"
                    : "조금 더 멀리 떨어지세요";
              } else {
                guidanceMessage = "얼굴 위치를 조정해주세요";
              }

              if (isCountdownActive) {
                cancelCountdown(guidanceMessage);
              }

              stableCount = 0; // 조건을 만족하지 않으면 카운터 리셋

              if (
                !isAutoCapturing &&
                faceDetectionIntervalRef.current &&
                guidanceMessage &&
                guidanceMessage !== lastMessageRef.current &&
                !isCountdownActive
              ) {
                setCaptureMessage(guidanceMessage);
                lastMessageRef.current = guidanceMessage;
              }
            }
          } else {
            stableCount = 0;
            // 촬영 중이 아니고, 카운트다운 중이 아니고, 얼굴 감지 interval이 활성화된 상태에서만 상태 업데이트 (깜빡임 방지)
            const isCountdownActive =
              countdownCountRef.current > 0 || countdownTimerRef.current;
            if (
              !isAutoCapturing &&
              !isCountdownActive &&
              faceDetectionIntervalRef.current
            ) {
              setFaceInGuide(false);
              const newMsg = "얼굴을 찾을 수 없습니다";
              if (newMsg !== lastMessageRef.current) {
                setCaptureMessage(newMsg);
                lastMessageRef.current = newMsg;
              }
            }
          }
        } catch (error) {
          console.error("얼굴 감지 오류:", error);
        }
      }, 300); // 0.3초 간격으로 변경하여 더 부드러운 감지
    };

    // startFaceDetection을 ref에 저장하여 외부에서 호출 가능하게 함
    startFaceDetectionRef.current = startFaceDetection;

    if (modelsLoaded || faceApiLoaded) {
      startAccessCamera();
    }

    // 촬영이 완료되면 얼굴 감지 다시 시작 (멈춤 방지)
    if (
      !isAutoCapturing &&
      shouldStartCamera &&
      modelsLoaded &&
      faceApiLoaded &&
      accessCameraVideoRef.current &&
      !faceDetectionIntervalRef.current &&
      captureStep !== "complete"
    ) {
      // 비디오가 준비되었는지 확인
      const video = accessCameraVideoRef.current;

      // 비디오 재생 보장 및 얼굴 감지 재시작
      const ensureVideoPlayingAndStartDetection = () => {
        if (!video) return;

        // 비디오 스트림 확인
        if (!video.srcObject && accessCameraStream) {
          video.srcObject = accessCameraStream;
        }

        // 비디오가 준비되었는지 확인
        if (
          video.readyState >= video.HAVE_ENOUGH_DATA &&
          video.videoWidth > 0 &&
          video.videoHeight > 0
        ) {
          // 비디오가 재생 중인지 확인
          if (video.paused || video.ended) {
            // 비디오 재생
            video
              .play()
              .then(() => {
                // 재생 성공 후 얼굴 감지 시작
                setTimeout(() => {
                  if (!faceDetectionIntervalRef.current) {
                    startFaceDetection();
                  }
                }, 300);
              })
              .catch((err) => {
                console.error("얼굴 감지 재시작을 위한 비디오 재생 실패:", err);
                // 재시도
                setTimeout(ensureVideoPlayingAndStartDetection, 500);
              });
          } else {
            // 이미 재생 중이면 얼굴 감지 시작
            setTimeout(() => {
              if (!faceDetectionIntervalRef.current) {
                startFaceDetection();
              }
            }, 300);
          }
        } else {
          // 비디오가 아직 준비되지 않았으면 재시도
          setTimeout(ensureVideoPlayingAndStartDetection, 500);
        }
      };

      ensureVideoPlayingAndStartDetection();
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        setAccessCameraStream(null);
      }
      if (faceDetectionIntervalRef.current) {
        clearInterval(faceDetectionIntervalRef.current);
        faceDetectionIntervalRef.current = null;
      }
    };
  }, [
    isCameraMode,
    showAccessCamera,
    showWorkerInput,
    isAutoCapturing,
    modelsLoaded,
    faceApiLoaded,
    captureStep,
  ]);

  useEffect(() => {
    if (activePage !== "dashboard") {
      // 대시보드가 아닐 때 기존 차트 정리
      if (gaugeChartRef.current) {
        gaugeChartRef.current.destroy();
        gaugeChartRef.current = null;
      }
      return;
    }

    // DOM이 완전히 렌더링된 후 차트 생성 (다음 프레임에 실행)
    const renderChart = () => {
      if (!gaugeRef.current) {
        // gaugeRef가 아직 준비되지 않았으면 재시도
        setTimeout(renderChart, 100);
        return;
      }

      const ctx = gaugeRef.current.getContext("2d");
      if (!ctx) {
        console.error("[차트] Gauge 차트 컨텍스트를 가져올 수 없습니다");
        return;
      }

      const isDark = theme === "dark";

      // 차트가 이미 존재하면 데이터만 업데이트
      if (gaugeChartRef.current) {
        try {
          gaugeChartRef.current.data.datasets[0].data = [
            safetyScore,
            Math.max(0, 100 - safetyScore),
          ];
          gaugeChartRef.current.data.datasets[0].backgroundColor = isDark
            ? ["#3B82F6", "#374151"]
            : ["#3B82F6", "#E5E7EB"];
          gaugeChartRef.current.update("none"); // 애니메이션 없이 업데이트
          console.log(
            "[차트] Gauge 차트 업데이트 완료, safetyScore:",
            safetyScore
          );
          return;
        } catch (error) {
          console.error("[차트] Gauge 차트 업데이트 오류:", error);
          // 업데이트 실패 시 차트 재생성
          gaugeChartRef.current.destroy();
          gaugeChartRef.current = null;
        }
      }

      // 차트 생성
      try {
        gaugeChartRef.current = new Chart(ctx, {
          type: "doughnut",
          data: {
            labels: ["안전 점수", "남은 점수"],
            datasets: [
              {
                data: [safetyScore, Math.max(0, 100 - safetyScore)],
                backgroundColor: isDark
                  ? ["#3B82F6", "#374151"]
                  : ["#3B82F6", "#E5E7EB"],
                borderWidth: 0,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "80%",
            circumference: 180,
            rotation: 270,
            animation: false, // 모든 애니메이션 비활성화
            plugins: {
              legend: { display: false },
              tooltip: { enabled: false },
            },
          },
        });
        console.log("[차트] Gauge 차트 생성 완료, safetyScore:", safetyScore);
      } catch (error) {
        console.error("[차트] Gauge 차트 생성 오류:", error);
      }
    };

    // 다음 프레임에 차트 생성 (DOM 렌더링 완료 대기)
    requestAnimationFrame(() => {
      setTimeout(renderChart, 50);
    });

    // Bar 차트 렌더링
    if (barRef.current) {
      const ctx = barRef.current.getContext("2d");
      if (!ctx) {
        console.error("[차트] Bar 차트 컨텍스트를 가져올 수 없습니다");
        return;
      }
      barChartRef.current?.destroy();

      // 최근 7주 데이터 생성 (이번 주 포함)
      const weekLabels = [];
      for (let i = 6; i >= 0; i--) {
        weekLabels.push(getWeekLabel(i, language, translations));
      }

      const isDark = theme === "dark";

      // 차트 데이터 준비 (DB에서 가져온 데이터 또는 기본값)
      console.log("[차트] chartData:", chartData);
      console.log("[차트] chartData.length:", chartData.length);

      // chartData가 7개 미만이면 7개로 맞춤
      let processedChartData = chartData;
      if (chartData.length < 7) {
        // 부족한 주차 데이터를 0으로 채움
        const emptyWeeks = Array(7 - chartData.length)
          .fill(null)
          .map((_, i) => ({
            week: chartData.length + i,
            violations: 0,
            helmet_violations: 0,
            fall_detections: 0,
            completed: 0,
          }));
        processedChartData = [...chartData, ...emptyWeeks];
      } else if (chartData.length > 7) {
        // 7개 초과면 최근 7개만 사용
        processedChartData = chartData.slice(-7);
      }

      const violationData = processedChartData.map((d) => d.violations || 0);
      const completedData = processedChartData.map((d) => d.completed || 0);

      console.log("[차트] violationData:", violationData);
      console.log("[차트] completedData:", completedData);
      console.log("[차트] weekLabels:", weekLabels);

      barChartRef.current = new Chart(ctx, {
        type: "bar",
        data: {
          labels: weekLabels,
          datasets: [
            {
              label: t.dashboard.chartViolation,
              data: violationData,
              backgroundColor: isDark ? "#DC2626" : "#FCA5A5",
              borderRadius: 4,
            },
            {
              label: t.dashboard.chartCompleted,
              data: completedData,
              backgroundColor: isDark ? "#3b82f6" : "#3b82f6",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              ticks: { color: isDark ? "#9CA3AF" : "#6B7280" },
              grid: { color: isDark ? "#374151" : "#E5E7EB" },
            },
            x: {
              grid: { display: false },
              ticks: { color: isDark ? "#9CA3AF" : "#6B7280" },
            },
          },
          plugins: {
            legend: {
              display: true,
              position: "bottom",
              align: "start",
              labels: { color: isDark ? "#F9FAFB" : "#1F2937" },
            },
          },
        },
      });
    }

    // cleanup 함수: 컴포넌트 언마운트 시 차트 정리
    return () => {
      if (gaugeChartRef.current) {
        try {
          gaugeChartRef.current.destroy();
          gaugeChartRef.current = null;
        } catch (error) {
          console.error("[차트] Gauge 차트 정리 오류:", error);
        }
      }
      if (barChartRef.current) {
        try {
          barChartRef.current.destroy();
          barChartRef.current = null;
        } catch (error) {
          console.error("[차트] Bar 차트 정리 오류:", error);
        }
      }
    };
  }, [activePage, language, theme, chartData, safetyScore]);

  // 사용자 위치 가져오기
  useEffect(() => {
    if (activePage !== "dashboard") return;

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lon: position.coords.longitude,
          });
        },
        (error) => {
          console.error("위치 권한 오류:", error);
          // 위치 권한이 거부되면 서울 기본값 사용
          setUserLocation({ lat: 37.5665, lon: 126.978 }); // 서울 좌표
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000, // 5분 캐시
        }
      );
    } else {
      console.warn("Geolocation이 지원되지 않습니다. 서울 기본값 사용");
      setUserLocation({ lat: 37.5665, lon: 126.978 }); // 서울 좌표
    }
  }, [activePage]);

  // 날씨 정보 로드 (모든 페이지에서)
  useEffect(() => {
    if (!userLocation) return;

    const loadWeather = async () => {
      setWeatherLoading(true);
      try {
        const weatherResponse = await api.getWeather(
          userLocation.lat,
          userLocation.lon
        );

        // Current Weather Data API 응답 구조: main 객체에 온도, weather 배열에 날씨 정보
        if (weatherResponse && weatherResponse.main) {
          const weatherMain = weatherResponse.weather[0]?.main || "";
          const hasRain =
            weatherResponse.rain &&
            (weatherResponse.rain["1h"] || weatherResponse.rain["3h"]);
          const hasSnow =
            weatherResponse.snow &&
            (weatherResponse.snow["1h"] || weatherResponse.snow["3h"]);

          const originalDescription =
            weatherResponse.weather[0]?.description || "";
          const koreanDescription =
            translateWeatherDescription(originalDescription);

          setWeather({
            temp: Math.round(weatherResponse.main.temp),
            description: koreanDescription,
            icon: weatherResponse.weather[0]?.icon || "",
            city: weatherResponse.name || "현재 위치",
            humidity: weatherResponse.main.humidity || 0,
            hasPrecipitation:
              hasRain ||
              hasSnow ||
              weatherMain === "Rain" ||
              weatherMain === "Snow",
            precipitationType: hasRain ? "rain" : hasSnow ? "snow" : null,
          });
        } else {
          setWeather(null);
        }
      } catch (error) {
        console.error("날씨 정보 로드 오류:", error);
        console.error("에러 상세:", error.message, error.stack);
        setWeather(null);
      } finally {
        setWeatherLoading(false);
      }
    };

    loadWeather();
    // 10분마다 날씨 정보 갱신
    const weatherInterval = setInterval(loadWeather, 600000);
    return () => clearInterval(weatherInterval);
  }, [userLocation]);

  // 현재 시간 업데이트 (1초마다)
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // 날씨 및 시간 바 컴포넌트
  const WeatherTimeBar = () => (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 16px",
        background: "var(--card-bg)",
        borderRadius: "8px",
        marginBottom: "16px",
        border: "1px solid var(--border-color)",
        gap: "16px",
        flexWrap: "wrap",
      }}
    >
      {/* 날씨 정보 */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          flexWrap: "wrap",
          minWidth: "300px" /* 최소 너비 고정 */,
        }}
      >
        {weatherLoading ? (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              minHeight: "24px" /* 아이콘 높이와 동일하게 유지 */,
            }}
          >
            <div
              style={{
                width: "24px",
                height: "24px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <i
                className="fas fa-spinner fa-spin"
                style={{
                  fontSize: "14px",
                  color: "var(--text-secondary)",
                }}
              ></i>
            </div>
            <span
              style={{
                fontSize: "14px",
                color: "var(--text-secondary)",
              }}
            >
              로딩 중...
            </span>
          </div>
        ) : weather ? (
          <>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              {weather.icon && (
                <img
                  src={`https://openweathermap.org/img/wn/${weather.icon}@2x.png`}
                  alt={weather.description}
                  style={{ width: "24px", height: "24px" }}
                />
              )}
              <span
                style={{
                  fontSize: "18px",
                  fontWeight: 600,
                  color: "var(--text-primary)",
                }}
              >
                {weather.temp}°C
              </span>
            </div>
            <div
              style={{
                fontSize: "16px",
                color: "var(--text-secondary)",
              }}
            >
              {weather.description}
            </div>
            {weather.humidity !== undefined && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "4px",
                  fontSize: "15px",
                  color: "var(--text-secondary)",
                }}
              >
                <i className="fas fa-tint" style={{ fontSize: "13px" }}></i>
                <span>습도 {weather.humidity}%</span>
              </div>
            )}
            {weather.hasPrecipitation && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "4px",
                  fontSize: "12px",
                }}
              >
                <i
                  className={`fas ${
                    weather.precipitationType === "snow"
                      ? "fa-snowflake"
                      : "fa-cloud-rain"
                  }`}
                  style={{
                    fontSize: "11px",
                    color:
                      weather.precipitationType === "snow"
                        ? "rgba(59, 130, 246, 1)"
                        : "rgba(59, 130, 246, 1)",
                  }}
                ></i>
                <span
                  style={{
                    color:
                      weather.precipitationType === "snow"
                        ? "rgba(59, 130, 246, 1)"
                        : "rgba(59, 130, 246, 1)",
                  }}
                >
                  {weather.precipitationType === "snow" ? "강설" : "강수"} 주의
                </span>
              </div>
            )}
          </>
        ) : (
          <span
            style={{
              fontSize: "14px",
              color: "var(--text-secondary)",
            }}
          >
            날씨 정보 없음
          </span>
        )}
      </div>

      {/* 시간 정보 */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          fontFamily: '"Chiron GoRound TC", sans-serif',
        }}
      >
        <div
          style={{
            fontSize: "16px",
            fontWeight: 600,
            color: "var(--text-primary)",
          }}
        >
          {currentTime.getFullYear()}년{" "}
          {String(currentTime.getMonth() + 1).padStart(2, "0")}월{" "}
          {String(currentTime.getDate()).padStart(2, "0")}일 (
          {["일", "월", "화", "수", "목", "금", "토"][currentTime.getDay()]})
        </div>
        <div
          style={{
            fontSize: "18px",
            fontWeight: 700,
            color: "var(--accent-blue)",
            fontFamily: '"Chiron GoRound TC", sans-serif',
          }}
        >
          {String(currentTime.getHours()).padStart(2, "0")}:
          {String(currentTime.getMinutes()).padStart(2, "0")}:
          {String(currentTime.getSeconds()).padStart(2, "0")}
        </div>
      </div>
    </div>
  );

  // Live updater - DB에서 주기적으로 갱신 (30초마다)
  useEffect(() => {
    if (activePage !== "dashboard") return;

    const loadData = async () => {
      try {
        // 대시보드 실시간 업데이트: 하루치 데이터 전체 (limit=null, days=1)
        const violationsResponse = await api.getViolations(null, null, 1);
        if (violationsResponse.success && violationsResponse.violations) {
          // 디버깅: 실시간 업데이트 API 응답 확인
          console.log("[실시간 이벤트] API 응답:", {
            success: violationsResponse.success,
            count: violationsResponse.violations?.length || 0,
          });

          // 새 이벤트만 필터링
          const allNewEvents = violationsResponse.violations
            .filter((v) => {
              const status = v.status || "new";
              return status === "new" || status === "pending";
            })
            .map((v) => {
              const formatted = formatViolationEvent(v);
              if (formatted) {
                // 원본 데이터 보존 (API 호출 시 DB에 저장된 형식 그대로 사용)
                formatted._original = {
                  worker_id: v.worker_id,
                  violation_datetime: v.violation_datetime,
                };
              }
              return formatted;
            })
            .filter((event) => event !== null);

          console.log("[실시간 이벤트] 변환된 이벤트:", allNewEvents.length);

          // 기존 eventRows에서 "done" 상태가 아닌 항목만 유지
          setEventRows((prevRows) => {
            const existingNewRows = prevRows.filter(
              (row) => row.status === "new" || !row.status
            );

            // 기존에 표시된 이벤트의 고유 키 목록 생성
            const existingKeys = new Set(
              existingNewRows.map(
                (r) => `${r.worker_id}_${r.violation_datetime}`
              )
            );

            // 새로운 이벤트 중 아직 표시되지 않은 것만 필터링
            const unseenEvents = allNewEvents.filter(
              (event) =>
                !existingKeys.has(
                  `${event.worker_id}_${event.violation_datetime}`
                )
            );

            // 하나씩만 추가 (가장 최신 이벤트 하나)
            if (unseenEvents.length > 0) {
              const newEvent = unseenEvents[0]; // 가장 최신 이벤트 하나만 선택

              // 넘어짐 감지가 있고 알림이 켜져 있으면 사운드 재생
              if (newEvent.risk === "넘어짐 감지" && notificationsEnabled) {
                try {
                  if (alarmRef.current) {
                    alarmRef.current.volume = 0.7;
                    alarmRef.current.pause();
                    alarmRef.current.currentTime = 0;
                    alarmRef.current.play().catch(() => {
                      // 알림 사운드 재생 실패 시 무시
                    });
                  }
                } catch {
                  // 알림 사운드 재생 오류 무시
                }
              }

              // 새 이벤트를 맨 앞에 추가 (제한 없음)
              const updatedRows = [newEvent, ...existingNewRows];
              setPendingCount(updatedRows.length);
              return updatedRows;
            }

            // 새 이벤트가 없으면 기존 항목만 유지 (제한 없음)
            setPendingCount(existingNewRows.length);
            return existingNewRows;
          });
        }

        // 대시보드 실시간 업데이트: 실시간 현장 요약 KPI는 하루치, 주간 위험 통계는 7주(49일) 데이터
        // 실시간 현장 요약 KPI: 하루치 통계
        const todayStatsResponse = await api.getViolationStats(1);
        if (todayStatsResponse.success && todayStatsResponse.kpi) {
          setKpiHelmet(todayStatsResponse.kpi.helmet || 0);
          setKpiVest(todayStatsResponse.kpi.vest || 0);
          setKpiFall(todayStatsResponse.kpi.fall || 0);
        }

        // 주간 위험 통계 차트: 7주(49일) 데이터
        const weeklyStatsResponse = await api.getViolationStats(49);
        if (weeklyStatsResponse.success && weeklyStatsResponse.chart_data) {
          setChartData(weeklyStatsResponse.chart_data);

          // 주간 통계 계산
          const totalWeekly = weeklyStatsResponse.chart_data.reduce(
            (sum, week) => sum + (week.violations || 0),
            0
          );
          setWeeklyTotalDetections(totalWeekly);

          const totalHelmet = weeklyStatsResponse.chart_data.reduce(
            (sum, week) => sum + (week.helmet_violations || 0),
            0
          );
          const totalFall = weeklyStatsResponse.chart_data.reduce(
            (sum, week) => sum + (week.fall_detections || 0),
            0
          );
          setWeeklyHelmetViolations(
            totalHelmet || weeklyStatsResponse.kpi?.helmet || 0
          );
          setWeeklyFallDetections(
            totalFall || weeklyStatsResponse.kpi?.fall || 0
          );
        }

        // 금일 총 알림: 일일 안전 점수 계산 부분에서 KPI 항목 기준으로 설정됨
        // 여기서는 임시로 설정하지 않음 (아래 일일 안전 점수 계산에서 설정)
        const kpiTotal =
          (todayStatsResponse.kpi?.helmet || 0) +
          (todayStatsResponse.kpi?.vest || 0) +
          (todayStatsResponse.kpi?.fall || 0);

        // 일일 안전 점수 계산: 금일 총 알림(KPI 합계) 사용
        if (violationsResponse.success && violationsResponse.violations) {
          // KPI에 해당하는 위반 사항만 필터링 (안전모, 안전조끼, 넘어짐)
          const kpiViolations = violationsResponse.violations.filter((v) => {
            const type = (v.type || v.violation_type || "").toLowerCase();
            return (
              type.includes("안전모") ||
              type.includes("헬멧") ||
              type.includes("helmet") ||
              type.includes("hardhat") ||
              type.includes("안전조끼") ||
              type.includes("조끼") ||
              type.includes("vest") ||
              type.includes("reflective") ||
              type.includes("넘어짐") ||
              type.includes("낙상") ||
              type.includes("fall")
            );
          });

          const completed = kpiViolations.filter(
            (v) => v.status === "done"
          ).length;
          setCompletedActions(completed);

          // 금일 총 알림(KPI 합계)을 total로 사용
          const total = kpiTotal;
          const score = total > 0 ? Math.round((completed / total) * 100) : 100;
          setSafetyScore(score);

          // 확인 필요: KPI 항목 중 status !== "done"인 항목 수 (일일 안전 점수 계산 방식과 일치)
          const pendingKpiItems = kpiTotal - completed;
          setPendingCount(pendingKpiItems);
          // 금일 총 알림도 확인 필요한 KPI 항목 수로 설정
          setTotalAlerts(pendingKpiItems);
        }
      } catch (error) {
        console.error("데이터 갱신 오류:", error);
      }
    };

    // 초기 로드와 중복 실행 방지를 위해 약간의 지연 추가 (초기 로드 완료 후 실행)
    const initialTimeout = setTimeout(() => {
      loadData();
    }, 2000); // 2초 지연으로 초기 로드 완료 후 실행

    // 30초마다 갱신 (실시간 업데이트)
    const id = setInterval(loadData, 30000);
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(id);
    };
  }, [activePage, notificationsEnabled, workersList, formatViolationEvent]);

  // 알림 설정이 꺼지면 재생 중인 사운드 즉시 중지
  useEffect(() => {
    if (!notificationsEnabled && alarmRef.current) {
      try {
        alarmRef.current.pause();
        alarmRef.current.currentTime = 0;
      } catch {
        // 사운드 중지 오류 무시
      }
    }
  }, [notificationsEnabled]);

  // fullLogs는 이제 DB에서 로드됨

  // 실시간 캠 스트림 URL (백엔드 MJPEG 스트림)
  const getStreamUrl = (camId) => {
    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    // const host = window.location.hostname;
    // 백엔드 포트 8081 사용
    return `${protocol}//${window.location.hostname}:8081/stream?cam_id=${camId}`;
  };

  // LiveCam 컴포넌트 렌더링 부분 수정
  // <video> 태그 대신 <img> 태그로 백엔드 스트림 직접 표시
  // 기존: <video ref={videoRef} ... /> -> 제거
  // 변경: <img src={getStreamUrl(camId)} ... />

  // 출입 카메라 촬영 함수
  const captureFromAccessCamera = async () => {
    const currentStep = captureStepRef.current || captureStep;
    const stepLabels = {
      front: "\uC815\uBA74",
      left: "\uC67C\uCABD",
      right: "\uC624\uB978\uCABD",
    };
    const nextStepMap = {
      front: "left",
      left: "right",
      right: "complete",
    };
    const nextStep = nextStepMap[currentStep];

    const video = accessCameraVideoRef.current;
    const canvas = accessCameraCanvasRef.current;

    if (!video || !canvas) {
      console.error(
        "\uCE78\uCC98\uC6A9 \uBE44\uB514\uC624 \uB610\uB294 \uCEAC\uBC84\uC2A4\uAC00 \uC5C6\uC2B5\uB2C8\uB2E4."
      );
      setIsAutoCapturing(false);
      return;
    }

    // 비디오가 준비될 때까지 대기하는 헬퍼 함수
    const waitForVideoReady = async (maxWaitMs = 3000) => {
      const start = performance.now();
      while (performance.now() - start < maxWaitMs) {
        const track = video.srcObject?.getVideoTracks?.()[0];
        const isReady =
          track &&
          track.readyState === "live" &&
          video.readyState >= video.HAVE_CURRENT_DATA &&
          video.videoWidth > 0 &&
          video.videoHeight > 0 &&
          !video.paused;

        if (isReady) {
          // 추가로 실제 프레임이 렌더링되는지 확인
          await new Promise((resolve) => requestAnimationFrame(resolve));
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            return true;
          }
        }
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      return false;
    };

    // 비디오가 준비되지 않았으면 대기
    if (
      video.videoWidth === 0 ||
      video.videoHeight === 0 ||
      video.readyState < video.HAVE_CURRENT_DATA
    ) {
      setCaptureMessage("카메라 준비 중입니다. 잠시만 기다려주세요.");

      const isReady = await waitForVideoReady(3000);
      if (!isReady) {
        cameraReadyRef.current = false;
        setIsAutoCapturing(false);
        setCaptureMessage("카메라 초기화 중입니다. 잠시만 기다려주세요.");
        setTimeout(() => {
          if (startFaceDetectionRef.current) {
            startFaceDetectionRef.current();
          }
        }, 500);
        throw new Error("VIDEO_NOT_READY");
      }
    }

    // 비디오가 일시정지 상태면 재생 시도
    if (video.paused) {
      try {
        await video.play();
        await new Promise((resolve) => setTimeout(resolve, 200));
        // 재생 후 다시 확인
        if (video.videoWidth === 0 || video.videoHeight === 0) {
          const isReady = await waitForVideoReady(2000);
          if (!isReady) {
            throw new Error("VIDEO_NOT_READY");
          }
        }
      } catch (playErr) {
        console.warn("비디오 재생 실패:", playErr);
        throw new Error("VIDEO_NOT_READY");
      }
    }

    // 캔버스 크기를 600x600으로 고정
    canvas.width = 600;
    canvas.height = 600;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      console.error(
        "\uCEAC\uBC84\uC2A4 \uCEE8\uD14D\uC2A4\uD2B8\uB97C \uAC00\uC838\uC624\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4."
      );
      setIsAutoCapturing(false);
      return;
    }

    const tryImageCapture = async () => {
      // ImageCapture 사용 안 함, 항상 Canvas 사용 (안정성 향상)
      // InvalidStateError 문제를 완전히 제거하기 위해 Canvas만 사용

      try {
        // 비디오 트랙 상태 확인
        const track = video.srcObject?.getVideoTracks?.()[0];
        if (!track) {
          console.warn("비디오 트랙이 없습니다.");
          throw new Error("VIDEO_NOT_READY");
        }

        // 트랙이 ended 상태면 스트림 재시작 시도
        if (track.readyState === "ended") {
          console.warn("비디오 트랙이 종료되었습니다. 스트림 재시작 시도...");
          // 스트림 재시작 시도
          try {
            if (accessCameraStream) {
              const newStream = await navigator.mediaDevices.getUserMedia({
                video: {
                  width: 1280,
                  height: 720,
                  facingMode: "user",
                },
              });
              if (video.srcObject) {
                video.srcObject.getTracks().forEach((t) => t.stop());
              }
              video.srcObject = newStream;
              setAccessCameraStream(newStream);
              // 스트림이 준비될 때까지 대기
              await new Promise((resolve) => setTimeout(resolve, 500));
              // 비디오 재생
              await video.play();
              await new Promise((resolve) => setTimeout(resolve, 300));
            }
          } catch (restartErr) {
            console.warn("스트림 재시작 실패:", restartErr);
            throw new Error("VIDEO_NOT_READY");
          }
        }

        if (track.readyState !== "live") {
          console.warn(
            "비디오 트랙이 준비되지 않았습니다. track:",
            track?.readyState
          );
          throw new Error("VIDEO_NOT_READY");
        }

        // 비디오가 준비되었는지 엄격하게 확인
        if (!video || video.readyState < video.HAVE_CURRENT_DATA) {
          console.warn(
            "비디오가 아직 준비되지 않았습니다. readyState:",
            video?.readyState
          );
          // 최대 1초 대기
          let retries = 0;
          while (retries < 10 && video.readyState < video.HAVE_CURRENT_DATA) {
            await new Promise((resolve) => setTimeout(resolve, 100));
            retries++;
          }
          if (video.readyState < video.HAVE_CURRENT_DATA) {
            throw new Error("VIDEO_NOT_READY");
          }
        }

        if (video.videoWidth === 0 || video.videoHeight === 0) {
          console.warn(
            "비디오 해상도가 0입니다. videoWidth:",
            video.videoWidth,
            "videoHeight:",
            video.videoHeight
          );
          // 최대 1초 대기
          let retries = 0;
          while (
            retries < 10 &&
            (video.videoWidth === 0 || video.videoHeight === 0)
          ) {
            await new Promise((resolve) => setTimeout(resolve, 100));
            retries++;
          }
          if (video.videoWidth === 0 || video.videoHeight === 0) {
            throw new Error("VIDEO_NOT_READY");
          }
        }

        // 비디오가 실제로 재생 중인지 확인
        if (video.paused) {
          console.warn("비디오가 일시정지 상태입니다. 재생 시도 중...");
          try {
            await video.play();
            // 재생 후 프레임이 렌더링될 때까지 대기
            await new Promise((resolve) => requestAnimationFrame(resolve));
            await new Promise((resolve) => setTimeout(resolve, 150));

            // 재생 후 다시 확인
            if (
              video.videoWidth === 0 ||
              video.videoHeight === 0 ||
              video.readyState < video.HAVE_CURRENT_DATA
            ) {
              throw new Error("VIDEO_NOT_READY");
            }
          } catch (playErr) {
            console.warn("비디오 재생 실패:", playErr);
            throw new Error("VIDEO_NOT_READY");
          }
        }

        // Canvas를 사용하여 비디오 프레임 캡처
        const captureCanvas = document.createElement("canvas");
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        const captureCtx = captureCanvas.getContext("2d", {
          willReadFrequently: true,
        });

        if (!captureCtx) {
          console.error("Canvas 컨텍스트를 가져올 수 없습니다.");
          throw new Error("CANVAS_CONTEXT_FAILED");
        }

        // 비디오가 실제로 프레임을 렌더링할 때까지 대기
        // 여러 프레임을 확인하여 비디오가 실제로 작동하는지 확인
        let frameReady = false;
        for (let i = 0; i < 3; i++) {
          await new Promise((resolve) => requestAnimationFrame(resolve));
          // 비디오가 실제로 프레임을 제공하는지 확인
          if (
            video.readyState >= video.HAVE_CURRENT_DATA &&
            video.videoWidth > 0 &&
            video.videoHeight > 0 &&
            !video.paused
          ) {
            frameReady = true;
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, 50));
        }

        if (!frameReady) {
          console.warn("비디오 프레임이 준비되지 않았습니다.");
          throw new Error("VIDEO_NOT_READY");
        }

        // 비디오를 Canvas에 그리기 (좌우 반전)
        captureCtx.save();
        captureCtx.translate(captureCanvas.width, 0);
        captureCtx.scale(-1, 1);
        captureCtx.drawImage(
          video,
          0,
          0,
          captureCanvas.width,
          captureCanvas.height
        );
        captureCtx.restore();

        // Canvas를 Blob으로 변환
        const capturedBlob = await new Promise((resolve, reject) => {
          captureCanvas.toBlob(
            (b) => {
              if (b) {
                resolve(b);
              } else {
                reject(new Error("BLOB_CREATION_FAILED"));
              }
            },
            "image/jpeg",
            0.95
          );
        });

        if (capturedBlob && capturedBlob.size > 0) {
          return capturedBlob;
        }

        throw new Error("BLOB_CREATION_FAILED");
      } catch (err) {
        console.warn("Canvas 캡처 실패:", err.message || err);
        // VIDEO_NOT_READY는 재시도 가능하므로 throw
        if (err?.message === "VIDEO_NOT_READY") {
          throw err;
        }
        return null;
      }
    };

    const ensureFrameReady = async () => {
      const videoWasPaused = video.paused;
      const pauseVideo = async () => {
        try {
          video.pause();
          await new Promise((resolve) => setTimeout(resolve, 120));
        } catch (pauseErr) {
          console.warn("비디오 일시정지 실패:", pauseErr);
        }
      };
      const playVideo = async () => {
        try {
          await video.play();
        } catch (playErr) {
          console.warn("비디오 재생 실패:", playErr);
        }
      };

      if (!videoWasPaused && video.readyState >= video.HAVE_CURRENT_DATA) {
        await pauseVideo();
      }

      const drawFrame = () => {
        ctx.save();
        // 중앙 부분을 600x600으로 크롭
        const targetSize = 600; // canvas.width와 canvas.height는 모두 600

        // 비디오의 중앙 부분을 정사각형으로 크롭
        let sourceX = 0;
        let sourceY = 0;
        let sourceWidth = video.videoWidth;
        let sourceHeight = video.videoHeight;

        // 비디오가 더 넓으면 좌우를 자르고, 더 높으면 상하를 자름
        if (video.videoWidth > video.videoHeight) {
          // 비디오가 더 넓음 - 중앙 부분을 정사각형으로 크롭
          sourceWidth = video.videoHeight;
          sourceX = (video.videoWidth - sourceWidth) / 2;
        } else {
          // 비디오가 더 높음 - 중앙 부분을 정사각형으로 크롭
          sourceHeight = video.videoWidth;
          sourceY = (video.videoHeight - sourceHeight) / 2;
        }

        // 배경을 검은색으로 채우기
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // 좌우 반전하여 중앙 크롭 부분을 600x600으로 그리기
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(
          video,
          sourceX,
          sourceY,
          sourceWidth,
          sourceHeight, // 소스 영역 (비디오의 중앙 정사각형)
          -targetSize,
          0,
          targetSize,
          targetSize // 대상 영역 (600x600)
        );
        ctx.restore();
      };

      // 비디오 준비 확인 함수
      const checkVideoReady = async () => {
        // 비디오가 실제로 프레임을 렌더링하고 있는지 확인
        if (video.videoWidth === 0 || video.videoHeight === 0) {
          throw new Error("VIDEO_NOT_READY");
        }

        // 비디오 트랙이 live 상태인지 확인
        const track = video.srcObject?.getVideoTracks?.()[0];
        if (!track || track.readyState !== "live") {
          throw new Error("VIDEO_NOT_READY");
        }

        // 비디오가 재생 중인지 확인
        if (video.paused) {
          try {
            await video.play();
            await new Promise((resolve) => setTimeout(resolve, 100));
          } catch (err) {
            console.warn("비디오 재생 실패:", err);
            throw new Error("VIDEO_NOT_READY");
          }
        }

        // 비디오가 실제로 프레임을 렌더링할 때까지 대기
        let frameWaitRetries = 0;
        while (frameWaitRetries < 20) {
          if (video.readyState >= video.HAVE_CURRENT_DATA && !video.paused) {
            // 추가로 실제 프레임이 있는지 확인
            await new Promise((resolve) => requestAnimationFrame(resolve));
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, 50));
          frameWaitRetries++;
        }

        if (video.readyState < video.HAVE_CURRENT_DATA || video.paused) {
          throw new Error("VIDEO_NOT_READY");
        }
      };

      let frameIsValid = false;
      const MAX_FRAME_ATTEMPTS = 5; // 재시도 횟수 증가 (3 -> 5)

      for (let attempt = 0; attempt < MAX_FRAME_ATTEMPTS; attempt += 1) {
        try {
          // 비디오 준비 확인
          await checkVideoReady();

          if (attempt > 0) {
            await playVideo();
            await new Promise((resolve) => setTimeout(resolve, 200));
            await pauseVideo();
          }

          // 여러 프레임을 기다려서 안정적인 프레임 캡처
          await new Promise((resolve) => requestAnimationFrame(resolve));
          await new Promise((resolve) => requestAnimationFrame(resolve));
          await new Promise((resolve) => requestAnimationFrame(resolve));

          drawFrame();

          // 캔버스에 실제로 이미지가 그려졌는지 확인
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const hasContent = imageData.data.some((pixel, index) => {
            // 알파 채널이 아닌 픽셀 값 확인 (검은색이 아닌지)
            if (index % 4 === 3) return false; // 알파 채널 스킵
            return pixel > 10; // 검은색(0)이 아닌 값이 있는지 확인
          });

          if (!hasContent) {
            console.warn(
              `[시도 ${attempt + 1}] 캔버스에 이미지가 그려지지 않았습니다.`
            );
            if (attempt < MAX_FRAME_ATTEMPTS - 1) {
              // 비디오 재생 확인 및 재시도
              if (video.paused) {
                try {
                  await video.play();
                  await new Promise((resolve) =>
                    requestAnimationFrame(resolve)
                  );
                } catch (playErr) {
                  console.warn("비디오 재생 실패:", playErr);
                }
              }
              // 재시도 간격을 점진적으로 증가
              await new Promise((resolve) =>
                setTimeout(resolve, 300 + attempt * 100)
              );
              continue;
            }
            throw new Error("FRAME_NOT_READY");
          }

          frameIsValid = true;
          break;
        } catch (err) {
          if (
            err.message === "VIDEO_NOT_READY" ||
            err.message === "FRAME_NOT_READY"
          ) {
            if (attempt < MAX_FRAME_ATTEMPTS - 1) {
              console.warn(`[시도 ${attempt + 1}] 비디오 준비 대기 중...`);
              // 재시도 간격을 점진적으로 증가 (300ms -> 500ms -> 700ms...)
              await new Promise((resolve) =>
                setTimeout(resolve, 300 + attempt * 200)
              );
              continue;
            }
          }
          throw err;
        }
      }

      if (!frameIsValid) {
        cameraReadyRef.current = false;
        setIsAutoCapturing(false);
        pauseFaceDetectionRef.current = false;
        console.warn("프레임을 가져오지 못했습니다. 재시도합니다.", {
          step: currentStep,
        });
        setCaptureMessage(
          "화면을 안정시키고 다시 시도합니다. 잠시만 기다려주세요."
        );
        setTimeout(() => {
          if (startFaceDetectionRef.current) {
            startFaceDetectionRef.current();
          }
        }, 250);
        await playVideo();
        throw new Error("FRAME_NOT_READY");
      }

      return new Promise((resolve, reject) => {
        canvas.toBlob(
          (b) => {
            if (b) {
              resolve(b);
            } else {
              reject(new Error("이미지 생성 실패"));
            }
          },
          "image/jpeg",
          0.9
        );
      })
        .catch((error) => {
          console.error("이미지 생성 실패:", error);
          setCaptureMessage("이미지 생성에 실패했습니다. 다시 시도해주세요.");
          throw error;
        })
        .finally(async () => {
          if (!videoWasPaused) {
            await playVideo();
          }
        });
    };

    const MIN_BLOB_SIZE = 0;

    const acquireValidBlob = async () => {
      let lastError = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const useEnsure = attempt > 0;
          const capturedBlob = useEnsure
            ? await ensureFrameReady()
            : await tryImageCapture();

          if (!capturedBlob) {
            console.warn(
              `[${currentStep}] Blob 획득 실패 (시도 ${attempt + 1})`
            );
            lastError = new Error("CAPTURE_FAILED");
            // 다음 시도에서는 ensureFrameReady 사용
            if (attempt < 2) {
              await new Promise((resolve) => setTimeout(resolve, 300));
            }
            continue;
          }

          if (capturedBlob.size <= MIN_BLOB_SIZE) {
            console.warn(
              `[${currentStep}] 촬영된 이미지 용량이 매우 작습니다.`,
              {
                attempt,
                size: capturedBlob.size,
                step: currentStep,
              }
            );
            lastError = new Error("CAPTURE_TOO_SMALL");
            cameraReadyRef.current = false;
            if (attempt < 2) {
              await new Promise((resolve) => setTimeout(resolve, 300));
            }
            continue;
          }

          cameraReadyRef.current = true;
          return capturedBlob;
        } catch (err) {
          console.warn(
            `[${currentStep}] Blob 획득 중 에러 (시도 ${attempt + 1}):`,
            err.message || err
          );
          lastError = err;
          // VIDEO_NOT_READY나 FRAME_NOT_READY는 재시도 가능
          if (
            err?.message === "VIDEO_NOT_READY" ||
            err?.message === "FRAME_NOT_READY"
          ) {
            if (attempt < 2) {
              // 비디오 스트림 상태 확인 및 재시작 시도
              const track = video?.srcObject?.getVideoTracks?.()[0];
              if (track && track.readyState === "ended") {
                console.warn(
                  "비디오 트랙이 종료됨. 재시도 전 대기 시간 증가..."
                );
                await new Promise((resolve) => setTimeout(resolve, 1000));
              } else {
                await new Promise((resolve) =>
                  setTimeout(resolve, 500 + attempt * 200)
                );
              }
              continue;
            }
          }
          if (attempt < 2) {
            await new Promise((resolve) => setTimeout(resolve, 300));
          }
        }
      }

      console.error(`[${currentStep}] Blob 획득 최종 실패`);
      throw lastError || new Error("CAPTURE_FAILED");
    };

    let blob;
    try {
      blob = await acquireValidBlob();
    } catch (error) {
      if (error?.message === "CAPTURE_TOO_SMALL") {
        setIsAutoCapturing(false);
        pauseFaceDetectionRef.current = false;
        cameraReadyRef.current = false;
        setCaptureMessage(
          "촬영된 이미지가 너무 어둡거나 작습니다. 다시 가이드에 맞춰주세요."
        );
        setTimeout(() => {
          if (startFaceDetectionRef.current) {
            startFaceDetectionRef.current();
          }
        }, 300);
      } else if (error?.message === "CAPTURE_FAILED") {
        cameraReadyRef.current = false;
        setIsAutoCapturing(false);
        setCaptureMessage(
          "카메라 프레임을 가져오지 못했습니다. 잠시 후 다시 시도합니다."
        );
        setTimeout(() => {
          if (startFaceDetectionRef.current) {
            startFaceDetectionRef.current();
          }
        }, 300);
      }
      throw error;
    }

    if (!blob) {
      cameraReadyRef.current = false;
      setIsAutoCapturing(false);
      return;
    }

    cameraReadyRef.current = true;

    capturedStepsRef.current.add(currentStep);
    lastCaptureTimeRef.current = Date.now();

    const formData = new FormData();
    formData.append("image", blob, "capture.jpg");
    formData.append("camera_id", "0");
    formData.append("camera_name", "CAM001");
    formData.append("location", "정문");
    formData.append("step", currentStep);

    // workerCode를 안전하게 문자열로 변환 (객체인 경우 workerId 추출)
    // workerCodeRef를 우선 사용 (항상 최신 값 보장)
    const currentWorkerCode = workerCodeRef.current;

    let workerCodeString = "UNKNOWN";

    // workerCodeRef.current가 문자열인지 확인
    if (typeof currentWorkerCode === "string" && currentWorkerCode.trim()) {
      workerCodeString = currentWorkerCode.trim();
    } else if (typeof workerCode === "string" && workerCode.trim()) {
      // workerCodeRef가 없으면 workerCode state 사용
      workerCodeString = workerCode.trim();
    } else if (
      currentWorkerCode &&
      typeof currentWorkerCode === "object" &&
      currentWorkerCode.workerId
    ) {
      workerCodeString = String(currentWorkerCode.workerId).trim();
    } else if (
      workerCode &&
      typeof workerCode === "object" &&
      workerCode.workerId
    ) {
      workerCodeString = String(workerCode.workerId).trim();
    } else if (currentWorkerCode) {
      // 객체가 아닌 다른 타입인 경우 문자열로 변환 시도
      const strValue = String(currentWorkerCode);
      if (strValue && strValue !== "[object Object]" && strValue.trim()) {
        workerCodeString = strValue.trim();
      }
    } else if (workerCode) {
      // workerCode state도 확인
      const strValue = String(workerCode);
      if (strValue && strValue !== "[object Object]" && strValue.trim()) {
        workerCodeString = strValue.trim();
      }
    }

    // 최종 검증: workerCodeString이 여전히 [object Object]인 경우 처리
    if (
      workerCodeString === "[object Object]" ||
      typeof workerCodeString !== "string"
    ) {
      workerCodeString = "UNKNOWN";
    }

    // 최종적으로 문자열로 변환 보장
    workerCodeString = String(workerCodeString).trim();
    if (!workerCodeString || workerCodeString === "[object Object]") {
      workerCodeString = "UNKNOWN";
    }

    formData.append("worker_code", workerCodeString); // 작업자 코드 추가
    formData.append("session_id", sessionId || ""); // 세션 ID 추가

    const uploadPromise = api
      .captureFromStream(formData)
      .then((response) => {
        if (response && response.success) {
          // 세션 ID 업데이트 (백엔드에서 생성된 경우)
          if (response.session_id && !sessionId) {
            setSessionId(response.session_id);
          }

          const filename =
            response.filename ||
            (response.image_path
              ? response.image_path.split(/[/\\]/).pop()
              : null);

          if (filename) {
            // 이미지 경로는 서버에서 처리됨
            const savedLog = `서버 저장 완료: ${filename} (세션: ${
              response.session_id || sessionId || "없음"
            })`;
            debugLogsRef.current.push(savedLog);
            if (debugLogsRef.current.length > 10) {
              debugLogsRef.current.shift();
            }
            debugInfoRef.current = {
              ...debugInfoRef.current,
              log: savedLog,
              logs: [...debugLogsRef.current],
            };
          }
        }
      })
      .catch((error) => {
        console.error(`[${currentStep}] 서버 전송 실패:`, error);
        setCaptureMessage(
          `${stepLabels[currentStep]} 촬영 저장 실패. 다시 시도해주세요.`
        );
      })
      .finally(() => {
        setIsAutoCapturing(false);
        pauseFaceDetectionRef.current = false;
      });

    // 업로드 결과는 비동기로 처리하되, 다음 단계 진행은 즉시 계속한다.
    void uploadPromise;

    setIsAutoCapturing(false);
    pauseFaceDetectionRef.current = false;

    const debugLog = `\uCD2C\uC601 \uC644\uB8CC: ${stepLabels[currentStep]}`;
    debugLogsRef.current.push(debugLog);
    if (debugLogsRef.current.length > 10) {
      debugLogsRef.current.shift();
    }

    const timestamp = new Date().toISOString();

    if (nextStep === "complete") {
      const completeMessage = `✓ 모든 촬영이 완료되었습니다`;
      setCaptureMessage(completeMessage);
      setCaptureStep("complete");
      captureStepRef.current = "complete";
      debugInfoRef.current = {
        step: "complete",
        message: completeMessage,
        isAutoCapturing: false,
        log: debugLog,
        logs: [...debugLogsRef.current],
        timestamp,
      };

      // 딜레이 없이 즉시 상태 초기화 및 작업자 입력 UI 표시
      capturedStepsRef.current.clear();
      cameraReadyRef.current = false;
      pauseFaceDetectionRef.current = false;
      countdownWaitingRef.current = false;
      lastCaptureTimeRef.current = 0;
      setFaceInGuide(false);
      setCountdown(0);
      countdownCountRef.current = 0;
      lastMessageRef.current = "";
      setCaptureStep("front");
      captureStepRef.current = "front";
      setCaptureMessage("얼굴을 가이드에 맞춰주세요");

      // 완료 처리
      setShowWorkerInput(true);
      setWorkerCode("");
      setWorkerName("");
      setSessionId(null);
    } else {
      const nextLabel = stepLabels[nextStep];
      const nextMessage = `${nextLabel} 촬영 준비`;
      setCaptureMessage(nextMessage);
      setCaptureStep(nextStep);
      captureStepRef.current = nextStep;
      debugInfoRef.current = {
        step: nextStep,
        message: nextMessage,
        isAutoCapturing: false,
        log: debugLog,
        logs: [...debugLogsRef.current],
        timestamp,
      };

      setTimeout(() => {
        // 실제 준비 상태 확인 후 얼굴 감지 시작
        const checkAndStart = () => {
          const video = accessCameraVideoRef.current;
          const isVideoReady =
            video &&
            video.readyState >= video.HAVE_ENOUGH_DATA &&
            video.videoWidth > 0 &&
            video.videoHeight > 0;

          const isModelsReady =
            modelsLoaded &&
            faceApiLoaded &&
            typeof window.faceapi !== "undefined";

          if (isVideoReady && isModelsReady) {
            if (startFaceDetectionRef.current) {
              startFaceDetectionRef.current();
            }
          } else {
            setCaptureMessage("장비 준비 중입니다...");
            // 준비될 때까지 대기 (최대 10초)
            const startTime = Date.now();
            const maxWaitMs = 10000;

            if (warmupCheckIntervalRef.current) {
              clearInterval(warmupCheckIntervalRef.current);
            }

            warmupCheckIntervalRef.current = setInterval(() => {
              const video = accessCameraVideoRef.current;
              const isReady =
                video &&
                video.readyState >= video.HAVE_ENOUGH_DATA &&
                video.videoWidth > 0 &&
                video.videoHeight > 0 &&
                modelsLoaded &&
                faceApiLoaded &&
                typeof window.faceapi !== "undefined";

              if (isReady) {
                clearInterval(warmupCheckIntervalRef.current);
                warmupCheckIntervalRef.current = null;
                if (startFaceDetectionRef.current) {
                  startFaceDetectionRef.current();
                }
              } else if (Date.now() - startTime > maxWaitMs) {
                clearInterval(warmupCheckIntervalRef.current);
                warmupCheckIntervalRef.current = null;
                setCaptureMessage("장비 준비 시간이 초과되었습니다.");
              }
            }, 200);
          }
        };

        checkAndStart();
      }, 300);
    }

    setCountdown(0);
    countdownCountRef.current = 0;
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }

    return { currentStep, nextStep };
  };

  // Calendar
  const [currentDate, setCurrentDate] = useState(new Date());
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [selectedDates, setSelectedDates] = useState([]); // 개별 선택된 날짜들
  const [selectedRiskType, setSelectedRiskType] = useState(null);
  const [hoveredDate, setHoveredDate] = useState(null); // 호버된 날짜
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 }); // 툴팁 위치
  const hoverTimeoutRef = useRef(null); // 호버 딜레이를 위한 timeout ref
  const [sortOrder, setSortOrder] = useState(null);

  // 빠른 기간 선택(캘린더)
  const [rangeMode, setRangeMode] = useState(null);

  // 리포트 탭 활성화 시 현재 날짜로 달력 초기화
  useEffect(() => {
    if (activePage === "calendar") {
      setCurrentDate(new Date());
    }
  }, [activePage]);

  // 언어 변경 시 selectedRiskType과 sortOrder 초기화
  useEffect(() => {
    setSelectedRiskType(t.calendar.all);
    setSortOrder(t.calendar.newest);
  }, [language, t]);

  // 테마에 따라 CSS 변수 변경
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") {
      root.style.setProperty("--main-bg", "#111827");
      root.style.setProperty("--card-bg", "#1F2937");
      root.style.setProperty("--text-primary", "#F9FAFB");
      root.style.setProperty("--text-secondary", "#9CA3AF");
      root.style.setProperty("--border-color", "#374151");
      root.style.setProperty("--sidebar-bg", "#0F172A");
      root.style.setProperty("--pending-bg", "#451A03");
      root.style.setProperty("--pending-text", "#FBBF24");
      root.style.setProperty("--draft-bg", "#1E3A8A");
      root.style.setProperty("--draft-text", "#3b82f6");
      root.style.setProperty("--badge-new-bg", "#1E1B4B");
      root.style.setProperty("--badge-new-hover", "#312E81");
      root.style.setProperty("--badge-new-text", "#C7D2FE");
      root.style.setProperty("--badge-ack-bg", "#431407");
      root.style.setProperty("--badge-ack-hover", "#7C2D12");
      root.style.setProperty("--badge-ack-text", "#FED7AA");
      root.style.setProperty("--badge-working-bg", "#1E3A5F");
      root.style.setProperty("--badge-working-hover", "#1E40AF");
      root.style.setProperty("--badge-working-text", "#DBEAFE");
      root.style.setProperty("--badge-done-bg", "#052E16");
      root.style.setProperty("--badge-done-text", "#86EFAC");
      root.style.setProperty("--kpi-red-bg", "#7F1D1D");
      root.style.setProperty("--kpi-red-text", "#FCA5A5");
      root.style.setProperty("--kpi-orange-bg", "#7C2D12");
      root.style.setProperty("--kpi-orange-text", "#FDBA74");
      root.style.setProperty("--calendar-bg", "#1F2937");
      root.style.setProperty("--calendar-hover", "#374151");
      root.style.setProperty("--calendar-range", "#1E3A8A");
      root.style.setProperty("--table-header-bg", "#1F2937");
      root.style.setProperty("--avatar-outline", "#374151");
      root.style.setProperty("--log-icon-bg", "#7F1D1D");
      root.style.setProperty("--chip-container-bg", "#374151");
      root.style.setProperty("--chip-bg", "transparent");
      root.style.setProperty("--chip-active-bg", "#4B5563");
      root.style.setProperty("--chip-active-border", "#6B7280");
      root.style.setProperty("--chip-hover-bg", "rgba(255,255,255,0.1)");
    } else {
      root.style.setProperty("--main-bg", "#F8FAFC");
      root.style.setProperty("--card-bg", "#FFFFFF");
      root.style.setProperty("--text-primary", "#1F2937");
      root.style.setProperty("--text-secondary", "#6B7280");
      root.style.setProperty("--border-color", "#E5E7EB");
      root.style.setProperty("--sidebar-bg", "#111827");
      root.style.setProperty("--pending-bg", "#FFFBEB");
      root.style.setProperty("--pending-text", "#B45309");
      root.style.setProperty("--draft-bg", "#EFF6FF");
      root.style.setProperty("--draft-text", "#3B82F6");
      root.style.setProperty("--badge-new-bg", "#EEF2FF");
      root.style.setProperty("--badge-new-hover", "#E0E7FF");
      root.style.setProperty("--badge-new-text", "#3730A3");
      root.style.setProperty("--badge-ack-bg", "#FFF7ED");
      root.style.setProperty("--badge-ack-hover", "#FFEDD5");
      root.style.setProperty("--badge-ack-text", "#9A3412");
      root.style.setProperty("--badge-working-bg", "#DBEAFE");
      root.style.setProperty("--badge-working-hover", "#BFDBFE");
      root.style.setProperty("--badge-working-text", "#1D4ED8");
      root.style.setProperty("--badge-done-bg", "#DCFCE7");
      root.style.setProperty("--badge-done-text", "#166534");
      root.style.setProperty("--kpi-red-bg", "#FEF2F2");
      root.style.setProperty("--kpi-red-text", "#EF4444");
      root.style.setProperty("--kpi-orange-bg", "#FFF4E6");
      root.style.setProperty("--kpi-orange-text", "#FF8800");
      root.style.setProperty("--calendar-bg", "#F9FAFB");
      root.style.setProperty("--calendar-hover", "#EFF6FF");
      root.style.setProperty("--calendar-range", "#DBEAFE");
      root.style.setProperty("--table-header-bg", "#F9FAFB");
      root.style.setProperty("--avatar-outline", "#E5E7EB");
      root.style.setProperty("--log-icon-bg", "#FEF2F2");
      root.style.setProperty("--chip-container-bg", "#F3F4F6");
      root.style.setProperty("--chip-bg", "transparent");
      root.style.setProperty("--chip-active-bg", "#fff");
      root.style.setProperty("--chip-active-border", "#E5E7EB");
      root.style.setProperty("--chip-hover-bg", "rgba(255,255,255,0.7)");
    }
  }, [theme]);
  function applyQuickRange(mode) {
    const today = new Date();
    today.setHours(23, 59, 59, 999); // 오늘의 끝 시간으로 설정
    const dow = today.getDay();
    const mondayOffset = dow === 0 ? -6 : 1 - dow;
    const weekStart = new Date(
      today.getFullYear(),
      today.getMonth(),
      today.getDate() + mondayOffset
    );
    weekStart.setHours(0, 0, 0, 0);
    const weekEnd = new Date(
      weekStart.getFullYear(),
      weekStart.getMonth(),
      weekStart.getDate() + 6
    );
    weekEnd.setHours(23, 59, 59, 999);

    setRangeMode(mode);
    // 빠른 기간 선택 시 개별 날짜 선택 초기화
    setSelectedDates([]);

    if (mode === "week") {
      setStartDate(weekStart);
      setEndDate(weekEnd);
      return;
    }
    if (mode === "1m") {
      // 현재 날짜 기준 1개월 전 ~ 현재 날짜
      const oneMonthAgo = new Date(today);
      oneMonthAgo.setMonth(today.getMonth() - 1);
      oneMonthAgo.setDate(today.getDate());
      oneMonthAgo.setHours(0, 0, 0, 0);
      setStartDate(oneMonthAgo);
      setEndDate(today);
      return;
    }
    if (mode === "3m") {
      // 현재 날짜 기준 3개월 전 ~ 현재 날짜
      const threeMonthsAgo = new Date(today);
      threeMonthsAgo.setMonth(today.getMonth() - 3);
      threeMonthsAgo.setDate(today.getDate());
      threeMonthsAgo.setHours(0, 0, 0, 0);
      setStartDate(threeMonthsAgo);
      setEndDate(today);
      return;
    }
    if (mode === "prev") {
      // 지난달 1일 ~ 지난달 마지막 날
      const lastMonth = new Date(today.getFullYear(), today.getMonth() - 1, 1);
      lastMonth.setHours(0, 0, 0, 0);
      const lastMonthEnd = new Date(today.getFullYear(), today.getMonth(), 0);
      lastMonthEnd.setHours(23, 59, 59, 999);
      setStartDate(lastMonth);
      setEndDate(lastMonthEnd);
      return;
    }
    if (mode === "year") {
      // 올해 1월 1일 ~ 현재 날짜
      const yearStart = new Date(today.getFullYear(), 0, 1);
      yearStart.setHours(0, 0, 0, 0);
      setStartDate(yearStart);
      setEndDate(today);
      return;
    }

    setStartDate(null);
    setEndDate(null);
  }

  const daysForCalendar = useMemo(() => {
    const year = currentDate.getFullYear();
    const month = currentDate.getMonth();
    const firstDay = new Date(year, month, 1).getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const prevDaysInMonth = new Date(year, month, 0).getDate();

    const cells = [];
    for (let i = firstDay - 1; i >= 0; i--)
      cells.push({ label: prevDaysInMonth - i, date: null, current: false });
    for (let d = 1; d <= daysInMonth; d++)
      cells.push({ label: d, date: new Date(year, month, d), current: true });
    return cells;
  }, [currentDate]); // fullLogs는 렌더링 시 직접 사용하므로 의존성에 포함하지 않음

  const inRange = (d) => {
    if (!d) return false;
    // 로컬 시간 기준으로 날짜 문자열 생성
    const formatDate = (date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };

    // 개별 선택된 날짜가 있으면 그것만 확인
    if (selectedDates.length > 0) {
      const dateStr = formatDate(d);
      return selectedDates.some(
        (selectedDate) => formatDate(selectedDate) === dateStr
      );
    }
    // 개별 선택이 없으면 기존 범위 선택 사용
    if (!startDate || !endDate) return false;
    const dateStr = formatDate(d);
    const startDateStr = formatDate(startDate);
    const endDateStr = formatDate(endDate);
    return dateStr >= startDateStr && dateStr <= endDateStr;
  };
  const isSameDay = (a, b) =>
    a && b && a.setHours(0, 0, 0, 0) === b.setHours(0, 0, 0, 0);

  const isDateSelected = (d) => {
    if (!d || selectedDates.length === 0) return false;
    // 로컬 시간 기준으로 날짜 문자열 생성
    const formatDate = (date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };
    const dateStr = formatDate(d);
    return selectedDates.some(
      (selectedDate) => formatDate(selectedDate) === dateStr
    );
  };
  // 드래그 상태 관리
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef(null);
  const mouseDownPosRef = useRef(null); // 마우스 다운 위치 저장
  const hasMovedRef = useRef(false); // 마우스가 실제로 움직였는지 확인

  const handleDayClick = (d) => {
    if (!d) return;
    // 드래그가 아니고 실제 클릭인 경우만 처리
    if (!isDragging && !hasMovedRef.current) {
      const clickedDate = new Date(d);
      clickedDate.setHours(0, 0, 0, 0);

      // 올해 모드일 때 현재 날짜 이후는 선택 불가
      if (rangeMode === "year") {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        if (clickedDate > today) {
          return; // 현재 날짜 이후는 선택 불가
        }
      }

      // 로컬 시간 기준으로 날짜 문자열 생성
      const formatDate = (date) => {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, "0");
        const day = String(date.getDate()).padStart(2, "0");
        return `${year}-${month}-${day}`;
      };
      const clickedDateStr = formatDate(clickedDate);

      // 선택된 날짜 배열에서 해당 날짜 찾기
      const dateIndex = selectedDates.findIndex(
        (date) => formatDate(date) === clickedDateStr
      );

      let updatedSelectedDates;
      if (dateIndex === -1) {
        // 선택되지 않은 날짜면 추가
        updatedSelectedDates = [...selectedDates, clickedDate];
      } else {
        // 이미 선택된 날짜면 제거
        updatedSelectedDates = selectedDates.filter(
          (_, idx) => idx !== dateIndex
        );
      }
      setSelectedDates(updatedSelectedDates);

      // 호환성을 위해 startDate와 endDate도 업데이트 (드래그용)
      if (updatedSelectedDates.length === 0) {
        setStartDate(null);
        setEndDate(null);
      } else {
        const sortedDates = updatedSelectedDates.sort((a, b) => a - b);
        setStartDate(sortedDates[0]);
        setEndDate(sortedDates[sortedDates.length - 1]);
      }
    }
    // 클릭 후 상태 초기화
    hasMovedRef.current = false;
  };

  const handleDayMouseDown = (d, e) => {
    if (!d) return;
    const date = new Date(d);
    mouseDownPosRef.current = { x: e.clientX, y: e.clientY };
    hasMovedRef.current = false;
    dragStartRef.current = date;
  };

  const handleDayMouseMove = (d, e) => {
    if (!d || !mouseDownPosRef.current || !dragStartRef.current) return;

    // 마우스가 5px 이상 움직였는지 확인
    const moveDistance = Math.sqrt(
      Math.pow(e.clientX - mouseDownPosRef.current.x, 2) +
        Math.pow(e.clientY - mouseDownPosRef.current.y, 2)
    );

    if (moveDistance > 5) {
      // 실제로 드래그가 시작됨
      if (!isDragging) {
        setIsDragging(true);
        hasMovedRef.current = true;
        const start = dragStartRef.current;
        start.setHours(0, 0, 0, 0);

        // 올해 모드일 때 현재 날짜 이후는 선택 불가
        if (rangeMode === "year") {
          const today = new Date();
          today.setHours(0, 0, 0, 0);
          if (start > today) {
            return; // 시작 날짜가 현재 날짜 이후면 드래그 불가
          }
        }

        // 드래그 시작 시 개별 선택 초기화하고 범위 선택으로 전환
        setSelectedDates([]);
        setStartDate(start);
        setEndDate(start);
      }

      // 드래그 중 범위 업데이트
      const date = new Date(d);
      date.setHours(0, 0, 0, 0);
      const start = dragStartRef.current;
      start.setHours(0, 0, 0, 0);

      // 올해 모드일 때 현재 날짜 이후는 선택 불가
      if (rangeMode === "year") {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        if (date > today) {
          // 현재 날짜까지만 선택 가능
          if (start <= today) {
            setStartDate(start);
            setEndDate(today);
          }
          return;
        }
      }

      if (date < start) {
        setStartDate(date);
        setEndDate(start);
      } else {
        setStartDate(start);
        setEndDate(date);
      }
    }
  };

  const handleDayMouseEnter = (d) => {
    if (!d) return;

    // 드래그 중이면 기존 로직 실행
    if (isDragging && dragStartRef.current) {
      const date = new Date(d);
      date.setHours(0, 0, 0, 0);
      const start = dragStartRef.current;
      start.setHours(0, 0, 0, 0);

      // 올해 모드일 때 현재 날짜 이후는 선택 불가
      if (rangeMode === "year") {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        if (date > today) {
          // 현재 날짜까지만 선택 가능
          if (start <= today) {
            setStartDate(start);
            setEndDate(today);
          }
          return;
        }
      }

      if (date < start) {
        setStartDate(date);
        setEndDate(start);
      } else {
        setStartDate(start);
        setEndDate(date);
      }
      return;
    }

    // 기존 timeout이 있으면 취소
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }

    // 드래그가 아닐 때는 0.5초 딜레이 후 호버 미리보기 표시
    const date = new Date(d);
    hoverTimeoutRef.current = setTimeout(() => {
      setHoveredDate(date);
      // useEffect에서 위치를 업데이트하므로 여기서는 날짜만 설정
    }, 500);
  };

  // hoveredDate가 변경될 때마다 해당 셀의 위치를 업데이트
  useEffect(() => {
    if (!hoveredDate) return;

    // 해당 날짜의 셀을 찾기
    const dateStr = hoveredDate.toISOString();
    const cellElement = document.querySelector(
      `.calendar-grid .day[data-date="${dateStr}"]`
    );

    if (cellElement) {
      const rect = cellElement.getBoundingClientRect();
      setTooltipPosition({ x: rect.right, y: rect.top });
    }
  }, [hoveredDate]);

  const handleDayMouseLeave = () => {
    if (!isDragging) {
      // timeout 취소
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
        hoverTimeoutRef.current = null;
      }
      setHoveredDate(null);
    }
  };

  const handleDayMouseUp = () => {
    setIsDragging(false);
    dragStartRef.current = null;
    mouseDownPosRef.current = null;
    hasMovedRef.current = false;
  };

  // 캘린더 영역을 벗어나면 드래그 종료
  useEffect(() => {
    const handleMouseUp = () => {
      if (isDragging) {
        setIsDragging(false);
        dragStartRef.current = null;
      }
    };

    const handleMouseLeave = () => {
      if (isDragging) {
        setIsDragging(false);
        dragStartRef.current = null;
      }
    };

    document.addEventListener("mouseup", handleMouseUp);
    const calendarBody = document.getElementById("calendarBody");
    if (calendarBody) {
      calendarBody.addEventListener("mouseleave", handleMouseLeave);
    }

    return () => {
      document.removeEventListener("mouseup", handleMouseUp);
      if (calendarBody) {
        calendarBody.removeEventListener("mouseleave", handleMouseLeave);
      }
    };
  }, [isDragging]);

  const periodDetails = useMemo(() => {
    const locale = language === "ko" ? "ko-KR" : "en-US";

    // 디버깅: fullLogs 상태 확인
    console.log("[리포트] periodDetails 계산 시작:", {
      fullLogsLength: fullLogs.length,
      selectedDatesLength: selectedDates.length,
      startDate: startDate,
      endDate: endDate,
      selectedRiskType: selectedRiskType,
      sortOrder: sortOrder,
    });

    // 개별 선택된 날짜가 있으면 그것만 사용
    if (selectedDates.length > 0) {
      // 선택된 날짜들의 문자열 배열 생성 (로컬 시간 기준)
      const selectedDateStrs = selectedDates.map((date) => {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, "0");
        const day = String(date.getDate()).padStart(2, "0");
        return `${year}-${month}-${day}`;
      });

      // DB에서 위반 사항 로드하여 캘린더에 표시
      const items = [];
      fullLogs.forEach((log) => {
        if (!log || !log.time) return;
        try {
          const logDateStr = log.time.split(" ")[0];
          // 선택된 날짜 중 하나와 일치하는지 확인
          if (selectedDateStrs.includes(logDateStr)) {
            const logDate = new Date(log.time);
            if (isNaN(logDate.getTime())) {
              console.warn("[리포트] 날짜 파싱 실패:", log.time);
              return;
            }
            const hour = logDate.getHours();
            const minute = logDate.getMinutes();
            const timeStr = `${hour}:${String(minute).padStart(2, "0")} ${
              hour >= 12 ? "PM" : "AM"
            }`;
            items.push({
              day: logDate.getDate(),
              risk: log.risk || "알 수 없음",
              who: mapWorkerName(log.worker || "알 수 없음"),
              time: timeStr,
              dateTime: logDate,
            });
          }
        } catch (e) {
          console.warn("[리포트] 로그 처리 오류:", e, log);
        }
      });

      // 유형 필터링 (유연한 매칭)
      let filteredItems = items;
      if (selectedRiskType && selectedRiskType !== t.calendar.all) {
        const riskMapReverse = {
          [t.dashboard.unhelmet]: ["안전모", "helmet", "헬멧"],
          [t.dashboard.unvest]: ["안전조끼", "vest", "조끼"],
          [t.dashboard.fall]: ["넘어짐", "fall", "낙상"],
        };
        const keywords = riskMapReverse[selectedRiskType] || [selectedRiskType];
        filteredItems = items.filter((item) => {
          if (!item || !item.risk) return false;
          const risk = String(item.risk).toLowerCase();
          return keywords.some((keyword) =>
            risk.includes(keyword.toLowerCase())
          );
        });
      }

      // 정렬
      const sortedItems = [...filteredItems].sort((a, b) => {
        if (sortOrder === t.calendar.newest) {
          return b.dateTime - a.dateTime;
        } else {
          return a.dateTime - b.dateTime;
        }
      });

      // 선택된 날짜들을 정렬하여 표시
      const sortedSelectedDates = [...selectedDates].sort((a, b) => a - b);
      const dateStrs = sortedSelectedDates.map((date) =>
        date.toLocaleDateString(locale)
      );
      const title =
        dateStrs.length === 1
          ? dateStrs[0]
          : `${dateStrs[0]} ~ ${dateStrs[dateStrs.length - 1]} (${
              dateStrs.length
            }일)`;

      return {
        title,
        items: sortedItems,
        hint:
          sortedItems.length === 0
            ? t.calendar.noData
            : `${sortedItems.length}${language === "ko" ? "건" : " items"}`,
      };
    }

    // 개별 선택이 없으면 기존 범위 선택 사용
    if (!startDate)
      return {
        title: t.calendar.selectStartDate,
        items: [],
        hint: t.calendar.noData,
      };
    if (!endDate)
      return {
        title: `${t.calendar.startPrefix}${startDate.toLocaleDateString(
          locale
        )}`,
        items: [],
        hint: t.calendar.selectEndDate,
      };

    // DB에서 위반 사항 로드하여 캘린더에 표시
    const items = [];
    // fullLogs에서 선택된 기간의 데이터 필터링 (로컬 시간 기준)
    const formatDate = (date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };
    const startDateStr = formatDate(startDate);
    const endDateStr = formatDate(endDate);

    console.log("[리포트] 기간 선택 모드:", {
      startDateStr,
      endDateStr,
      fullLogsLength: fullLogs.length,
      fullLogsSample: fullLogs.slice(0, 3),
    });

    fullLogs.forEach((log) => {
      if (!log.time) return;
      const logDateStr = log.time.split(" ")[0];
      if (logDateStr >= startDateStr && logDateStr <= endDateStr) {
        const logDate = new Date(log.time);
        const hour = logDate.getHours();
        const minute = logDate.getMinutes();
        const timeStr = `${hour}:${String(minute).padStart(2, "0")} ${
          hour >= 12 ? "PM" : "AM"
        }`;
        items.push({
          day: logDate.getDate(),
          risk: log.risk,
          who: log.worker,
          time: timeStr,
          dateTime: logDate,
        });
      }
    });

    // 유형 필터링 (유연한 매칭)
    let filteredItems = items;
    if (selectedRiskType && selectedRiskType !== t.calendar.all) {
      const riskMapReverse = {
        [t.dashboard.unhelmet]: ["안전모", "helmet", "헬멧"],
        [t.dashboard.unvest]: ["안전조끼", "vest", "조끼"],
        [t.dashboard.fall]: ["넘어짐", "fall", "낙상"],
      };
      const keywords = riskMapReverse[selectedRiskType] || [selectedRiskType];
      filteredItems = items.filter((item) => {
        if (!item || !item.risk) return false;
        const risk = String(item.risk).toLowerCase();
        return keywords.some((keyword) => risk.includes(keyword.toLowerCase()));
      });
    }

    // 정렬
    const sortedItems = [...filteredItems].sort((a, b) => {
      if (sortOrder === t.calendar.newest) {
        return b.dateTime - a.dateTime;
      } else {
        return a.dateTime - b.dateTime;
      }
    });

    const startStr = startDate.toLocaleDateString(locale);
    const endStr = endDate.toLocaleDateString(locale);

    console.log("[리포트] periodDetails 결과:", {
      title: `${startStr} ~ ${endStr}`,
      itemsCount: sortedItems.length,
      itemsSample: sortedItems.slice(0, 3),
    });

    return {
      title: `${startStr} ~ ${endStr}`,
      items: sortedItems,
      hint:
        sortedItems.length === 0
          ? t.calendar.noData
          : `${sortedItems.length}${language === "ko" ? "건" : " items"}`,
    };
  }, [
    startDate,
    endDate,
    selectedDates,
    fullLogs,
    selectedRiskType,
    sortOrder,
    language,
    t,
  ]);

  /* =======================
     리포트 내보내기 공통 유틸
     ======================= */

  // 선택한 기간에 따른 데이터 추출 (캘린더에서 선택한 기간 사용)
  function getRowsBySelectedPeriod() {
    if (!fullLogs || fullLogs.length === 0) {
      return [];
    }

    const formatDate = (date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };

    let result = [];

    // 개별 선택된 날짜가 있으면 그것만 사용
    if (selectedDates.length > 0) {
      const selectedDateStrs = selectedDates.map((date) => formatDate(date));
      result = fullLogs
        .filter((r) => {
          if (!r.time) return false;
          const logDateStr = r.time.split(" ")[0];
          return selectedDateStrs.includes(logDateStr);
        })
        .map((r, idx) => ({
          No: idx + 1,
          발생일시: r.time || "",
          작업자: r.worker || "",
          위험유형: translateRiskType(r.risk, language, t) || r.risk || "",
          구역: r.zone || "",
          상태: r.status === "normal" ? "조치 완료" : "확인 필요",
        }));
    } else if (startDate && endDate) {
      // 범위 선택 사용
      const startDateStr = formatDate(startDate);
      const endDateStr = formatDate(endDate);
      result = fullLogs
        .filter((r) => {
          if (!r.time) return false;
          const logDate = new Date(r.time);
          logDate.setHours(0, 0, 0, 0);
          const logDateStr = formatDate(logDate);
          return logDateStr >= startDateStr && logDateStr <= endDateStr;
        })
        .map((r, idx) => ({
          No: idx + 1,
          발생일시: r.time || "",
          작업자: r.worker || "",
          위험유형: translateRiskType(r.risk, language, t) || r.risk || "",
          구역: r.zone || "",
          상태: r.status === "normal" ? "조치 완료" : "확인 필요",
        }));
    }

    // 최신순 정렬
    result.sort((a, b) => {
      if (!a.발생일시 || !b.발생일시) return 0;
      return new Date(b.발생일시) - new Date(a.발생일시);
    });

    // No 재정렬
    result = result.map((r, idx) => ({
      ...r,
      No: idx + 1,
    }));

    return result;
  }

  // 파일명 규칙
  function makeFileName(prefix) {
    const now = new Date();
    const yyyy = now.getFullYear();
    const mm = String(now.getMonth() + 1).padStart(2, "0");
    const dd = String(now.getDate()).padStart(2, "0");
    return `AIVIS_${prefix}_${yyyy}${mm}${dd}`;
  }

  // 엑셀 공통 내보내기 (한글 인코딩 설정)
  function exportXLSX(rows, prefix) {
    if (!rows || rows.length === 0) {
      alert("내보낼 데이터가 없습니다.");
      return;
    }

    const headerOrder = [
      "No",
      "발생일시",
      "작업자",
      "위험유형",
      "구역",
      "상태",
    ];
    const aoa = [headerOrder, ...rows.map((r) => headerOrder.map((h) => r[h]))];

    const ws = XLSX.utils.aoa_to_sheet(aoa);

    // 컬럼폭(간단 Auto-fit)
    const colWidths = headerOrder.map((h, i) => {
      const maxLen = Math.max(...aoa.map((row) => String(row[i] ?? "").length));
      // 발생일시/작업자/위험유형은 넉넉히
      const base =
        h === "발생일시"
          ? 20
          : h === "작업자"
          ? 16
          : h === "위험유형"
          ? 14
          : 10;
      return { wch: Math.max(base, maxLen + 2) };
    });
    ws["!cols"] = colWidths;

    // 첫 행 고정 + 필터
    ws["!autofilter"] = {
      ref: `A1:${String.fromCharCode(64 + headerOrder.length)}${
        rows.length + 1
      }`,
    };

    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Report");

    // 파일 다운로드 (한글 인코딩 자동 처리)
    const fileName = `${makeFileName(prefix)}.xlsx`;
    XLSX.writeFile(wb, fileName);
  }

  // 선택한 기간에 따른 리포트 내보내기 함수
  function exportSelectedPeriodXLSX() {
    const rows = getRowsBySelectedPeriod();
    if (rows.length === 0) {
      alert("선택한 기간에 데이터가 없습니다.");
      return;
    }

    const formatDate = (date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}${month}${day}`;
    };

    let prefix = "기간리포트";
    if (selectedDates.length > 0) {
      prefix = `선택기간_${selectedDates.length}일`;
    } else if (startDate && endDate) {
      prefix = `기간리포트_${formatDate(startDate)}_${formatDate(endDate)}`;
    }

    exportXLSX(rows, prefix);
  }

  async function exportSelectedPeriodPDF() {
    // 템플릿 PDF 기반 리포트 생성
    try {
      await generateSummaryPdf(startDate, endDate, selectedDates);
    } catch (error) {
      console.error("[PDF 생성] 오류:", error);
      // generateSummaryPdf 내부에서 이미 alert를 표시하므로 여기서는 로그만
    }
  }

  /* ======================= JSX ======================= */

  // 드롭다운 외부 클릭 시 닫기
  useEffect(() => {
    const handleClickOutside = (e) => {
      // 버튼 자체를 클릭한 경우는 무시 (버튼의 onClick이 처리)
      if (
        e.target.closest(".sidebar-action-btn") ||
        e.target.closest(".sidebar-dropdown") ||
        e.target.closest(".header-actions")
      ) {
        return;
      }

      // 사이드바 액션 영역 내부이지만 버튼이 아닌 경우도 체크
      const sidebarActions = e.target.closest(".sidebar-actions");
      if (sidebarActions) {
        // sidebar-action-item 내부이지만 버튼이 아닌 경우만 무시
        const actionItem = e.target.closest(".sidebar-action-item");
        if (actionItem && !e.target.closest(".sidebar-action-btn")) {
          return;
        }
      }

      if (showNotifications) {
        // 알림 드롭다운이 열려있을 때 외부 클릭으로 닫히면 배지 초기화
        setPendingCount(0);
      }
      setShowSearch(false);
      setShowNotifications(false);
      setShowSettings(false);
    };
    // mousedown 이벤트 사용 (버튼의 click 이벤트보다 먼저 발생하지만, 버튼 클릭 핸들러에서 stopPropagation으로 차단)
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showNotifications, showSearch, showSettings]);

  return (
    <div
      className={`min-h-screen flex flex-col ${dense ? "dense" : ""}`}
      style={{
        backgroundColor: "var(--main-bg)",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <header className="top-header">
        {/* 헤더는 최소화 - 필요시 나중에 추가 */}
      </header>

      <div className="main-layout">
        {/* 좌측 사이드바 */}
        <aside className="sidebar">
          {/* 사이드바 로고 */}
          <div
            className="sidebar-logo"
            title="AIVIS"
            onClick={() => setActivePage("dashboard")}
            style={{ cursor: "pointer" }}
          >
            <img
              className="logo-image"
              src="/assets/aivis-logo.png"
              alt="AIVIS"
            />
          </div>

          {/* 날씨 정보 및 시간 표시 (사이드바에서 숨김) */}
          <div className="sidebar-weather-time" style={{ display: "none" }}>
            {/* 날씨 정보 */}
            {weatherLoading ? (
              <div
                style={{
                  fontSize: "13px",
                  color: "var(--text-secondary)",
                  textAlign: "center",
                  padding: "8px",
                }}
              >
                로딩 중...
              </div>
            ) : weather ? (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "6px",
                  padding: "12px",
                  background: "var(--chip-container-bg)",
                  borderRadius: "8px",
                  border: "1px solid var(--border-color)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                  }}
                >
                  {weather.icon && (
                    <img
                      src={`https://openweathermap.org/img/wn/${weather.icon}@2x.png`}
                      alt={weather.description}
                      style={{
                        width: "28px",
                        height: "28px",
                      }}
                    />
                  )}
                  <div
                    style={{
                      fontSize: "16px",
                      fontWeight: 700,
                      color: "var(--text-primary)",
                    }}
                  >
                    {weather.temp}°C
                  </div>
                </div>
                {/* 날씨 설명 */}
                <div
                  style={{
                    fontSize: "13px",
                    color: "var(--text-secondary)",
                  }}
                >
                  {weather.description}
                </div>
                {/* 습도 정보 */}
                {weather.humidity !== undefined && (
                  <div
                    style={{
                      fontSize: "12px",
                      color: "var(--text-secondary)",
                      display: "flex",
                      alignItems: "center",
                      gap: "4px",
                    }}
                  >
                    <i className="fas fa-tint" style={{ fontSize: "11px" }}></i>
                    <span>습도 {weather.humidity}%</span>
                  </div>
                )}
                {/* 강수 경고 배지 */}
                {weather.hasPrecipitation && (
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "4px",
                      padding: "3px 6px",
                      background:
                        weather.precipitationType === "snow"
                          ? "rgba(59, 130, 246, 0.2)"
                          : "rgba(59, 130, 246, 0.2)",
                      borderRadius: "6px",
                      border:
                        weather.precipitationType === "snow"
                          ? "1px solid rgba(59, 130, 246, 0.5)"
                          : "1px solid rgba(59, 130, 246, 0.5)",
                    }}
                  >
                    <i
                      className={`fas ${
                        weather.precipitationType === "snow"
                          ? "fa-snowflake"
                          : "fa-cloud-rain"
                      }`}
                      style={{
                        fontSize: "11px",
                        color:
                          weather.precipitationType === "snow"
                            ? "rgba(59, 130, 246, 1)"
                            : "rgba(59, 130, 246, 1)",
                      }}
                    ></i>
                    <span
                      style={{
                        fontSize: "11px",
                        fontWeight: 600,
                        color:
                          weather.precipitationType === "snow"
                            ? "rgba(59, 130, 246, 1)"
                            : "rgba(59, 130, 246, 1)",
                      }}
                    >
                      {weather.precipitationType === "snow" ? "강설" : "강수"}{" "}
                      주의
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <div
                style={{
                  fontSize: "13px",
                  color: "var(--text-secondary)",
                  textAlign: "center",
                  padding: "8px",
                }}
              >
                날씨 정보 없음
              </div>
            )}

            {/* 현재 시간 표시 (년월일시분초) */}
            <div
              style={{
                padding: "12px",
                background: "var(--chip-container-bg)",
                borderRadius: "8px",
                border: "1px solid var(--border-color)",
                marginTop: "8px",
              }}
            >
              <div
                style={{
                  fontSize: "14px",
                  fontWeight: 600,
                  color: "var(--text-primary)",
                  textAlign: "center",
                  lineHeight: 1.4,
                }}
              >
                {currentTime.getFullYear()}년{" "}
                {String(currentTime.getMonth() + 1).padStart(2, "0")}월{" "}
                {String(currentTime.getDate()).padStart(2, "0")}일
              </div>
              <div
                style={{
                  fontSize: "16px",
                  fontWeight: 700,
                  color: "var(--accent-blue)",
                  textAlign: "center",
                  marginTop: "4px",
                  fontFamily: '"Chiron GoRound TC", sans-serif',
                }}
              >
                {String(currentTime.getHours()).padStart(2, "0")}:
                {String(currentTime.getMinutes()).padStart(2, "0")}:
                {String(currentTime.getSeconds()).padStart(2, "0")}
              </div>
            </div>
          </div>

          <nav className="sidebar-nav">
            {[
              { id: "dashboard", key: "dashboard", icon: "fas fa-chart-line" },
              { id: "cctv", key: "cctv", icon: "fas fa-video" },
              { id: "logs", key: "logs", icon: "fas fa-list-alt" },
              { id: "workers", key: "workers", icon: "fas fa-users" },
              { id: "calendar", key: "reports", icon: "fas fa-calendar-alt" },
            ].map((item) => (
              <a
                key={item.id}
                href="#"
                className={`sidebar-nav-item ${
                  activePage === item.id ? "active" : ""
                }`}
                onClick={(e) => {
                  e.preventDefault();
                  setActivePage(item.id);
                }}
                data-page={item.id}
              >
                <i className={item.icon}></i>
                <span>{t.nav[item.key]}</span>
              </a>
            ))}
          </nav>

          {/* 사이드바 하단 액션 버튼들 */}
          <div className="sidebar-actions">
            {/* 검색 버튼 */}
            <div
              className="sidebar-action-item"
              style={{ position: "relative" }}
            >
              <button
                className="sidebar-action-btn"
                title="검색"
                onMouseDown={(e) => {
                  e.stopPropagation();
                }}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  const newState = !showSearch;
                  setShowSearch(newState);
                  // 알림, 설정 드롭다운이 열려있으면 닫기
                  if (newState) {
                    if (showNotifications) {
                      setShowNotifications(false);
                    }
                    if (showSettings) {
                      setShowSettings(false);
                    }
                  }
                }}
              >
                <i className="fas fa-search"></i>
                <span>검색</span>
              </button>
            </div>

            {/* 알림 버튼 */}
            <div
              className="sidebar-action-item"
              style={{ position: "relative" }}
            >
              <button
                className="sidebar-action-btn"
                title="알림"
                onMouseDown={(e) => {
                  e.stopPropagation();
                }}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  const newState = !showNotifications;
                  if (newState === false) {
                    setPendingCount(0);
                  }
                  setShowNotifications(newState);
                  // 설정 드롭다운이 열려있으면 닫기
                  if (newState && showSettings) {
                    setShowSettings(false);
                  }
                }}
              >
                <i
                  className={`fas ${
                    notificationsEnabled ? "fa-bell" : "fa-bell-slash"
                  }`}
                ></i>
                <span>알림</span>
              </button>
              {showNotifications && (
                <div
                  className="sidebar-dropdown sidebar-notifications-dropdown"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div
                    style={{
                      padding: "12px 16px",
                      borderBottom: "1px solid var(--border-color)",
                      fontWeight: 600,
                    }}
                  >
                    {t.notifications.title} ({pendingCount}
                    {language === "ko" ? "건" : ""})
                  </div>
                  <div style={{ maxHeight: "300px", overflowY: "auto" }}>
                    {eventRows.filter((r) => r.status === "new").length > 0 ? (
                      eventRows
                        .filter((r) => r.status === "new")
                        .map((r, idx) => {
                          return (
                            <div
                              key={idx}
                              style={{
                                padding: "12px 16px",
                                borderBottom: "1px solid var(--border-color)",
                                cursor: "pointer",
                              }}
                              onMouseEnter={(e) =>
                                (e.currentTarget.style.background =
                                  "var(--chip-container-bg)")
                              }
                              onMouseLeave={(e) =>
                                (e.currentTarget.style.background =
                                  "transparent")
                              }
                            >
                              <div
                                style={{
                                  fontWeight: 600,
                                  fontSize: "13px",
                                  marginBottom: "4px",
                                }}
                              >
                                <span
                                  style={{
                                    display: "inline-block",
                                    padding: "4px 12px",
                                    borderRadius: "12px",
                                    fontSize: "13px",
                                    fontWeight: 600,
                                    backgroundColor:
                                      r.risk === "넘어짐 감지" ||
                                      r.risk === "Fall Detection"
                                        ? "rgba(239, 68, 68, 0.3)" // 빨간색 반투명
                                        : "rgba(245, 158, 11, 0.3)", // 노란색 반투명
                                    color: getRiskTypeColor(r.risk),
                                    border: `1px solid ${getRiskTypeColor(
                                      r.risk
                                    )}40`,
                                  }}
                                >
                                  {translateRiskType(r.risk, language, t)}
                                </span>
                              </div>
                              <div
                                style={{
                                  fontSize: "12px",
                                  color: "var(--text-secondary)",
                                }}
                              >
                                {r.worker}
                              </div>
                              <div
                                className="approvers"
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "8px",
                                  marginTop: "4px",
                                }}
                              >
                                <div
                                  style={{
                                    fontSize: "14px",
                                    fontWeight: 700,
                                    color: "#ffffff",
                                    width: "32px",
                                    height: "32px",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    borderRadius: "50%",
                                    background: getZoneColor(r.zone),
                                    flexShrink: 0,
                                  }}
                                >
                                  {getZoneLetter(r.zone)}
                                </div>
                                <span>{r.manager}</span>
                              </div>
                            </div>
                          );
                        })
                    ) : (
                      <div
                        style={{
                          padding: "24px",
                          textAlign: "center",
                          color: "var(--text-secondary)",
                          fontSize: "13px",
                        }}
                      >
                        {t.notifications.empty}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* 설정 버튼 */}
            <div
              className="sidebar-action-item"
              style={{ position: "relative" }}
            >
              <button
                className="sidebar-action-btn"
                title="설정"
                onMouseDown={(e) => {
                  e.stopPropagation();
                }}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  const newState = !showSettings;
                  setShowSettings(newState);
                  // 검색, 알림 드롭다운이 열려있으면 닫기
                  if (newState) {
                    if (showSearch) {
                      setShowSearch(false);
                    }
                    if (showNotifications) {
                      setShowNotifications(false);
                    }
                  }
                }}
              >
                <i className="fas fa-cog"></i>
                <span>설정</span>
              </button>
            </div>

            {/* 출입 관제 버튼 */}
            <button
              className="sidebar-action-btn sidebar-access-btn"
              onClick={() => {
                const newWindow = window.open(
                  `${window.location.origin}${window.location.pathname}?camera=true`,
                  "출입 카메라",
                  "width=1280,height=720,resizable=yes,scrollbars=no"
                );
                if (!newWindow) {
                  alert(
                    "팝업이 차단되었습니다. 브라우저 설정에서 팝업을 허용해주세요."
                  );
                }
              }}
            >
              <i
                className="fas fa-video"
                style={{
                  fontSize: "16px",
                  animation: "pulse 2s infinite",
                }}
              ></i>
              <span>출입 관제</span>
            </button>
          </div>
        </aside>

        {/* 메인 콘텐츠 영역 */}
        <main className="main-wrapper">
          <div className="page-content">
            {/* Dashboard */}
            {activePage === "dashboard" && (
              <div id="page-dashboard" className="page active">
                <WeatherTimeBar />
                <div className="dashboard-grid-top">
                  {/* Overview */}
                  <div className="card overview-card">
                    <div className="card-header">
                      <h2 className="card-title">{t.dashboard.siteSummary}</h2>
                    </div>
                    <div className="card-body">
                      <div className="overview-main">
                        <div className="total-value">
                          <span className="total-number" id="totalWorkers">
                            {totalWorkers}
                          </span>
                          <span>{language === "ko" ? "명" : ""}</span>
                          <span>
                            {language === "ko"
                              ? "(총 작업자)"
                              : "(${t.dashboard.totalWorkers})"}
                          </span>
                        </div>
                      </div>
                      <div className="overview-kpis">
                        <div className="kpi-item kpi-helmet">
                          <div className="kpi-value">{kpiHelmet}건</div>
                          <span className="kpi-label red">
                            <i
                              className="fas fa-hard-hat"
                              style={{ marginRight: "4px", fontSize: "10px" }}
                            ></i>
                            {t.dashboard.unhelmet}
                          </span>
                          <div className="kpi-trend-label">
                            {language === "ko"
                              ? "금일 기준 / 전일 대비"
                              : "Today / vs Yesterday"}
                          </div>
                          <div className="kpi-trend-value trend-up">▲1건</div>
                          <i className="fas fa-hard-hat kpi-bg-icon"></i>
                        </div>
                        <div className="kpi-item kpi-vest">
                          <div className="kpi-value">{kpiVest}건</div>
                          <span className="kpi-label orange">
                            <i
                              className="fas fa-vest"
                              style={{ marginRight: "4px", fontSize: "10px" }}
                            ></i>
                            {t.dashboard.unvest}
                          </span>
                          <div className="kpi-trend-label">
                            {language === "ko"
                              ? "금일 기준 / 전일 대비"
                              : "Today / vs Yesterday"}
                          </div>
                          <div className="kpi-trend-value trend-up">▲2건</div>
                          <i className="fas fa-vest kpi-bg-icon"></i>
                        </div>
                        <div className="kpi-item kpi-fall">
                          <div className="kpi-value">{kpiFall}건</div>
                          <span className="kpi-label yellow">
                            <i
                              className="fas fa-exclamation-triangle"
                              style={{ marginRight: "4px", fontSize: "10px" }}
                            ></i>
                            {t.dashboard.fall}
                          </span>
                          <div className="kpi-trend-label">
                            {language === "ko"
                              ? "금일 기준 / 전일 대비"
                              : "Today / vs Yesterday"}
                          </div>
                          <div className="kpi-trend-value trend-down">▼1건</div>
                          <i className="fas fa-exclamation-triangle kpi-bg-icon"></i>
                        </div>
                      </div>
                      <div className="overview-stats">
                        <div className="stat-item">
                          <div className="value">
                            {facialRecognitionAccuracy}%
                          </div>
                          <div className="label">
                            {t.dashboard.facialRecognition}
                          </div>
                        </div>
                        <div className="stat-item">
                          <div className="value">
                            {equipmentDetectionAccuracy}%
                          </div>
                          <div className="label">
                            {t.dashboard.equipmentDetection}
                          </div>
                        </div>
                        <div className="stat-item">
                          <div className="value">
                            {behaviorDetectionAccuracy}%
                          </div>
                          <div className="label">
                            {t.dashboard.behaviorDetection}
                          </div>
                        </div>
                        <div className="stat-item">
                          <div className="value">
                            {controlTeamCount}
                            {language === "ko" ? "명" : ""}
                          </div>
                          <div className="label">{t.dashboard.controlTeam}</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Logs (개선된 배지 버튼) */}
                  <div className="card log-card">
                    <div className="card-header">
                      <h2 className="card-title">{t.dashboard.riskLog}</h2>
                    </div>
                    <div className="card-body">
                      <div className="total-value">
                        <span className="total-number" id="totalAlerts">
                          {totalAlerts}
                        </span>{" "}
                        <span>
                          {language === "ko"
                            ? "건 (금일 총 알림)"
                            : `cases (Total alerts today)`}
                        </span>
                      </div>
                      <div className="log-table-wrapper">
                        <table className="log-table">
                          <thead>
                            <tr>
                              <th>{t.dashboard.worker}</th>
                              <th>{t.dashboard.riskType}</th>
                              <th>{t.dashboard.manager}</th>
                              <th style={{ textAlign: "right" }}>
                                {t.dashboard.action}
                              </th>
                            </tr>
                          </thead>
                          <tbody id="eventLogBody">
                            {eventRows.map((r, idx) => {
                              // 위험 유형별 색상 결정
                              const isFall =
                                r.risk === "넘어짐" ||
                                r.risk === "넘어짐 감지" ||
                                r.risk === "Fall Detection" ||
                                r.risk?.toLowerCase().includes("fall");

                              const isPPE =
                                r.risk === "안전모 미착용" ||
                                r.risk === "안전조끼 미착용" ||
                                r.risk === "마스크 미착용" ||
                                r.risk === "Unworn Safety Helmet" ||
                                r.risk === "Unworn Safety Vest" ||
                                r.risk === "Unworn Mask" ||
                                r.risk?.includes("안전모") ||
                                r.risk?.includes("안전조끼") ||
                                r.risk?.includes("마스크") ||
                                r.risk?.toLowerCase().includes("helmet") ||
                                r.risk?.toLowerCase().includes("vest") ||
                                r.risk?.toLowerCase().includes("mask");

                              return (
                                <tr
                                  key={idx}
                                  style={{
                                    backgroundColor: isFall
                                      ? "rgba(239, 68, 68, 0.1)" // 넘어짐: 반투명 빨간색
                                      : isPPE
                                      ? "rgba(245, 158, 11, 0.1)" // PPE 미착용: 반투명 노란색
                                      : "transparent", // 나머지: 투명
                                  }}
                                >
                                  <td className="worker">{r.worker}</td>
                                  <td>
                                    <span
                                      className={`kpi-label ${
                                        isFall
                                          ? "red"
                                          : isPPE
                                          ? "orange"
                                          : "yellow"
                                      }`}
                                      style={{
                                        fontSize: "11px",
                                        fontWeight: 600,
                                        padding: "5px 10px",
                                        borderRadius: "10px",
                                        display: "inline-flex",
                                        alignItems: "center",
                                        gap: "4px",
                                      }}
                                    >
                                      {isFall ? (
                                        <i
                                          className="fas fa-exclamation-triangle"
                                          style={{ fontSize: "10px" }}
                                        />
                                      ) : (
                                        <i
                                          className="fas fa-hard-hat"
                                          style={{ fontSize: "10px" }}
                                        />
                                      )}
                                      {translateRiskType(r.risk, language, t)}
                                    </span>
                                  </td>
                                  <td className="approvers">
                                    <div
                                      style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: "8px",
                                      }}
                                    >
                                      <div
                                        style={{
                                          fontSize: "13px",
                                          fontWeight: 700,
                                          color: "#ffffff",
                                          width: "28px",
                                          height: "28px",
                                          display: "flex",
                                          alignItems: "center",
                                          justifyContent: "center",
                                          borderRadius: "50%",
                                          background: getZoneColor(r.zone),
                                          flexShrink: 0,
                                        }}
                                      >
                                        {getZoneLetter(r.zone)}
                                      </div>
                                      <span style={{ fontSize: "12px" }}>
                                        {r.manager}
                                      </span>
                                    </div>
                                  </td>
                                  <td className="status">
                                    <button
                                      className={
                                        r.status === "done"
                                          ? "badge-btn complete"
                                          : "badge-btn new"
                                      }
                                      disabled={r.status === "done"}
                                      style={{
                                        display: "inline-flex",
                                        alignItems: "center",
                                        gap: "6px",
                                      }}
                                      onClick={() => {
                                        // 이미 완료된 경우 처리하지 않음
                                        if (r.status === "done") {
                                          return;
                                        }

                                        // 구역 정보 추출 (r.zone에서 직접 가져오기)
                                        const zoneLetter = r.zone
                                          ? r.zone.charAt(0).toUpperCase()
                                          : "A";

                                        // CCTV 화면으로 이동
                                        setActivePage("cctv");

                                        // 페이지 전환이 완료된 후 팝업 표시 (약간의 딜레이)
                                        setTimeout(() => {
                                          // zone에 따라 카메라 설정
                                          if (zoneLetter === "A") {
                                            setFocusCam({
                                              type: "webcam",
                                              title: "LIVE: A",
                                            });
                                            // 정보 팝업은 전체화면 팝업과 동시에 표시
                                            setTimeout(() => {
                                              setShowBottomRightPopup({
                                                zone: "A",
                                                title: "LIVE: A",
                                                manager: r.manager,
                                                eventData: r, // 이벤트 정보 저장
                                              });
                                            }, 50);
                                          } else if (zoneLetter === "B") {
                                            setFocusCam({
                                              id: "nAE3sui8G6A",
                                              title: "LIVE: B",
                                            });
                                            // 정보 팝업은 전체화면 팝업과 동시에 표시
                                            setTimeout(() => {
                                              setShowBottomRightPopup({
                                                zone: "B",
                                                title: "LIVE: B",
                                                manager: r.manager,
                                                eventData: r, // 이벤트 정보 저장
                                              });
                                            }, 50);
                                          } else if (zoneLetter === "C") {
                                            setFocusCam({
                                              type: "webcam3",
                                              title: "LIVE: C",
                                            });
                                            // 정보 팝업은 전체화면 팝업과 동시에 표시
                                            setTimeout(() => {
                                              setShowBottomRightPopup({
                                                zone: "C",
                                                title: "LIVE: C",
                                                manager: r.manager,
                                                eventData: r, // 이벤트 정보 저장
                                              });
                                            }, 50);
                                          } else if (zoneLetter === "D") {
                                            setFocusCam({
                                              type: "webcam4",
                                              title: "LIVE: D",
                                            });
                                            // 정보 팝업은 전체화면 팝업과 동시에 표시
                                            setTimeout(() => {
                                              setShowBottomRightPopup({
                                                zone: "D",
                                                title: "LIVE: D",
                                                manager: r.manager,
                                                eventData: r, // 이벤트 정보 저장
                                              });
                                            }, 50);
                                          } else {
                                            // 알 수 없는 구역이면 기본값으로 "A" 사용
                                            console.warn(
                                              `[대시보드] 알 수 없는 구역: ${zoneLetter}, 기본값 'A' 사용`
                                            );
                                            setFocusCam({
                                              type: "webcam",
                                              title: "LIVE: A",
                                            });
                                            setTimeout(() => {
                                              setShowBottomRightPopup({
                                                zone: "A",
                                                title: "LIVE: A",
                                                manager: r.manager,
                                                eventData: r, // 이벤트 정보 저장
                                              });
                                            }, 50);
                                          }
                                        }, 100); // 100ms 딜레이로 페이지 전환 후 팝업 표시
                                      }}
                                    >
                                      {r.status === "done" ? (
                                        "확인 완료"
                                      ) : (
                                        <>
                                          <span
                                            style={{
                                              width: "6px",
                                              height: "6px",
                                              borderRadius: "50%",
                                              backgroundColor: "#ffffff",
                                              display: "inline-block",
                                            }}
                                          />
                                          확인 필요
                                        </>
                                      )}
                                    </button>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="dashboard-grid-bottom">
                  {/* Gauge */}
                  <div className="card gauge-card">
                    <div className="card-header">
                      <h2 className="card-title">{t.dashboard.safetyScore}</h2>
                    </div>
                    <div className="card-body">
                      <div className="gauge-chart-container">
                        <canvas ref={gaugeRef} id="safetyGaugeChart" />
                        <div className="gauge-center-text">
                          <div className="value">
                            {safetyScore}
                            {language === "ko" ? "점" : ""}
                          </div>
                          <div className="label">{t.dashboard.safetyIndex}</div>
                        </div>
                      </div>
                      <div className="gauge-stats">
                        <div className="stat-item">
                          <div className="value">{completedActions}</div>
                          <div className="label green">
                            {t.dashboard.completedAction}
                          </div>
                        </div>
                        <div className="stat-item">
                          <div className="value">{pendingCount}</div>
                          <div className="label red">
                            {t.dashboard.needsCheck}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Bar */}
                  <div className="card chart-card">
                    <div className="card-header">
                      <h2 className="card-title">{t.dashboard.weeklyStats}</h2>
                    </div>
                    <div className="card-body">
                      <div className="chart-kpis">
                        <div className="kpi-item">
                          <div className="value">
                            {weeklyTotalDetections.toLocaleString()}
                          </div>
                          <div className="label">
                            {t.dashboard.totalDetections}
                          </div>
                        </div>
                        <div className="kpi-item">
                          <div className="value">{weeklyHelmetViolations}</div>
                          <div className="label">
                            {t.dashboard.helmetViolation}{" "}
                            <i className="fas fa-arrow-up"></i>
                          </div>
                        </div>
                        <div className="kpi-item">
                          <div className="value">{weeklyFallDetections}</div>
                          <div className="label">
                            {t.dashboard.fallDetection}{" "}
                            <i className="fas fa-arrow-up green"></i>
                          </div>
                        </div>
                      </div>
                      <div className="bar-chart-container">
                        <canvas ref={barRef} id="weeklyBarChart" />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* CCTV */}
            {activePage === "cctv" && (
              <div id="page-cctv" className="page active">
                <WeatherTimeBar />
                <div className="cctv-grid">
                  {/* 웹캠 (첫 번째 카메라) */}
                  <div
                    className={`card cctv-feed ${
                      activeAlertZones.includes("A") ? "alert-zone-active" : ""
                    }`}
                  >
                    <div
                      className="cctv-header"
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <span>
                        <span className="live-dot"></span> LIVE: A
                      </span>
                    </div>
                    <div
                      className="cctv-video"
                      onClick={() => {
                        setFocusCam({ type: "webcam", title: "LIVE: A" });
                      }}
                    >
                      <img
                        src={`/api/stream?cam_id=0&processed=false&t=${cameraRefreshKey}`}
                        alt="Camera 0"
                        crossOrigin="anonymous"
                        style={{
                          position: "absolute",
                          top: 0,
                          left: 0,
                          width: "100%",
                          height: "100%",
                          objectFit: "cover",
                        }}
                        onError={(e) => {
                          console.error("Camera 0 stream error:", e);
                          const img = e.target;
                          const isSslError =
                            img.src &&
                            img.src.includes("https://") &&
                            (img.naturalWidth === 0 || img.naturalHeight === 0);

                          if (isSslError) {
                            setStreamErrors((prev) => ({
                              ...prev,
                              0: {
                                hasError: true,
                                retryCount: prev[0].retryCount + 1,
                                isSslError: true,
                              },
                            }));
                          } else {
                            if (streamRetryTimers.current[0]) {
                              clearTimeout(streamRetryTimers.current[0]);
                            }
                            streamRetryTimers.current[0] = setTimeout(() => {
                              setCameraRefreshKey((prev) => prev + 1);
                            }, 2000);
                          }
                        }}
                        onLoad={() => {
                          if (streamErrors[0].hasError) {
                            setStreamErrors((prev) => ({
                              ...prev,
                              0: {
                                hasError: false,
                                retryCount: 0,
                                isSslError: false,
                              },
                            }));
                          }
                        }}
                      />
                      {streamErrors[0].hasError &&
                        streamErrors[0].isSslError && (
                          <div
                            style={{
                              position: "absolute",
                              top: "50%",
                              left: "50%",
                              transform: "translate(-50%, -50%)",
                              color: "#FFD700",
                              textAlign: "center",
                              backgroundColor: "rgba(0, 0, 0, 0.8)",
                              padding: "20px",
                              borderRadius: "8px",
                              zIndex: 10,
                              maxWidth: "400px",
                            }}
                          >
                            <i
                              className="fas fa-exclamation-triangle"
                              style={{ fontSize: "32px", marginBottom: "10px" }}
                            ></i>
                            <div
                              style={{
                                fontSize: "14px",
                                fontWeight: "bold",
                                marginBottom: "8px",
                              }}
                            >
                              SSL 인증서 승인 필요
                            </div>
                            <div
                              style={{
                                fontSize: "12px",
                                opacity: 0.9,
                                marginBottom: "12px",
                              }}
                            >
                              브라우저에서 인증서를 승인해주세요
                            </div>
                            <a
                              href="https://localhost:8081"
                              target="_blank"
                              rel="noopener noreferrer"
                              style={{
                                display: "inline-block",
                                padding: "8px 16px",
                                background: "var(--accent-blue)",
                                color: "white",
                                textDecoration: "none",
                                borderRadius: "4px",
                                fontSize: "12px",
                                marginTop: "8px",
                              }}
                              onClick={(e) => {
                                e.stopPropagation();
                                window.open("https://localhost:8081", "_blank");
                              }}
                            >
                              인증서 승인하러 가기
                            </a>
                            <div
                              style={{
                                fontSize: "11px",
                                opacity: 0.7,
                                marginTop: "12px",
                              }}
                            >
                              팁: 새 탭에서 열린 페이지에서 "고급" → "안전하지
                              않음(권장되지 않음)" 클릭
                            </div>
                          </div>
                        )}
                    </div>
                  </div>

                  {/* 두 번째 카메라 */}
                  <div
                    className={`card cctv-feed ${
                      activeAlertZones.includes("B") ? "alert-zone-active" : ""
                    }`}
                  >
                    <div
                      className="cctv-header"
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <span>
                        <span className="live-dot"></span> LIVE: B
                      </span>
                    </div>
                    <div
                      className="cctv-video"
                      onClick={() => {
                        setFocusCam({ type: "webcam2", title: "LIVE: B" });
                      }}
                    >
                      <img
                        src={`/api/stream?cam_id=1&processed=false&t=${cameraRefreshKey}`}
                        alt="Camera 1"
                        crossOrigin="anonymous"
                        style={{
                          position: "absolute",
                          top: 0,
                          left: 0,
                          width: "100%",
                          height: "100%",
                          objectFit: "cover",
                        }}
                        onError={(e) => {
                          console.error("Camera 1 stream error:", e);
                          const img = e.target;
                          const isSslError =
                            img.src &&
                            img.src.includes("https://") &&
                            (img.naturalWidth === 0 || img.naturalHeight === 0);

                          if (isSslError) {
                            setStreamErrors((prev) => ({
                              ...prev,
                              1: {
                                hasError: true,
                                retryCount: prev[1].retryCount + 1,
                                isSslError: true,
                              },
                            }));
                          } else {
                            if (streamRetryTimers.current[1]) {
                              clearTimeout(streamRetryTimers.current[1]);
                            }
                            streamRetryTimers.current[1] = setTimeout(() => {
                              setCameraRefreshKey((prev) => prev + 1);
                            }, 2000);
                          }
                        }}
                        onLoad={() => {
                          if (streamErrors[1].hasError) {
                            setStreamErrors((prev) => ({
                              ...prev,
                              1: {
                                hasError: false,
                                retryCount: 0,
                                isSslError: false,
                              },
                            }));
                          }
                        }}
                      />
                      {streamErrors[1].hasError &&
                        streamErrors[1].isSslError && (
                          <div
                            style={{
                              position: "absolute",
                              top: "50%",
                              left: "50%",
                              transform: "translate(-50%, -50%)",
                              color: "#FFD700",
                              textAlign: "center",
                              backgroundColor: "rgba(0, 0, 0, 0.8)",
                              padding: "20px",
                              borderRadius: "8px",
                              zIndex: 10,
                              maxWidth: "400px",
                            }}
                          >
                            <i
                              className="fas fa-exclamation-triangle"
                              style={{ fontSize: "32px", marginBottom: "10px" }}
                            ></i>
                            <div
                              style={{
                                fontSize: "14px",
                                fontWeight: "bold",
                                marginBottom: "8px",
                              }}
                            >
                              SSL 인증서 승인 필요
                            </div>
                            <div
                              style={{
                                fontSize: "12px",
                                opacity: 0.9,
                                marginBottom: "12px",
                              }}
                            >
                              브라우저에서 인증서를 승인해주세요
                            </div>
                            <a
                              href="https://localhost:8081"
                              target="_blank"
                              rel="noopener noreferrer"
                              style={{
                                display: "inline-block",
                                padding: "8px 16px",
                                background: "var(--accent-blue)",
                                color: "white",
                                textDecoration: "none",
                                borderRadius: "4px",
                                fontSize: "12px",
                                marginTop: "8px",
                              }}
                              onClick={(e) => {
                                e.stopPropagation();
                                window.open("https://localhost:8081", "_blank");
                              }}
                            >
                              인증서 승인하러 가기
                            </a>
                            <div
                              style={{
                                fontSize: "11px",
                                opacity: 0.7,
                                marginTop: "12px",
                              }}
                            >
                              팁: 새 탭에서 열린 페이지에서 "고급" → "안전하지
                              않음(권장되지 않음)" 클릭
                            </div>
                          </div>
                        )}
                    </div>
                  </div>

                  {/* 세 번째 카메라 (LIVE: C) */}
                  <div
                    className={`card cctv-feed ${
                      activeAlertZones.includes("C") ? "alert-zone-active" : ""
                    }`}
                  >
                    <div
                      className="cctv-header"
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <span>
                        <span className="live-dot"></span> LIVE: C
                      </span>
                    </div>
                    <div
                      className="cctv-video"
                      onClick={() => {
                        setFocusCam({ type: "webcam3", title: "LIVE: C" });
                      }}
                    >
                      <img
                        src={`/api/stream?cam_id=2&processed=false&t=${cameraRefreshKey}`}
                        alt="Camera 2"
                        crossOrigin="anonymous"
                        style={{
                          position: "absolute",
                          top: 0,
                          left: 0,
                          width: "100%",
                          height: "100%",
                          objectFit: "cover",
                        }}
                        onError={(e) => {
                          console.error("Camera 2 stream error:", e);
                          const img = e.target;
                          const isSslError =
                            img.src &&
                            img.src.includes("https://") &&
                            (img.naturalWidth === 0 || img.naturalHeight === 0);

                          if (isSslError) {
                            setStreamErrors((prev) => ({
                              ...prev,
                              2: {
                                hasError: true,
                                retryCount: (prev[2]?.retryCount || 0) + 1,
                                isSslError: true,
                              },
                            }));
                          } else {
                            if (streamRetryTimers.current[2]) {
                              clearTimeout(streamRetryTimers.current[2]);
                            }
                            streamRetryTimers.current[2] = setTimeout(() => {
                              setCameraRefreshKey((prev) => prev + 1);
                            }, 2000);
                          }
                        }}
                        onLoad={() => {
                          if (streamErrors[2]?.hasError) {
                            setStreamErrors((prev) => ({
                              ...prev,
                              2: {
                                hasError: false,
                                retryCount: 0,
                                isSslError: false,
                              },
                            }));
                          }
                        }}
                      />
                      {streamErrors[2]?.hasError &&
                        streamErrors[2]?.isSslError && (
                          <div
                            style={{
                              position: "absolute",
                              top: "50%",
                              left: "50%",
                              transform: "translate(-50%, -50%)",
                              color: "#FFD700",
                              textAlign: "center",
                              backgroundColor: "rgba(0, 0, 0, 0.8)",
                              padding: "20px",
                              borderRadius: "8px",
                              zIndex: 10,
                              maxWidth: "400px",
                            }}
                          >
                            <i
                              className="fas fa-exclamation-triangle"
                              style={{ fontSize: "32px", marginBottom: "10px" }}
                            ></i>
                            <div
                              style={{
                                fontSize: "14px",
                                fontWeight: "bold",
                                marginBottom: "8px",
                              }}
                            >
                              SSL 인증서 승인 필요
                            </div>
                            <div
                              style={{
                                fontSize: "12px",
                                opacity: 0.9,
                                marginBottom: "12px",
                              }}
                            >
                              브라우저에서 인증서를 승인해주세요
                            </div>
                            <a
                              href="https://localhost:8081"
                              target="_blank"
                              rel="noopener noreferrer"
                              style={{
                                display: "inline-block",
                                padding: "8px 16px",
                                background: "var(--accent-blue)",
                                color: "white",
                                textDecoration: "none",
                                borderRadius: "4px",
                                fontSize: "12px",
                                marginTop: "8px",
                              }}
                              onClick={(e) => {
                                e.stopPropagation();
                                window.open("https://localhost:8081", "_blank");
                              }}
                            >
                              인증서 승인하러 가기
                            </a>
                            <div
                              style={{
                                fontSize: "11px",
                                opacity: 0.7,
                                marginTop: "12px",
                              }}
                            >
                              팁: 새 탭에서 열린 페이지에서 "고급" → "안전하지
                              않음(권장되지 않음)" 클릭
                            </div>
                          </div>
                        )}
                    </div>
                  </div>

                  {/* 네 번째 카메라 (LIVE: D) */}
                  <div
                    className={`card cctv-feed ${
                      activeAlertZones.includes("D") ? "alert-zone-active" : ""
                    }`}
                  >
                    <div
                      className="cctv-header"
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <span>
                        <span className="live-dot"></span> LIVE: D
                      </span>
                    </div>
                    <div
                      className="cctv-video"
                      onClick={() => {
                        setFocusCam({ type: "webcam4", title: "LIVE: D" });
                      }}
                    >
                      <img
                        src={`/api/stream?cam_id=3&processed=false&t=${cameraRefreshKey}`}
                        alt="Camera 3"
                        crossOrigin="anonymous"
                        style={{
                          position: "absolute",
                          top: 0,
                          left: 0,
                          width: "100%",
                          height: "100%",
                          objectFit: "cover",
                        }}
                        onError={(e) => {
                          console.error("Camera 3 stream error:", e);
                          const img = e.target;
                          const isSslError =
                            img.src &&
                            img.src.includes("https://") &&
                            (img.naturalWidth === 0 || img.naturalHeight === 0);

                          if (isSslError) {
                            setStreamErrors((prev) => ({
                              ...prev,
                              3: {
                                hasError: true,
                                retryCount: (prev[3]?.retryCount || 0) + 1,
                                isSslError: true,
                              },
                            }));
                          } else {
                            if (streamRetryTimers.current[3]) {
                              clearTimeout(streamRetryTimers.current[3]);
                            }
                            streamRetryTimers.current[3] = setTimeout(() => {
                              setCameraRefreshKey((prev) => prev + 1);
                            }, 2000);
                          }
                        }}
                        onLoad={() => {
                          if (streamErrors[3]?.hasError) {
                            setStreamErrors((prev) => ({
                              ...prev,
                              3: {
                                hasError: false,
                                retryCount: 0,
                                isSslError: false,
                              },
                            }));
                          }
                        }}
                      />
                      {streamErrors[3]?.hasError &&
                        streamErrors[3]?.isSslError && (
                          <div
                            style={{
                              position: "absolute",
                              top: "50%",
                              left: "50%",
                              transform: "translate(-50%, -50%)",
                              color: "#FFD700",
                              textAlign: "center",
                              backgroundColor: "rgba(0, 0, 0, 0.8)",
                              padding: "20px",
                              borderRadius: "8px",
                              zIndex: 10,
                              maxWidth: "400px",
                            }}
                          >
                            <i
                              className="fas fa-exclamation-triangle"
                              style={{ fontSize: "32px", marginBottom: "10px" }}
                            ></i>
                            <div
                              style={{
                                fontSize: "14px",
                                fontWeight: "bold",
                                marginBottom: "8px",
                              }}
                            >
                              SSL 인증서 승인 필요
                            </div>
                            <div
                              style={{
                                fontSize: "12px",
                                opacity: 0.9,
                                marginBottom: "12px",
                              }}
                            >
                              브라우저에서 인증서를 승인해주세요
                            </div>
                            <a
                              href="https://localhost:8081"
                              target="_blank"
                              rel="noopener noreferrer"
                              style={{
                                display: "inline-block",
                                padding: "8px 16px",
                                background: "var(--accent-blue)",
                                color: "white",
                                textDecoration: "none",
                                borderRadius: "4px",
                                fontSize: "12px",
                                marginTop: "8px",
                              }}
                              onClick={(e) => {
                                e.stopPropagation();
                                window.open("https://localhost:8081", "_blank");
                              }}
                            >
                              인증서 승인하러 가기
                            </a>
                            <div
                              style={{
                                fontSize: "11px",
                                opacity: 0.7,
                                marginTop: "12px",
                              }}
                            >
                              팁: 새 탭에서 열린 페이지에서 "고급" → "안전하지
                              않음(권장되지 않음)" 클릭
                            </div>
                          </div>
                        )}
                    </div>
                  </div>
                </div>

                {focusCam && (
                  <div
                    className="fs-overlay"
                    onClick={() => {
                      setFocusCam(null);
                      setShowBottomRightPopup(null);
                    }}
                  >
                    <div
                      className="fs-player"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {focusCam.type === "webcam" ? (
                        <div
                          style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            backgroundColor: "#000",
                          }}
                        >
                          <img
                            ref={focusImageRef}
                            src={`/api/stream?cam_id=0&processed=true&t=${cameraRefreshKey}`}
                            alt="Camera 0"
                            crossOrigin="anonymous"
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              objectFit: "contain",
                              display: "block",
                            }}
                            onError={(e) => {
                              console.error("Camera 0 stream error:", e);
                              if (streamRetryTimers.current[0]) {
                                clearTimeout(streamRetryTimers.current[0]);
                              }
                              streamRetryTimers.current[0] = setTimeout(() => {
                                setCameraRefreshKey((prev) => prev + 1);
                              }, 2000);
                            }}
                            onLoad={() => {
                              if (streamErrors[0].hasError) {
                                setStreamErrors((prev) => ({
                                  ...prev,
                                  0: {
                                    hasError: false,
                                    retryCount: 0,
                                    isSslError: false,
                                  },
                                }));
                              }
                            }}
                          />
                          {/* Canvas 바운딩 박스 비활성화: 백엔드에서 이미 그려진 프레임 사용 */}
                          <canvas
                            ref={focusCanvasRef}
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              pointerEvents: "none",
                              display: "none", // 숨김 처리
                            }}
                          />
                        </div>
                      ) : focusCam.type === "webcam2" ? (
                        <div
                          style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            backgroundColor: "#000",
                          }}
                        >
                          <img
                            ref={focusImageRef2}
                            src={`/api/stream?cam_id=1&processed=true&t=${cameraRefreshKey}`}
                            alt="Camera 1"
                            crossOrigin="anonymous"
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              objectFit: "contain",
                              display: "block",
                            }}
                            onError={(e) => {
                              console.error("Camera 1 stream error:", e);
                              if (streamRetryTimers.current[1]) {
                                clearTimeout(streamRetryTimers.current[1]);
                              }
                              streamRetryTimers.current[1] = setTimeout(() => {
                                setCameraRefreshKey((prev) => prev + 1);
                              }, 2000);
                            }}
                            onLoad={() => {
                              if (streamErrors[1].hasError) {
                                setStreamErrors((prev) => ({
                                  ...prev,
                                  1: {
                                    hasError: false,
                                    retryCount: 0,
                                    isSslError: false,
                                  },
                                }));
                              }
                            }}
                          />
                          {/* Canvas 바운딩 박스 비활성화: 백엔드에서 이미 그려진 프레임 사용 */}
                          <canvas
                            ref={focusCanvasRef2}
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              pointerEvents: "none",
                              display: "none", // 숨김 처리
                            }}
                          />
                        </div>
                      ) : focusCam.type === "webcam3" ? (
                        <div
                          style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            backgroundColor: "#000",
                          }}
                        >
                          <img
                            ref={focusImageRef2}
                            src={`/api/stream?cam_id=2&processed=true&t=${cameraRefreshKey}`}
                            alt="Camera 2"
                            crossOrigin="anonymous"
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              objectFit: "contain",
                              display: "block",
                            }}
                            onError={(e) => {
                              console.error("Camera 2 stream error:", e);
                              if (streamRetryTimers.current[2]) {
                                clearTimeout(streamRetryTimers.current[2]);
                              }
                              streamRetryTimers.current[2] = setTimeout(() => {
                                setCameraRefreshKey((prev) => prev + 1);
                              }, 2000);
                            }}
                            onLoad={() => {
                              if (streamErrors[2]?.hasError) {
                                setStreamErrors((prev) => ({
                                  ...prev,
                                  2: {
                                    hasError: false,
                                    retryCount: 0,
                                    isSslError: false,
                                  },
                                }));
                              }
                            }}
                          />
                          {/* Canvas 바운딩 박스 비활성화: 백엔드에서 이미 그려진 프레임 사용 */}
                          <canvas
                            ref={focusCanvasRef2}
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              pointerEvents: "none",
                              display: "none", // 숨김 처리
                            }}
                          />
                        </div>
                      ) : focusCam.type === "webcam4" ? (
                        <div
                          style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            backgroundColor: "#000",
                          }}
                        >
                          <img
                            ref={focusImageRef2}
                            src={`/api/stream?cam_id=3&processed=true&t=${cameraRefreshKey}`}
                            alt="Camera 3"
                            crossOrigin="anonymous"
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              objectFit: "contain",
                              display: "block",
                            }}
                            onError={(e) => {
                              console.error("Camera 3 stream error:", e);
                              if (streamRetryTimers.current[3]) {
                                clearTimeout(streamRetryTimers.current[3]);
                              }
                              streamRetryTimers.current[3] = setTimeout(() => {
                                setCameraRefreshKey((prev) => prev + 1);
                              }, 2000);
                            }}
                            onLoad={() => {
                              if (streamErrors[3]?.hasError) {
                                setStreamErrors((prev) => ({
                                  ...prev,
                                  3: {
                                    hasError: false,
                                    retryCount: 0,
                                    isSslError: false,
                                  },
                                }));
                              }
                            }}
                          />
                          {/* Canvas 바운딩 박스 비활성화: 백엔드에서 이미 그려진 프레임 사용 */}
                          <canvas
                            ref={focusCanvasRef2}
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              pointerEvents: "none",
                              display: "none", // 숨김 처리
                            }}
                          />
                        </div>
                      ) : (
                        <iframe
                          src={`https://www.youtube.com/embed/${focusCam.id}?autoplay=1&mute=1&controls=1`}
                          title={focusCam.title}
                          frameBorder="0"
                          allow="autoplay; encrypted-media"
                          allowFullScreen
                        />
                      )}
                    </div>
                  </div>
                )}

                {/* 오른쪽 하단 팝업 */}
                {showBottomRightPopup && (
                  <div
                    style={{
                      position: "fixed",
                      bottom: "20px",
                      right: "20px",
                      width: "300px",
                      background: "var(--card-bg)",
                      borderRadius: "12px",
                      padding: "20px",
                      boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)",
                      zIndex: 1001,
                      border: "1px solid var(--border-color)",
                      animation: "slideUpFromBottom 0.3s ease-out",
                    }}
                  >
                    <h3
                      style={{
                        margin: 0,
                        marginBottom: "12px",
                        fontSize: "16px",
                        fontWeight: 600,
                        color: "var(--text-primary)",
                      }}
                    >
                      {showBottomRightPopup.title}
                    </h3>
                    <div
                      style={{
                        fontSize: "14px",
                        color: "var(--text-secondary)",
                        marginBottom: "8px",
                      }}
                    >
                      구역: {showBottomRightPopup.zone}
                    </div>
                    {showBottomRightPopup.manager && (
                      <div
                        style={{
                          fontSize: "14px",
                          color: "var(--text-secondary)",
                          marginBottom: "16px",
                        }}
                      >
                        관리자: {showBottomRightPopup.manager}
                      </div>
                    )}
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: "8px",
                        marginTop: "16px",
                      }}
                    >
                      <button
                        onClick={async () => {
                          // 이벤트 정보 확인
                          const eventData = showBottomRightPopup?.eventData;
                          if (!eventData) {
                            showToast(
                              "이벤트 정보를 찾을 수 없습니다",
                              "error"
                            );
                            return;
                          }

                          // 원본 데이터 사용 (API 호출 시 DB에 저장된 형식 그대로 사용)
                          const originalData = eventData._original || {};
                          const workerId = originalData.worker_id
                            ? String(originalData.worker_id).trim()
                            : eventData.worker_id
                            ? String(eventData.worker_id).trim()
                            : "";
                          const violationDatetime =
                            originalData.violation_datetime
                              ? String(originalData.violation_datetime).trim()
                              : eventData.violation_datetime
                              ? String(eventData.violation_datetime).trim()
                              : "";

                          if (!workerId || !violationDatetime) {
                            showToast(
                              "상태 업데이트에 필요한 정보가 없습니다",
                              "error"
                            );
                            return;
                          }

                          try {
                            // DB 상태 업데이트
                            const response = await api.updateViolationStatus(
                              workerId,
                              violationDatetime,
                              "done"
                            );

                            // API 응답 확인
                            if (!response || !response.success) {
                              const errorMessage =
                                response?.error ||
                                response?.message ||
                                "상태 업데이트에 실패했습니다.";
                              throw new Error(errorMessage);
                            }

                            // 해당 항목이 KPI 항목인지 확인 (안전모, 안전조끼, 넘어짐만 카운트)
                            const riskType =
                              eventData.risk || eventData.type || "";
                            const riskTypeLower = riskType.toLowerCase();
                            const isKpiItem =
                              riskType.includes("안전모") ||
                              riskType.includes("헬멧") ||
                              riskTypeLower.includes("helmet") ||
                              riskTypeLower.includes("hardhat") ||
                              riskType.includes("안전조끼") ||
                              riskType.includes("조끼") ||
                              riskTypeLower.includes("vest") ||
                              riskTypeLower.includes("reflective") ||
                              riskType.includes("넘어짐") ||
                              riskType.includes("낙상") ||
                              riskTypeLower.includes("fall");

                            // KPI 항목인 경우에만 completedActions와 safetyScore 즉시 업데이트
                            if (isKpiItem) {
                              setCompletedActions((prev) => {
                                const newCompleted = prev + 1;
                                // safetyScore 재계산
                                setSafetyScore(() => {
                                  const total = totalAlerts;
                                  return total > 0
                                    ? Math.round((newCompleted / total) * 100)
                                    : 100;
                                });
                                return newCompleted;
                              });
                            }

                            // eventRows에서 해당 항목 제거 (done 상태는 표시하지 않음)
                            setEventRows((prevRows) => {
                              const filtered = prevRows.filter(
                                (row) =>
                                  !(
                                    row.worker_id === eventData.worker_id &&
                                    row.violation_datetime ===
                                      eventData.violation_datetime
                                  )
                              );
                              // pendingCount 업데이트
                              setPendingCount((prev) => Math.max(0, prev - 1));
                              return filtered;
                            });

                            // 확인 요청 메시지 전송 알림
                            showToast(
                              "확인 요청 메시지가 전송되었습니다",
                              "success"
                            );

                            // 대시보드로 복귀
                            setTimeout(() => {
                              setActivePage("dashboard");
                              setShowBottomRightPopup(null);
                            }, 500); // toast가 표시된 후 대시보드로 복귀
                          } catch (error) {
                            console.error("[상태 업데이트] 오류:", error);
                            const errorMessage =
                              error.message ||
                              error.data?.error ||
                              error.data?.message ||
                              "상태 업데이트에 실패했습니다. 다시 시도해주세요.";
                            showToast(errorMessage, "error");
                          }
                        }}
                        style={{
                          width: "100%",
                          padding: "10px 16px",
                          background: "var(--accent-blue)",
                          color: "white",
                          border: "none",
                          borderRadius: "8px",
                          cursor: "pointer",
                          fontSize: "14px",
                          fontWeight: 600,
                          transition: "opacity 0.2s",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.opacity = "0.9";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.opacity = "1";
                        }}
                      >
                        확인 요청
                      </button>
                      <button
                        onClick={() => {
                          // 119 호출 기능 (현재는 활성화만)
                          showToast("119 호출 기능은 준비 중입니다", "info");
                        }}
                        style={{
                          width: "100%",
                          padding: "10px 16px",
                          background: "var(--accent-red)",
                          color: "white",
                          border: "none",
                          borderRadius: "8px",
                          cursor: "pointer",
                          fontSize: "14px",
                          fontWeight: 600,
                          transition: "opacity 0.2s",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.opacity = "0.9";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.opacity = "1";
                        }}
                      >
                        119 호출
                      </button>
                      <button
                        onClick={() => {
                          // 취소 버튼: 상태 변경하지 않고 해당 구역의 실시간 캠으로 복귀
                          // 팝업만 닫고 CCTV 화면은 유지
                          setShowBottomRightPopup(null);
                        }}
                        style={{
                          width: "100%",
                          padding: "10px 16px",
                          background: "var(--text-secondary)",
                          color: "white",
                          border: "none",
                          borderRadius: "8px",
                          cursor: "pointer",
                          fontSize: "14px",
                          fontWeight: 600,
                          transition: "opacity 0.2s",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.opacity = "0.9";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.opacity = "1";
                        }}
                      >
                        취소
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 출입 카메라 모드 (새 창) */}
            {(isCameraMode || showAccessCamera) && (
              <div
                style={{
                  position: "fixed",
                  top: 0,
                  left: 0,
                  width: "100vw",
                  height: "100vh",
                  background: "#000",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                  alignItems: "center",
                  zIndex: 9999,
                }}
              >
                {/* 작업자 선택 UI */}
                {showWorkerInput && (
                  <div
                    style={{
                      position: "fixed",
                      top: "50%",
                      left: "50%",
                      transform: "translate(-50%, -50%)",
                      background: "#1a1a1a",
                      padding: "40px",
                      borderRadius: "12px",
                      border: "1px solid #333",
                      zIndex: 10000,
                      minWidth: "400px",
                      maxWidth: "90%",
                      boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)",
                    }}
                  >
                    <h2
                      style={{
                        color: "#fff",
                        marginBottom: "20px",
                        fontSize: "24px",
                        textAlign: "center",
                      }}
                    >
                      작업자 ID 입력
                    </h2>
                    {loadingWorkers ? (
                      <div
                        style={{
                          color: "#fff",
                          textAlign: "center",
                          padding: "20px",
                        }}
                      >
                        작업자 목록을 불러오는 중...
                      </div>
                    ) : (
                      <div>
                        <label
                          style={{
                            display: "block",
                            color: "#fff",
                            marginBottom: "8px",
                            fontSize: "14px",
                          }}
                        >
                          작업자 ID (Worker ID)
                        </label>
                        <input
                          type="text"
                          value={workerCode}
                          onChange={(e) => {
                            // 숫자만 입력 허용
                            const newValue = e.target.value.replace(
                              /[^0-9]/g,
                              ""
                            );
                            setWorkerCode(newValue);
                            // workerCodeRef도 즉시 업데이트
                            workerCodeRef.current = newValue;
                            // 에러 메시지 초기화
                            setWorkerCodeError("");
                            // 작업자 목록에서 찾아서 이름 설정 (선택사항)
                            const selected = workersList.find(
                              (w) => (w.workerId || w.worker_id) === newValue
                            );
                            setWorkerName(
                              selected
                                ? selected.workerName || selected.name || ""
                                : ""
                            );
                          }}
                          onKeyPress={(e) => {
                            if (e.key === "Enter") {
                              // 현재 입력 필드의 값을 직접 전달
                              const currentValue =
                                workerCodeRef.current || workerCode;
                              const codeString = currentValue
                                ? String(currentValue).trim()
                                : "";
                              if (codeString) {
                                handleStartSession(codeString);
                              } else {
                                setWorkerCodeError("작업자 ID를 입력해주세요.");
                              }
                            }
                          }}
                          placeholder="작업자 ID를 입력하세요"
                          style={{
                            width: "100%",
                            padding: "12px",
                            fontSize: "18px",
                            background: "#2a2a2a",
                            color: "#fff",
                            border: workerCodeError
                              ? "1px solid #DC2626"
                              : "1px solid #444",
                            borderRadius: "8px",
                            marginBottom: workerCodeError ? "8px" : "20px",
                            outline: "none",
                            boxSizing: "border-box",
                            cursor: "text",
                          }}
                          autoFocus
                        />
                        {workerCodeError && (
                          <div
                            style={{
                              color: "#DC2626",
                              marginBottom: "20px",
                              fontSize: "14px",
                              textAlign: "center",
                            }}
                          >
                            {workerCodeError}
                          </div>
                        )}
                        {workerName && !workerCodeError && (
                          <p
                            style={{
                              color: "#4CAF50",
                              marginBottom: "10px",
                              fontSize: "14px",
                              textAlign: "center",
                            }}
                          >
                            {workerName}
                          </p>
                        )}
                        <button
                          onClick={() => {
                            // 현재 입력 필드의 값을 직접 전달 (상태 업데이트 대기 없이)
                            const currentValue =
                              workerCodeRef.current || workerCode;
                            const codeString = currentValue
                              ? String(currentValue).trim()
                              : "";
                            if (codeString) {
                              handleStartSession(codeString);
                            } else {
                              setWorkerCodeError("작업자 ID를 입력해주세요.");
                            }
                          }}
                          disabled={!workerCode.trim()}
                          style={{
                            width: "100%",
                            padding: "12px",
                            fontSize: "18px",
                            background: workerCode.trim() ? "#4CAF50" : "#555",
                            color: "#fff",
                            border: "none",
                            borderRadius: "8px",
                            cursor: workerCode.trim()
                              ? "pointer"
                              : "not-allowed",
                            fontWeight: "bold",
                          }}
                        >
                          시작
                        </button>
                      </div>
                    )}
                  </div>
                )}

                <h1
                  style={{
                    color: "#fff",
                    fontWeight: 300,
                    marginBottom: "10px",
                    fontSize: "24px",
                    position: "absolute",
                    top: "10px",
                    left: "50%",
                    transform: "translateX(-50%)",
                    zIndex: 10,
                    textAlign: "center",
                    width: "100%",
                    padding: "0 20px",
                    display: showWorkerInput ? "none" : "block",
                  }}
                >
                  {captureStep === "complete"
                    ? ""
                    : captureMessage ||
                      (modelsLoaded
                        ? captureStep === "front"
                          ? "정면을 향해주세요"
                          : captureStep === "left"
                          ? "왼쪽을 향해주세요"
                          : "오른쪽을 향해주세요"
                        : "AI 모델 로딩 중...")}
                </h1>

                <div
                  style={{
                    position: "relative",
                    width: "100%",
                    maxWidth: "1280px",
                    height: "100%",
                    maxHeight: window.innerWidth <= 768 ? "100vh" : "720px",
                    minHeight: window.innerWidth <= 768 ? "50vh" : "auto",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    margin: "0 auto",
                  }}
                >
                  <video
                    ref={accessCameraVideoRef}
                    autoPlay
                    playsInline
                    muted
                    style={{
                      width: "100%",
                      height: "100%",
                      transform: "scaleX(-1)",
                      objectFit: "cover",
                      willChange: "auto",
                      backfaceVisibility: "hidden",
                      WebkitBackfaceVisibility: "hidden",
                      display: showWorkerInput ? "none" : "block",
                    }}
                  />

                  {/* 촬영 중 오버레이 - 자연스러운 플래시 효과 */}
                  {!showWorkerInput && isAutoCapturing && (
                    <div
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",
                        height: "100%",
                        background: "rgba(255, 255, 255, 0)",
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        zIndex: 2000,
                        animation: "cameraFlash 0.4s ease-out",
                        pointerEvents: "none",
                      }}
                    >
                      <div
                        style={{
                          background: "rgba(0, 0, 0, 0.75)",
                          color: "#fff",
                          padding: "14px 28px",
                          borderRadius: "10px",
                          fontSize: "18px",
                          fontWeight: "600",
                          display: "flex",
                          alignItems: "center",
                          gap: "12px",
                          boxShadow: "0 4px 16px rgba(0, 0, 0, 0.5)",
                          animation: "fadeInOut 0.5s ease-out",
                        }}
                      >
                        <i
                          className="fas fa-check-circle"
                          style={{ fontSize: "22px", color: "#4ade80" }}
                        ></i>
                        <span>촬영 완료</span>
                      </div>
                    </div>
                  )}

                  {/* 카운트다운 타이머 표시 (정면 촬영 시) - 얼굴 위쪽에 작게 배치 */}
                  {!showWorkerInput &&
                    countdown > 0 &&
                    captureStep !== "complete" && (
                      <div
                        style={{
                          position: "absolute",
                          top: "20%",
                          left: "50%",
                          transform: "translateX(-50%)",
                          zIndex: 2000,
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          justifyContent: "center",
                          gap: "10px",
                          pointerEvents: "none",
                        }}
                      >
                        <div
                          style={{
                            width: "100px",
                            height: "100px",
                            borderRadius: "50%",
                            background: "rgba(59, 130, 246, 0.95)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: "60px",
                            fontWeight: "bold",
                            color: "#fff",
                            boxShadow: "0 0 20px rgba(59, 130, 246, 0.6)",
                            animation: "pulse 1s ease-in-out infinite",
                            border: "3px solid #fff",
                          }}
                        >
                          {countdown}
                        </div>
                        <div
                          style={{
                            background: "rgba(0, 0, 0, 0.7)",
                            color: "#fff",
                            padding: "8px 16px",
                            borderRadius: "8px",
                            fontSize: "16px",
                            fontWeight: "600",
                          }}
                        >
                          {captureStep === "front"
                            ? "정면 촬영 준비 중..."
                            : captureStep === "left"
                            ? "왼쪽 촬영 준비 중..."
                            : "오른쪽 촬영 준비 중..."}
                        </div>
                      </div>
                    )}

                  {/* 진행 상태 표시 */}
                  {captureStep !== "complete" && (
                    <div
                      style={{
                        position: "absolute",
                        top: "60px",
                        left: "50%",
                        transform: "translateX(-50%)",
                        zIndex: 1000,
                        display: "flex",
                        gap: "8px",
                        alignItems: "center",
                        background: "rgba(0, 0, 0, 0.6)",
                        padding: "12px 20px",
                        borderRadius: "30px",
                        backdropFilter: "blur(10px)",
                      }}
                    >
                      {["front", "left", "right"].map((step, index) => {
                        const stepIndex = ["front", "left", "right"].indexOf(
                          captureStep
                        );
                        const isCompleted = index < stepIndex;
                        const isCurrent = index === stepIndex;

                        return (
                          <div
                            key={step}
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "8px",
                            }}
                          >
                            <div
                              style={{
                                width: "36px",
                                height: "36px",
                                borderRadius: "50%",
                                background: isCompleted
                                  ? "#4ade80"
                                  : isCurrent
                                  ? "#3b82f6"
                                  : "rgba(255, 255, 255, 0.3)",
                                color:
                                  isCompleted || isCurrent
                                    ? "#fff"
                                    : "rgba(255, 255, 255, 0.6)",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                fontWeight: "bold",
                                fontSize: "14px",
                                transition: "all 0.3s ease",
                                border: isCurrent
                                  ? "2px solid #fff"
                                  : "2px solid transparent",
                                boxShadow: isCurrent
                                  ? "0 0 10px rgba(59, 130, 246, 0.5)"
                                  : "none",
                              }}
                            >
                              {isCompleted ? "✓" : index + 1}
                            </div>
                            {index < 2 && (
                              <div
                                style={{
                                  width: "30px",
                                  height: "2px",
                                  background: isCompleted
                                    ? "#4ade80"
                                    : "rgba(255, 255, 255, 0.3)",
                                  transition: "all 0.3s ease",
                                }}
                              />
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* 가이드라인 오버레이 */}
                  <div
                    style={{
                      position: "absolute",
                      top: "50%",
                      left: "50%",
                      transform: "translate(-50%, -50%)",
                      width: "400px",
                      height: "500px",
                      border: faceInGuide
                        ? "3px solid #4ade80"
                        : "3px solid rgba(255, 255, 255, 0.8)",
                      borderRadius: "20px",
                      pointerEvents: "none",
                      boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.7)",
                      transition: "border-color 0.3s ease",
                      overflow: "hidden",
                    }}
                  >
                    {/* 단계별 가이드 이미지 */}
                    {captureStep === "front" && (
                      <img
                        src="/guide-front.png"
                        alt="정면 가이드"
                        style={{
                          width: "110%",
                          height: "110%",
                          objectFit: "cover",
                          objectPosition: "center",
                          marginLeft: "-5%",
                          marginTop: "-5%",
                          opacity: faceInGuide ? 0.9 : 0.7,
                          filter: faceInGuide
                            ? "brightness(1.2) saturate(1.5) drop-shadow(0 0 8px rgba(74, 222, 128, 0.8))"
                            : "none",
                          transition: "opacity 0.3s ease, filter 0.3s ease",
                          pointerEvents: "none",
                        }}
                        onError={(e) => {
                          e.target.style.display = "none";
                        }}
                      />
                    )}

                    {captureStep === "left" && (
                      <img
                        src="/guide-left.png"
                        alt="왼쪽 측면 가이드"
                        style={{
                          width: "110%",
                          height: "110%",
                          objectFit: "cover",
                          objectPosition: "center",
                          marginLeft: "-5%",
                          marginTop: "-5%",
                          opacity: faceInGuide ? 0.9 : 0.7,
                          filter: faceInGuide
                            ? "brightness(1.2) saturate(1.5) drop-shadow(0 0 8px rgba(74, 222, 128, 0.8))"
                            : "none",
                          transition: "opacity 0.3s ease, filter 0.3s ease",
                          pointerEvents: "none",
                        }}
                        onError={(e) => {
                          e.target.style.display = "none";
                        }}
                      />
                    )}

                    {captureStep === "right" && (
                      <img
                        src="/guide-right.png"
                        alt="오른쪽 측면 가이드"
                        style={{
                          width: "110%",
                          height: "110%",
                          objectFit: "cover",
                          objectPosition: "center",
                          marginLeft: "-5%",
                          marginTop: "-5%",
                          opacity: faceInGuide ? 0.9 : 0.7,
                          filter: faceInGuide
                            ? "brightness(1.2) saturate(1.5) drop-shadow(0 0 8px rgba(74, 222, 128, 0.8))"
                            : "none",
                          transition: "opacity 0.3s ease, filter 0.3s ease",
                          pointerEvents: "none",
                        }}
                        onError={(e) => {
                          e.target.style.display = "none";
                        }}
                      />
                    )}
                  </div>

                  {/* 숨겨진 캔버스 (촬영용) */}
                  <canvas
                    ref={accessCameraCanvasRef}
                    style={{ display: "none" }}
                  />
                </div>
              </div>
            )}

            {/* Logs */}
            {activePage === "logs" && (
              <div id="page-logs" className="page active">
                <WeatherTimeBar />
                <div className="card">
                  <div className="card-header">
                    <h2 className="card-title">{t.logs.title}</h2>
                  </div>
                  <div className="card-body">
                    {/* 탭 메뉴 */}
                    <div className="log-tabs">
                      <button
                        className={`log-tab ${
                          logTab === "table" ? "active" : ""
                        }`}
                        onClick={() => setLogTab("table")}
                      >
                        {language === "ko" ? "테이블" : "Table"}
                      </button>
                      <button
                        className={`log-tab ${
                          logTab === "timeline" ? "active" : ""
                        }`}
                        onClick={() => setLogTab("timeline")}
                      >
                        {language === "ko" ? "타임라인" : "Timeline"}
                      </button>
                    </div>

                    {/* 테이블 뷰 */}
                    {logTab === "table" && (
                      <div
                        style={{
                          overflowX: "auto",
                          overflowY: "auto",
                          flex: 1,
                          minHeight: 0,
                        }}
                      >
                        <table className="full-log-table" id="fullLogTable">
                          <thead>
                            <tr>
                              <th className="col-worker">
                                <div className="table-header-filter">
                                  {t.logs.worker}
                                  <button
                                    className={`filter-btn ${
                                      tableFilters.worker.length > 0
                                        ? "active"
                                        : ""
                                    }`}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setShowFilterDropdown(
                                        showFilterDropdown === "worker"
                                          ? null
                                          : "worker"
                                      );
                                    }}
                                  >
                                    <i className="fas fa-filter"></i>
                                  </button>
                                  {showFilterDropdown === "worker" && (
                                    <div
                                      ref={filterDropdownRef}
                                      className="filter-dropdown"
                                    >
                                      <div className="filter-dropdown-header">
                                        <span>필터</span>
                                        {tableFilters.worker.length > 0 && (
                                          <button
                                            onClick={() =>
                                              clearFilter("worker")
                                            }
                                            className="clear-filter-btn"
                                          >
                                            초기화
                                          </button>
                                        )}
                                      </div>
                                      <div className="filter-options">
                                        {filterOptions.workers.map((worker) => (
                                          <label
                                            key={worker}
                                            className="filter-option"
                                          >
                                            <input
                                              type="checkbox"
                                              checked={tableFilters.worker.includes(
                                                worker
                                              )}
                                              onChange={() =>
                                                toggleFilter("worker", worker)
                                              }
                                            />
                                            <span>{worker}</span>
                                          </label>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </th>
                              <th className="col-zone">
                                <div className="table-header-filter">
                                  {t.logs.zone}
                                  <button
                                    className={`filter-btn ${
                                      tableFilters.zone.length > 0
                                        ? "active"
                                        : ""
                                    }`}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setShowFilterDropdown(
                                        showFilterDropdown === "zone"
                                          ? null
                                          : "zone"
                                      );
                                    }}
                                  >
                                    <i className="fas fa-filter"></i>
                                  </button>
                                  {showFilterDropdown === "zone" && (
                                    <div
                                      ref={filterDropdownRef}
                                      className="filter-dropdown"
                                    >
                                      <div className="filter-dropdown-header">
                                        <span>필터</span>
                                        {tableFilters.zone.length > 0 && (
                                          <button
                                            onClick={() => clearFilter("zone")}
                                            className="clear-filter-btn"
                                          >
                                            초기화
                                          </button>
                                        )}
                                      </div>
                                      <div className="filter-options">
                                        {filterOptions.zones.map((zone) => (
                                          <label
                                            key={zone}
                                            className="filter-option"
                                          >
                                            <input
                                              type="checkbox"
                                              checked={tableFilters.zone.includes(
                                                zone
                                              )}
                                              onChange={() =>
                                                toggleFilter("zone", zone)
                                              }
                                            />
                                            <span>{zone}</span>
                                          </label>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </th>
                              <th className="col-time">{t.logs.time}</th>
                              <th className="col-processing-time">
                                {t.logs.processingTime}
                              </th>
                              <th className="col-result">
                                <div className="table-header-filter">
                                  {t.logs.result}
                                  <button
                                    className={`filter-btn ${
                                      tableFilters.status.length > 0
                                        ? "active"
                                        : ""
                                    }`}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setShowFilterDropdown(
                                        showFilterDropdown === "status"
                                          ? null
                                          : "status"
                                      );
                                    }}
                                  >
                                    <i className="fas fa-filter"></i>
                                  </button>
                                  {showFilterDropdown === "status" && (
                                    <div
                                      ref={filterDropdownRef}
                                      className="filter-dropdown"
                                    >
                                      <div className="filter-dropdown-header">
                                        <span>필터</span>
                                        {tableFilters.status.length > 0 && (
                                          <button
                                            onClick={() =>
                                              clearFilter("status")
                                            }
                                            className="clear-filter-btn"
                                          >
                                            초기화
                                          </button>
                                        )}
                                      </div>
                                      <div className="filter-options">
                                        <label className="filter-option">
                                          <input
                                            type="checkbox"
                                            checked={tableFilters.status.includes(
                                              "normal"
                                            )}
                                            onChange={() =>
                                              toggleFilter("status", "normal")
                                            }
                                          />
                                          <span>{t.logs.actionComplete}</span>
                                        </label>
                                        <label className="filter-option">
                                          <input
                                            type="checkbox"
                                            checked={tableFilters.status.includes(
                                              "critical"
                                            )}
                                            onChange={() =>
                                              toggleFilter("status", "critical")
                                            }
                                          />
                                          <span>{t.logs.needsCheck}</span>
                                        </label>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {filteredLogs.length === 0 ? (
                              <tr>
                                <td colSpan="5" className="no-data">
                                  {t.calendar.noData}
                                </td>
                              </tr>
                            ) : (
                              filteredLogs.map((row, i) => (
                                <tr
                                  key={i}
                                  onClick={() => {
                                    // 이미지 경로가 있으면 이미지 모달 표시
                                    if (row.image_path) {
                                      setSelectedImagePath(row.image_path);
                                      setShowImageModal(true);
                                    }
                                  }}
                                  style={{
                                    backgroundColor:
                                      row.status === "normal"
                                        ? "rgba(34, 197, 94, 0.1)" // 조치완료: 반투명 초록색
                                        : "rgba(239, 68, 68, 0.1)", // 확인 필요: 반투명 빨간색
                                    cursor: row.image_path
                                      ? "pointer"
                                      : "default", // 이미지가 있으면 포인터 커서
                                  }}
                                >
                                  <td className="col-worker">{row.worker}</td>
                                  <td className="col-zone">{row.zone}</td>
                                  <td className="col-time">{row.time}</td>
                                  <td className="col-processing-time">
                                    {row.processing_time || ""}
                                  </td>
                                  <td className="col-result">
                                    <span
                                      className="status-tag"
                                      style={{
                                        fontWeight: 600,
                                        padding: "6px 12px",
                                        borderRadius: 12,
                                        fontSize: "16px",
                                        backgroundColor:
                                          row.status === "normal"
                                            ? "var(--accent-green)"
                                            : "var(--accent-red)",
                                        color: "white",
                                      }}
                                    >
                                      {row.status === "normal"
                                        ? t.logs.actionComplete
                                        : t.logs.needsCheck}
                                    </span>
                                  </td>
                                </tr>
                              ))
                            )}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {/* 타임라인 뷰 */}
                    {logTab === "timeline" && (
                      <div className="timeline-container">
                        {fullLogs.length === 0 ? (
                          <div
                            className="no-data"
                            style={{
                              padding: "40px",
                              textAlign: "center",
                              color: "var(--text-secondary)",
                            }}
                          >
                            {t.calendar.noData}
                          </div>
                        ) : (
                          <div className="timeline">
                            {fullLogs.map((row, i) => {
                              const date = new Date(row.time);
                              const timeStr = date.toLocaleTimeString("ko-KR", {
                                hour: "2-digit",
                                minute: "2-digit",
                              });
                              const riskType = row.risk || "";
                              const isFallDetection =
                                riskType === "넘어짐 감지" ||
                                riskType === "Fall Detection";

                              // 날짜 구분선을 위한 날짜 비교
                              const currentDateStr = date.toLocaleDateString(
                                "ko-KR",
                                {
                                  year: "numeric",
                                  month: "long",
                                  day: "numeric",
                                }
                              );
                              const prevDateStr =
                                i > 0
                                  ? new Date(
                                      fullLogs[i - 1].time
                                    ).toLocaleDateString("ko-KR", {
                                      year: "numeric",
                                      month: "long",
                                      day: "numeric",
                                    })
                                  : null;
                              const showDateDivider =
                                prevDateStr !== currentDateStr;

                              return (
                                <React.Fragment key={i}>
                                  {showDateDivider && (
                                    <div className="timeline-date-divider">
                                      <div className="timeline-date-divider-line"></div>
                                      <span className="timeline-date-divider-text">
                                        {currentDateStr}
                                      </span>
                                      <div className="timeline-date-divider-line"></div>
                                    </div>
                                  )}
                                  <div
                                    className="timeline-item"
                                    onClick={() => {
                                      // 이미지 경로가 있으면 이미지 모달 표시
                                      if (row.image_path) {
                                        setSelectedImagePath(row.image_path);
                                        setShowImageModal(true);
                                      }
                                    }}
                                    style={{
                                      cursor: row.image_path
                                        ? "pointer"
                                        : "default",
                                    }}
                                  >
                                    <div
                                      className="timeline-dot"
                                      style={{
                                        backgroundColor:
                                          row.status === "normal"
                                            ? "#10b981"
                                            : "#ef4444",
                                      }}
                                    ></div>
                                    <div
                                      className="timeline-content"
                                      style={{
                                        backgroundColor: isFallDetection
                                          ? "rgba(239, 68, 68, 0.1)"
                                          : "transparent",
                                      }}
                                    >
                                      <div className="timeline-header">
                                        <span className="timeline-worker">
                                          {row.worker}
                                        </span>
                                        <span className="timeline-time">
                                          {timeStr}
                                        </span>
                                      </div>
                                      <div className="timeline-body">
                                        <button
                                          className="badge-btn"
                                          disabled
                                          style={{
                                            display: "inline-flex",
                                            alignItems: "center",
                                            gap: "6px",
                                            background: isFallDetection
                                              ? "#EF4444" // 빨간색 (넘어짐)
                                              : "#F97316", // 오렌지색 (PPE)
                                            color: "#FFFFFF",
                                            padding: "6px 14px",
                                            fontSize: "12px",
                                            fontWeight: 600,
                                            cursor: "default",
                                            marginRight: "8px",
                                          }}
                                        >
                                          {isFallDetection ? (
                                            <i
                                              className="fas fa-exclamation-triangle"
                                              style={{ fontSize: "12px" }}
                                            />
                                          ) : (
                                            <i
                                              className="fas fa-hard-hat"
                                              style={{ fontSize: "12px" }}
                                            />
                                          )}
                                          {translateRiskType(
                                            riskType,
                                            language,
                                            t
                                          )}
                                        </button>
                                        <button
                                          className="timeline-status-btn"
                                          style={{
                                            display: "inline-flex",
                                            alignItems: "center",
                                            gap: "6px",
                                            fontWeight: 600,
                                            padding: "6px 14px",
                                            borderRadius: "9999px",
                                            fontSize: "12px",
                                            backgroundColor:
                                              row.status === "normal"
                                                ? "#10b981"
                                                : "#ef4444",
                                            color: "white",
                                            border: "none",
                                            cursor: "pointer",
                                          }}
                                        >
                                          {row.status !== "normal" && (
                                            <span
                                              style={{
                                                width: "6px",
                                                height: "6px",
                                                borderRadius: "50%",
                                                backgroundColor: "#ffffff",
                                                display: "inline-block",
                                              }}
                                            />
                                          )}
                                          {row.status === "normal"
                                            ? t.logs.actionComplete
                                            : t.logs.needsCheck}
                                        </button>
                                      </div>
                                    </div>
                                  </div>
                                </React.Fragment>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Calendar */}
            {activePage === "calendar" && (
              <div id="page-calendar" className="page active">
                <WeatherTimeBar />
                <div className="calendar-page-grid">
                  <div className="card">
                    <div className="card-body">
                      <div className="calendar-header">
                        <h3 id="calendarMonthYear">
                          {currentDate.getFullYear()}년{" "}
                          {currentDate.getMonth() + 1}월
                        </h3>
                        <div className="calendar-nav">
                          <button
                            id="calPrevBtn"
                            onClick={() =>
                              setCurrentDate(
                                new Date(
                                  currentDate.getFullYear(),
                                  currentDate.getMonth() - 1,
                                  1
                                )
                              )
                            }
                          >
                            <i className="fas fa-chevron-left"></i>
                          </button>
                          <button
                            id="calNextBtn"
                            onClick={() =>
                              setCurrentDate(
                                new Date(
                                  currentDate.getFullYear(),
                                  currentDate.getMonth() + 1,
                                  1
                                )
                              )
                            }
                          >
                            <i className="fas fa-chevron-right"></i>
                          </button>
                        </div>
                      </div>

                      <div className="calendar-grid">
                        {t.calendar.weekDays.map((d) => (
                          <div key={d} className="day-header">
                            {d}
                          </div>
                        ))}
                      </div>

                      <div className="calendar-grid" id="calendarBody">
                        {daysForCalendar.map((cell, idx) => {
                          const classes = [
                            "day",
                            cell.current ? "current-month" : "",
                          ];
                          // DB에서 로드한 데이터 기반으로 이벤트 점 표시 및 개수 계산
                          let showDot = false;
                          let fallCount = 0;
                          let ppeCount = 0;
                          const today = new Date();
                          const isToday =
                            cell.current &&
                            cell.date &&
                            cell.date.getDate() === today.getDate() &&
                            cell.date.getMonth() === today.getMonth() &&
                            cell.date.getFullYear() === today.getFullYear();

                          if (cell.current && cell.date) {
                            // 해당 날짜에 이벤트가 있는지 확인 (로컬 시간 기준)
                            const formatDate = (date) => {
                              const year = date.getFullYear();
                              const month = String(
                                date.getMonth() + 1
                              ).padStart(2, "0");
                              const day = String(date.getDate()).padStart(
                                2,
                                "0"
                              );
                              return `${year}-${month}-${day}`;
                            };
                            const dateStr = formatDate(cell.date);
                            const eventsForDate = fullLogs.filter((log) => {
                              if (!log || !log.time) return false;
                              try {
                                const logDateStr = log.time.split(" ")[0];
                                return logDateStr === dateStr;
                              } catch (e) {
                                console.warn(
                                  "[캘린더] 날짜 파싱 오류:",
                                  e,
                                  log
                                );
                                return false;
                              }
                            });

                            // 위험 유형별 개수 계산 (유연한 매칭)
                            fallCount = eventsForDate.filter((e) => {
                              if (!e || !e.risk) return false;
                              const risk = String(e.risk).toLowerCase();
                              return (
                                risk.includes("넘어짐") ||
                                risk.includes("fall") ||
                                risk.includes("낙상")
                              );
                            }).length;
                            ppeCount = eventsForDate.filter((e) => {
                              if (!e || !e.risk) return false;
                              const risk = String(e.risk).toLowerCase();
                              return (
                                risk.includes("안전모") ||
                                risk.includes("helmet") ||
                                risk.includes("헬멧") ||
                                risk.includes("안전조끼") ||
                                risk.includes("vest") ||
                                risk.includes("조끼") ||
                                risk.includes("마스크") ||
                                risk.includes("mask")
                              );
                            }).length;

                            showDot = fallCount > 0 || ppeCount > 0;
                          }

                          // 오늘 날짜 클래스 추가
                          if (isToday) {
                            classes.push("today");
                          }
                          // 개별 선택된 날짜 표시
                          if (
                            cell.current &&
                            cell.date &&
                            isDateSelected(cell.date)
                          )
                            classes.push("selected-date");
                          // 범위 선택 표시 (개별 선택이 없을 때만)
                          if (selectedDates.length === 0) {
                            if (
                              cell.current &&
                              startDate &&
                              endDate &&
                              cell.date &&
                              inRange(new Date(cell.date))
                            )
                              classes.push("in-range");
                            if (
                              cell.current &&
                              cell.date &&
                              startDate &&
                              isSameDay(cell.date, startDate)
                            )
                              classes.push("start-date");
                            if (
                              cell.current &&
                              cell.date &&
                              endDate &&
                              isSameDay(cell.date, endDate)
                            )
                              classes.push("end-date");
                          }

                          return (
                            <div
                              key={idx}
                              className={classes.join(" ")}
                              data-date={cell.date?.toISOString()}
                              onClick={() => handleDayClick(cell.date)}
                              onMouseDown={(e) =>
                                handleDayMouseDown(cell.date, e)
                              }
                              onMouseMove={(e) => {
                                handleDayMouseMove(cell.date, e);
                              }}
                              onMouseEnter={() =>
                                handleDayMouseEnter(cell.date)
                              }
                              onMouseLeave={handleDayMouseLeave}
                              onMouseUp={handleDayMouseUp}
                              style={{ userSelect: "none" }}
                            >
                              {cell.label}
                              {(fallCount > 0 || ppeCount > 0) && (
                                <div
                                  style={{
                                    marginTop: "4px",
                                    fontSize: "9px",
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: "2px",
                                  }}
                                >
                                  {fallCount > 0 && (
                                    <div
                                      style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: "4px",
                                        color:
                                          classes.includes("selected-date") ||
                                          classes.includes("start-date") ||
                                          classes.includes("end-date")
                                            ? "white"
                                            : "#EF4444",
                                        fontWeight: 700,
                                      }}
                                    >
                                      <i
                                        className="fas fa-exclamation-triangle"
                                        style={{ fontSize: "15px" }}
                                      ></i>
                                      <span style={{ fontSize: "15px" }}>
                                        {fallCount}건
                                      </span>
                                    </div>
                                  )}
                                  {ppeCount > 0 && (
                                    <div
                                      style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: "4px",
                                        color:
                                          classes.includes("selected-date") ||
                                          classes.includes("start-date") ||
                                          classes.includes("end-date")
                                            ? "white"
                                            : "#F59E0B",
                                        fontWeight: 700,
                                      }}
                                    >
                                      <i
                                        className="fas fa-hard-hat"
                                        style={{ fontSize: "15px" }}
                                      ></i>
                                      <span style={{ fontSize: "15px" }}>
                                        {ppeCount}건
                                      </span>
                                    </div>
                                  )}
                                </div>
                              )}
                              {showDot && fallCount === 0 && ppeCount === 0 && (
                                <div className="event-dot" />
                              )}
                            </div>
                          );
                        })}
                      </div>

                      {/* 호버 미리보기 툴팁 */}
                      {hoveredDate &&
                        (() => {
                          const formatDate = (date) => {
                            const year = date.getFullYear();
                            const month = String(date.getMonth() + 1).padStart(
                              2,
                              "0"
                            );
                            const day = String(date.getDate()).padStart(2, "0");
                            return `${year}-${month}-${day}`;
                          };
                          const dateStr = formatDate(hoveredDate);
                          const eventsForDate = fullLogs.filter((log) => {
                            if (!log || !log.time) return false;
                            try {
                              const logDateStr = log.time.split(" ")[0];
                              return logDateStr === dateStr;
                            } catch (e) {
                              console.warn(
                                "[캘린더 툴팁] 날짜 파싱 오류:",
                                e,
                                log
                              );
                              return false;
                            }
                          });

                          const fallEvents = eventsForDate.filter((e) => {
                            if (!e || !e.risk) return false;
                            const risk = String(e.risk).toLowerCase();
                            return (
                              risk.includes("넘어짐") ||
                              risk.includes("fall") ||
                              risk.includes("낙상")
                            );
                          });
                          const ppeEvents = eventsForDate.filter((e) => {
                            if (!e || !e.risk) return false;
                            const risk = String(e.risk).toLowerCase();
                            return (
                              risk.includes("안전모") ||
                              risk.includes("helmet") ||
                              risk.includes("헬멧") ||
                              risk.includes("안전조끼") ||
                              risk.includes("vest") ||
                              risk.includes("조끼") ||
                              risk.includes("마스크") ||
                              risk.includes("mask")
                            );
                          });

                          if (eventsForDate.length === 0) return null;

                          return (
                            <div
                              style={{
                                position: "fixed",
                                left: `${tooltipPosition.x + 10}px`,
                                top: `${tooltipPosition.y + 10}px`,
                                background: "var(--card-bg)",
                                border: "1px solid var(--border-color)",
                                borderRadius: "8px",
                                padding: "12px",
                                minWidth: "250px",
                                maxWidth: "350px",
                                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
                                zIndex: 1000,
                                pointerEvents: "none",
                              }}
                            >
                              <div
                                style={{
                                  fontSize: "15px",
                                  fontWeight: 700,
                                  color: "var(--text-primary)",
                                  marginBottom: "10px",
                                  borderBottom: "1px solid var(--border-color)",
                                  paddingBottom: "8px",
                                }}
                              >
                                {hoveredDate.getFullYear()}년{" "}
                                {String(hoveredDate.getMonth() + 1).padStart(
                                  2,
                                  "0"
                                )}
                                월{" "}
                                {String(hoveredDate.getDate()).padStart(2, "0")}
                                일
                              </div>

                              {fallEvents.length > 0 && (
                                <div style={{ marginBottom: "10px" }}>
                                  <div
                                    style={{
                                      display: "flex",
                                      alignItems: "center",
                                      gap: "8px",
                                      fontSize: "15px",
                                      fontWeight: 600,
                                      color: "#EF4444",
                                      marginBottom: "6px",
                                    }}
                                  >
                                    <i
                                      className="fas fa-exclamation-triangle"
                                      style={{ fontSize: "15px" }}
                                    ></i>
                                    <span style={{ fontSize: "15px" }}>
                                      {fallEvents.length}건
                                    </span>
                                  </div>
                                  <div
                                    style={{
                                      fontSize: "15px",
                                      color: "var(--text-secondary)",
                                    }}
                                  >
                                    {fallEvents.slice(0, 3).map((e, i) => (
                                      <div
                                        key={i}
                                        style={{ marginBottom: "3px" }}
                                      >
                                        • {e.worker} -{" "}
                                        {e.time?.split(" ")[1] || ""}
                                      </div>
                                    ))}
                                    {fallEvents.length > 3 && (
                                      <div
                                        style={{
                                          marginTop: "6px",
                                          fontStyle: "italic",
                                        }}
                                      >
                                        외 {fallEvents.length - 3}건...
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}

                              {ppeEvents.length > 0 && (
                                <div>
                                  <div
                                    style={{
                                      display: "flex",
                                      alignItems: "center",
                                      gap: "8px",
                                      fontSize: "15px",
                                      fontWeight: 600,
                                      color: "#F59E0B",
                                      marginBottom: "6px",
                                    }}
                                  >
                                    <i
                                      className="fas fa-hard-hat"
                                      style={{ fontSize: "15px" }}
                                    ></i>
                                    <span style={{ fontSize: "15px" }}>
                                      {ppeEvents.length}건
                                    </span>
                                  </div>
                                  <div
                                    style={{
                                      fontSize: "15px",
                                      color: "var(--text-secondary)",
                                    }}
                                  >
                                    {ppeEvents.slice(0, 3).map((e, i) => (
                                      <div
                                        key={i}
                                        style={{ marginBottom: "3px" }}
                                      >
                                        • {e.worker} -{" "}
                                        {e.time?.split(" ")[1] || ""}
                                      </div>
                                    ))}
                                    {ppeEvents.length > 3 && (
                                      <div
                                        style={{
                                          marginTop: "6px",
                                          fontStyle: "italic",
                                        }}
                                      >
                                        외 {ppeEvents.length - 3}건...
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })()}

                      {/* 아이콘 범례 */}
                      <div
                        style={{
                          marginTop: "16px",
                          paddingTop: "16px",
                          borderTop: "1px solid var(--border-color)",
                          display: "flex",
                          justifyContent: "flex-end",
                          gap: "24px",
                          flexWrap: "wrap",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "6px",
                          }}
                        >
                          <i
                            className="fas fa-exclamation-triangle"
                            style={{
                              fontSize: "15px",
                              color: "#EF4444",
                            }}
                          ></i>
                          <span
                            style={{
                              fontSize: "15px",
                              color: "var(--text-secondary)",
                            }}
                          >
                            넘어짐 감지
                          </span>
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "6px",
                          }}
                        >
                          <i
                            className="fas fa-hard-hat"
                            style={{
                              fontSize: "15px",
                              color: "#F59E0B",
                            }}
                          ></i>
                          <span
                            style={{
                              fontSize: "15px",
                              color: "var(--text-secondary)",
                            }}
                          >
                            PPE 미착용
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="card calendar-details">
                    <div className="card-header">
                      <h2 className="card-title">{t.calendar.details}</h2>
                    </div>
                    <div className="card-body">
                      <div className="filter-section">
                        <div className="filter-label">{t.calendar.period}</div>
                        <div className="range-chips">
                          <button
                            className={`chip ${
                              rangeMode === "week" ? "active" : ""
                            }`}
                            onClick={() => applyQuickRange("week")}
                          >
                            {t.calendar.thisWeek}
                          </button>
                          <button
                            className={`chip ${
                              rangeMode === "1m" ? "active" : ""
                            }`}
                            onClick={() => applyQuickRange("1m")}
                          >
                            {t.calendar.oneMonth}
                          </button>
                          <button
                            className={`chip ${
                              rangeMode === "3m" ? "active" : ""
                            }`}
                            onClick={() => applyQuickRange("3m")}
                          >
                            {t.calendar.threeMonths}
                          </button>
                          <button
                            className={`chip ${
                              rangeMode === "prev" ? "active" : ""
                            }`}
                            onClick={() => applyQuickRange("prev")}
                          >
                            {t.calendar.lastMonth}
                          </button>
                          <button
                            className={`chip ${
                              rangeMode === "year" ? "active" : ""
                            }`}
                            onClick={() => applyQuickRange("year")}
                          >
                            {t.calendar.thisYear}
                          </button>
                        </div>
                      </div>

                      <div className="filter-section">
                        <div className="filter-label">
                          {t.calendar.typeSelect}
                        </div>
                        <div className="type-chips">
                          <button
                            className={`chip ${
                              selectedRiskType === t.calendar.all
                                ? "active"
                                : ""
                            }`}
                            onClick={() => setSelectedRiskType(t.calendar.all)}
                          >
                            {t.calendar.all}
                          </button>
                          <button
                            className={`chip ${
                              selectedRiskType === t.dashboard.unhelmet
                                ? "active"
                                : ""
                            }`}
                            onClick={() =>
                              setSelectedRiskType(t.dashboard.unhelmet)
                            }
                          >
                            {t.dashboard.unhelmet}
                          </button>
                          <button
                            className={`chip ${
                              selectedRiskType === t.dashboard.unvest
                                ? "active"
                                : ""
                            }`}
                            onClick={() =>
                              setSelectedRiskType(t.dashboard.unvest)
                            }
                          >
                            {t.dashboard.unvest}
                          </button>
                          <button
                            className={`chip ${
                              selectedRiskType === t.dashboard.fall
                                ? "active"
                                : ""
                            }`}
                            onClick={() =>
                              setSelectedRiskType(t.dashboard.fall)
                            }
                          >
                            {t.dashboard.fall}
                          </button>
                        </div>
                      </div>

                      <div className="filter-section">
                        <div className="filter-label">
                          {t.calendar.sortSelect}
                        </div>
                        <div className="sort-chips">
                          <button
                            className={`chip ${
                              sortOrder === t.calendar.newest ? "active" : ""
                            }`}
                            onClick={() => setSortOrder(t.calendar.newest)}
                          >
                            {t.calendar.newest}
                          </button>
                          <button
                            className={`chip ${
                              sortOrder === t.calendar.oldest ? "active" : ""
                            }`}
                            onClick={() => setSortOrder(t.calendar.oldest)}
                          >
                            {t.calendar.oldest}
                          </button>
                        </div>
                      </div>

                      <div className="detail-header">
                        <h3 className="detail-title" id="detailDateTitle">
                          {periodDetails.title}
                        </h3>
                        <button
                          className="reset-btn"
                          id="calendarResetBtn"
                          onClick={() => {
                            setCurrentDate(new Date()); // 캘린더를 오늘 날짜로 초기화
                            setRangeMode(null);
                            setStartDate(null);
                            setEndDate(null);
                            setSelectedDates([]); // 개별 날짜 선택도 초기화
                            setSelectedRiskType(t.calendar.all);
                            setSortOrder(t.calendar.newest);
                          }}
                        >
                          {t.calendar.reset}
                        </button>
                      </div>

                      <ul className="log-list" id="detailLogList">
                        {periodDetails.items.length === 0 ? (
                          <li
                            style={{
                              textAlign: "center",
                              color: "var(--text-secondary)",
                            }}
                          >
                            {periodDetails.hint}
                          </li>
                        ) : (
                          periodDetails.items.map((it, i) => (
                            <li className="log-item" key={i}>
                              <div className="log-icon">
                                <i
                                  className={`fas ${
                                    it.risk === "안전모 미착용"
                                      ? "fa-hard-hat"
                                      : it.risk === "안전조끼 미착용"
                                      ? "fa-vest"
                                      : "fa-exclamation-triangle"
                                  }`}
                                ></i>
                              </div>
                              <div className="log-info">
                                <p>
                                  {it.day}
                                  {t.calendar.daySuffix} -{" "}
                                  {translateRiskType(it.risk, language, t)}
                                </p>
                                <div className="time">
                                  {it.who} / {it.time}
                                </div>
                              </div>
                            </li>
                          ))
                        )}
                      </ul>

                      {/* 리포트 내보내기 버튼 - 우측 하단 */}
                      <div
                        style={{
                          marginTop: "auto",
                          paddingTop: "20px",
                          borderTop: "1px solid var(--border-color)",
                          display: "flex",
                          flexDirection: "column",
                          gap: "8px",
                        }}
                      >
                        <div
                          style={{
                            fontSize: "13px",
                            fontWeight: 600,
                            color: "var(--text-primary)",
                            marginBottom: "8px",
                          }}
                        >
                          리포트 내보내기
                        </div>
                        <div
                          style={{
                            display: "flex",
                            gap: "8px",
                            flexWrap: "wrap",
                          }}
                        >
                          <button
                            className="rbtn primary"
                            onClick={exportSelectedPeriodXLSX}
                            disabled={!startDate && selectedDates.length === 0}
                            style={{
                              flex: 1,
                              minWidth: "120px",
                              padding: "10px 16px",
                              fontSize: "13px",
                              opacity:
                                !startDate && selectedDates.length === 0
                                  ? 0.5
                                  : 1,
                              cursor:
                                !startDate && selectedDates.length === 0
                                  ? "not-allowed"
                                  : "pointer",
                            }}
                          >
                            <i className="fa-solid fa-file-excel"></i> 엑셀
                          </button>
                          <button
                            className="rbtn ghost"
                            onClick={exportSelectedPeriodPDF}
                            disabled={!startDate && selectedDates.length === 0}
                            style={{
                              flex: 1,
                              minWidth: "120px",
                              padding: "10px 16px",
                              fontSize: "13px",
                              opacity:
                                !startDate && selectedDates.length === 0
                                  ? 0.5
                                  : 1,
                              cursor:
                                !startDate && selectedDates.length === 0
                                  ? "not-allowed"
                                  : "pointer",
                            }}
                          >
                            <i className="fa-solid fa-file-pdf"></i> PDF
                          </button>
                        </div>
                        {!startDate && selectedDates.length === 0 && (
                          <div
                            style={{
                              fontSize: "11px",
                              color: "var(--text-secondary)",
                              marginTop: "4px",
                            }}
                          >
                            기간을 선택해주세요
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Workers */}
            {activePage === "workers" && (
              <div id="page-workers" className="page active">
                <WeatherTimeBar />
                <div className="card">
                  <div
                    className="card-header"
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <h3 className="card-title" style={{ margin: 0 }}>
                      등록된 작업자 목록
                    </h3>
                    <button
                      onClick={() => {
                        setEditingWorker(null);
                        setWorkerFormData({
                          worker_id: "",
                          name: "",
                          contact: "",
                          team: "",
                          role: "worker",
                          blood_type: "",
                        });
                        setWorkerIdError("");
                        // 출입 관제 관련 상태 초기화 (모달 열 때 workerCode 초기화)
                        setWorkerCode("");
                        setWorkerName("");
                        setShowWorkerModal(true);
                      }}
                      style={{
                        padding: "10px 20px",
                        background: "#3b82f6",
                        color: "white",
                        border: "none",
                        borderRadius: "8px",
                        cursor: "pointer",
                        fontWeight: 600,
                        fontSize: "14px",
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                      }}
                    >
                      <i className="fas fa-plus"></i>
                      <span>작업자 추가</span>
                    </button>
                  </div>
                  <div className="card-body">
                    {/* 팀 필터 및 검색창 일렬 배치 */}
                    <div
                      style={{
                        marginBottom: "20px",
                        display: "flex",
                        alignItems: "center",
                        gap: "20px",
                        flexWrap: "wrap",
                      }}
                    >
                      {/* 팀 필터 드롭다운 */}
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "10px",
                        }}
                      >
                        <label
                          style={{
                            color: "var(--text-primary)",
                            fontWeight: "600",
                            fontSize: "14px",
                            whiteSpace: "nowrap",
                          }}
                        >
                          팀 필터:
                        </label>
                        <select
                          value={selectedTeamFilter}
                          onChange={(e) =>
                            setSelectedTeamFilter(e.target.value)
                          }
                          style={{
                            padding: "8px 12px",
                            borderRadius: "6px",
                            border: "1px solid var(--border-color)",
                            background: "var(--card-bg)",
                            color: "var(--text-primary)",
                            fontSize: "14px",
                            cursor: "pointer",
                            minWidth: "150px",
                            outline: "none",
                          }}
                        >
                          <option value="all">전체</option>
                          {Array.from(
                            new Set(
                              workersList
                                .map((w) => w.team)
                                .filter((team) => team && team.trim() !== "")
                            )
                          )
                            .sort()
                            .map((team) => (
                              <option key={team} value={team}>
                                {team}
                              </option>
                            ))}
                        </select>
                      </div>

                      {/* 통합 검색창 */}
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "10px",
                          position: "relative",
                          flex: 1,
                          minWidth: "300px",
                        }}
                      >
                        <label
                          style={{
                            color: "var(--text-primary)",
                            fontWeight: "600",
                            fontSize: "14px",
                            whiteSpace: "nowrap",
                          }}
                        >
                          검색:
                        </label>
                        <div
                          style={{
                            position: "relative",
                            flex: 1,
                            maxWidth: "400px",
                          }}
                        >
                          <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => {
                              const value = e.target.value;
                              setSearchQuery(value);
                              setShowSearchHistory(value === "" ? false : true);
                            }}
                            onFocus={() => {
                              if (searchHistory.length > 0) {
                                setShowSearchHistory(true);
                              }
                            }}
                            onBlur={() => {
                              // 약간의 딜레이를 주어 클릭 이벤트가 먼저 실행되도록
                              setTimeout(
                                () => setShowSearchHistory(false),
                                200
                              );
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" && searchQuery.trim()) {
                                addToSearchHistory(searchQuery.trim());
                                setShowSearchHistory(false);
                              }
                            }}
                            placeholder="작업자 ID, 이름, 연락처로 검색..."
                            style={{
                              width: "100%",
                              padding: "10px 40px 10px 12px",
                              borderRadius: "8px",
                              border: "1px solid var(--border-color)",
                              background: "var(--card-bg)",
                              color: "var(--text-primary)",
                              fontSize: "14px",
                              outline: "none",
                              transition: "border-color 0.2s",
                            }}
                            onMouseEnter={(e) => {
                              e.target.style.borderColor = "var(--accent-blue)";
                            }}
                            onMouseLeave={(e) => {
                              e.target.style.borderColor =
                                "var(--border-color)";
                            }}
                          />
                          {searchQuery && (
                            <button
                              onClick={() => {
                                setSearchQuery("");
                                setShowSearchHistory(false);
                              }}
                              style={{
                                position: "absolute",
                                right: "8px",
                                top: "50%",
                                transform: "translateY(-50%)",
                                background: "transparent",
                                border: "none",
                                color: "var(--text-secondary)",
                                cursor: "pointer",
                                padding: "4px",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                borderRadius: "4px",
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.background =
                                  "var(--chip-container-bg)";
                                e.currentTarget.style.color =
                                  "var(--text-primary)";
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.background =
                                  "transparent";
                                e.currentTarget.style.color =
                                  "var(--text-secondary)";
                              }}
                            >
                              <i
                                className="fas fa-times"
                                style={{ fontSize: "12px" }}
                              ></i>
                            </button>
                          )}
                          <i
                            className="fas fa-search"
                            style={{
                              position: "absolute",
                              right: searchQuery ? "32px" : "12px",
                              top: "50%",
                              transform: "translateY(-50%)",
                              color: "var(--text-secondary)",
                              fontSize: "14px",
                              pointerEvents: "none",
                            }}
                          ></i>

                          {/* 검색 히스토리 드롭다운 */}
                          {showSearchHistory && searchHistory.length > 0 && (
                            <div
                              style={{
                                position: "absolute",
                                top: "100%",
                                left: 0,
                                right: 0,
                                marginTop: "4px",
                                background: "var(--card-bg)",
                                border: "1px solid var(--border-color)",
                                borderRadius: "8px",
                                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
                                zIndex: 1000,
                                maxHeight: "200px",
                                overflowY: "auto",
                              }}
                            >
                              {searchHistory.map((historyItem, index) => (
                                <div
                                  key={index}
                                  onClick={() => selectFromHistory(historyItem)}
                                  style={{
                                    padding: "10px 12px",
                                    cursor: "pointer",
                                    borderBottom:
                                      index < searchHistory.length - 1
                                        ? "1px solid var(--border-color)"
                                        : "none",
                                    color: "var(--text-primary)",
                                    fontSize: "14px",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "8px",
                                  }}
                                  onMouseEnter={(e) => {
                                    e.currentTarget.style.background =
                                      "var(--chip-container-bg)";
                                  }}
                                  onMouseLeave={(e) => {
                                    e.currentTarget.style.background =
                                      "transparent";
                                  }}
                                >
                                  <i
                                    className="fas fa-history"
                                    style={{
                                      fontSize: "12px",
                                      color: "var(--text-secondary)",
                                    }}
                                  ></i>
                                  <span>{historyItem}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* 검색 결과 개수 표시 */}
                    {(searchQuery || selectedTeamFilter !== "all") && (
                      <div
                        style={{
                          marginBottom: "12px",
                          fontSize: "13px",
                          color: "var(--text-secondary)",
                        }}
                      >
                        {(() => {
                          let filteredCount = workersList.length;
                          if (selectedTeamFilter !== "all") {
                            filteredCount = workersList.filter(
                              (w) => w.team === selectedTeamFilter
                            ).length;
                          }
                          if (searchQuery) {
                            const query = searchQuery.trim().toLowerCase();
                            filteredCount = workersList.filter((worker) => {
                              const workerId = (
                                worker.workerId ||
                                worker.worker_id ||
                                ""
                              ).toLowerCase();
                              const workerName = (
                                worker.workerName ||
                                worker.name ||
                                ""
                              ).toLowerCase();
                              const contact = (
                                worker.contact || ""
                              ).toLowerCase();
                              return (
                                workerId.includes(query) ||
                                workerName.includes(query) ||
                                contact.includes(query)
                              );
                            }).length;
                            if (selectedTeamFilter !== "all") {
                              filteredCount = workersList
                                .filter((w) => w.team === selectedTeamFilter)
                                .filter((worker) => {
                                  const workerId = (
                                    worker.workerId ||
                                    worker.worker_id ||
                                    ""
                                  ).toLowerCase();
                                  const workerName = (
                                    worker.workerName ||
                                    worker.name ||
                                    ""
                                  ).toLowerCase();
                                  const contact = (
                                    worker.contact || ""
                                  ).toLowerCase();
                                  return (
                                    workerId.includes(query) ||
                                    workerName.includes(query) ||
                                    contact.includes(query)
                                  );
                                }).length;
                            }
                          }
                          return `검색 결과: ${filteredCount}명`;
                        })()}
                      </div>
                    )}
                    <div style={{ overflowX: "auto" }}>
                      <table
                        style={{
                          width: "100%",
                          borderCollapse: "collapse",
                          minWidth: "800px",
                        }}
                      >
                        <thead>
                          <tr
                            style={{
                              borderBottom: "2px solid var(--border-color)",
                            }}
                          >
                            <th
                              style={{
                                padding: "8px 12px",
                                textAlign: "left",
                                fontWeight: "600",
                              }}
                            >
                              작업자 ID
                            </th>
                            <th
                              style={{
                                padding: "8px 12px",
                                textAlign: "left",
                                fontWeight: "600",
                              }}
                            >
                              이름
                            </th>
                            <th
                              style={{
                                padding: "8px 12px",
                                textAlign: "left",
                                fontWeight: "600",
                              }}
                            >
                              팀
                            </th>
                            <th
                              style={{
                                padding: "8px 12px",
                                textAlign: "left",
                                fontWeight: "600",
                              }}
                            >
                              연락처
                            </th>
                            <th
                              style={{
                                padding: "8px 12px",
                                textAlign: "left",
                                fontWeight: "600",
                              }}
                            >
                              혈액형
                            </th>
                            <th
                              style={{
                                padding: "8px 12px",
                                textAlign: "left",
                                fontWeight: "600",
                              }}
                            >
                              역할
                            </th>
                            <th
                              style={{
                                padding: "8px 12px",
                                textAlign: "center",
                                fontWeight: "600",
                                width: "150px",
                                cursor: "pointer",
                                userSelect: "none",
                              }}
                              onClick={() =>
                                setShowEditButtons(!showEditButtons)
                              }
                            >
                              작업 {showEditButtons ? "▼" : "▲"}
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {(() => {
                            // 팀 필터링 적용
                            let filteredWorkers =
                              selectedTeamFilter === "all"
                                ? workersList
                                : workersList.filter(
                                    (w) => w.team === selectedTeamFilter
                                  );

                            // 검색 필터링 적용
                            if (searchQuery && searchQuery.trim() !== "") {
                              const query = searchQuery.trim().toLowerCase();
                              filteredWorkers = filteredWorkers.filter(
                                (worker) => {
                                  const workerId = (
                                    worker.workerId ||
                                    worker.worker_id ||
                                    ""
                                  ).toLowerCase();
                                  const workerName = (
                                    worker.workerName ||
                                    worker.name ||
                                    ""
                                  ).toLowerCase();
                                  const contact = (
                                    worker.contact || ""
                                  ).toLowerCase();

                                  return (
                                    workerId.includes(query) ||
                                    workerName.includes(query) ||
                                    contact.includes(query)
                                  );
                                }
                              );
                            }

                            if (filteredWorkers.length === 0) {
                              return (
                                <tr>
                                  <td
                                    colSpan="7"
                                    style={{
                                      textAlign: "center",
                                      padding: "40px",
                                      color: "var(--text-secondary)",
                                    }}
                                  >
                                    {loadingWorkers
                                      ? "작업자 목록을 불러오는 중..."
                                      : searchQuery
                                      ? `"${searchQuery}"에 대한 검색 결과가 없습니다.`
                                      : selectedTeamFilter === "all"
                                      ? "등록된 작업자가 없습니다."
                                      : "선택한 팀에 등록된 작업자가 없습니다."}
                                  </td>
                                </tr>
                              );
                            }

                            return filteredWorkers.map((worker, idx) => (
                              <tr
                                key={
                                  worker._id ||
                                  worker.workerId ||
                                  worker.worker_id ||
                                  `worker-${idx}`
                                }
                                style={{
                                  borderBottom: "1px solid var(--border-color)",
                                  borderLeft:
                                    worker.role === "manager"
                                      ? "4px solid #DC2626"
                                      : "none",
                                  borderRight:
                                    worker.role === "manager"
                                      ? "4px solid #DC2626"
                                      : "none",
                                  backgroundColor:
                                    worker.role === "manager"
                                      ? "rgba(220, 38, 38, 0.203)"
                                      : "transparent",
                                }}
                              >
                                <td
                                  style={{
                                    padding: "12px",
                                    fontWeight: "600",
                                    color: "var(--accent-blue)",
                                    verticalAlign: "middle",
                                    height: "58.5px",
                                  }}
                                >
                                  {highlightText(
                                    worker.workerId || worker.worker_id || "-",
                                    searchQuery
                                  )}
                                </td>
                                <td
                                  style={{
                                    padding: "12px",
                                    verticalAlign: "middle",
                                    height: "58.5px",
                                  }}
                                >
                                  {highlightText(
                                    worker.workerName || worker.name || "-",
                                    searchQuery
                                  )}
                                </td>
                                <td
                                  style={{
                                    padding: "12px",
                                    verticalAlign: "middle",
                                    height: "58.5px",
                                  }}
                                >
                                  {worker.team || "-"}
                                </td>
                                <td
                                  style={{
                                    padding: "12px",
                                    verticalAlign: "middle",
                                    height: "58.5px",
                                  }}
                                >
                                  {highlightText(
                                    worker.contact || "-",
                                    searchQuery
                                  )}
                                </td>
                                <td
                                  style={{
                                    padding: "12px",
                                    verticalAlign: "middle",
                                    height: "58.5px",
                                  }}
                                >
                                  {worker.blood_type || "-"}
                                </td>
                                <td
                                  style={{
                                    padding: "12px",
                                    verticalAlign: "middle",
                                    height: "58.5px",
                                  }}
                                >
                                  {worker.role === "manager"
                                    ? "관리자"
                                    : worker.role === "worker"
                                    ? "작업자"
                                    : worker.role}
                                </td>
                                <td
                                  style={{
                                    padding: "12px",
                                    textAlign: "center",
                                    verticalAlign: "middle",
                                    height: "58.5px",
                                    maxHeight: "58.5px",
                                    overflow: "hidden",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                  }}
                                >
                                  {showEditButtons && (
                                    <button
                                      onClick={async () => {
                                        setEditingWorker(worker);
                                        setWorkerFormData({
                                          worker_id:
                                            worker.workerId ||
                                            worker.worker_id ||
                                            "",
                                          name:
                                            worker.workerName ||
                                            worker.name ||
                                            "",
                                          contact: worker.contact || "",
                                          team: worker.team || "",
                                          role: worker.role || "worker",
                                          blood_type: worker.blood_type || "",
                                        });
                                        setWorkerIdError("");
                                        setWorkerCode("");
                                        setWorkerName("");
                                        setShowWorkerModal(true);

                                        setLoadingTodayImages(true);
                                        try {
                                          const response =
                                            await api.getTodayImages(
                                              worker.workerId ||
                                                worker.worker_id ||
                                                ""
                                            );
                                          if (
                                            response.success &&
                                            response.images
                                          ) {
                                            setTodayImages(response.images);
                                          } else {
                                            setTodayImages([]);
                                          }
                                        } catch (error) {
                                          console.error(
                                            "오늘 이미지 로드 오류:",
                                            error
                                          );
                                          setTodayImages([]);
                                        } finally {
                                          setLoadingTodayImages(false);
                                        }
                                      }}
                                      style={{
                                        padding: "8px 16px",
                                        background: "#3b82f6",
                                        color: "white",
                                        border: "none",
                                        borderRadius: "8px",
                                        cursor: "pointer",
                                        fontSize: "14px",
                                        fontWeight: 600,
                                        display: "flex",
                                        alignItems: "center",
                                        gap: "6px",
                                      }}
                                    >
                                      <i className="fas fa-edit"></i>
                                      <span>수정</span>
                                    </button>
                                  )}
                                </td>
                              </tr>
                            ));
                          })()}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>

                {/* 작업자 생성/수정 모달 */}
                {showWorkerModal && (
                  <div
                    style={{
                      position: "fixed",
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      background: "rgba(0, 0, 0, 0.7)",
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                      zIndex: 10000,
                    }}
                    onClick={(e) => {
                      if (e.target === e.currentTarget) {
                        setShowWorkerModal(false);
                      }
                    }}
                  >
                    <div
                      style={{
                        background: "var(--card-bg)",
                        padding: "30px",
                        borderRadius: "12px",
                        width: "90%",
                        maxWidth: "500px",
                        border: "1px solid var(--border-color)",
                      }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <h3 style={{ marginTop: 0, marginBottom: "20px" }}>
                        {editingWorker ? "작업자 수정" : "작업자 추가"}
                      </h3>
                      <form
                        onSubmit={async (e) => {
                          e.preventDefault();
                          try {
                            if (
                              !workerFormData.worker_id ||
                              !workerFormData.name
                            ) {
                              showToast(
                                "작업자 ID와 이름은 필수입니다.",
                                "error"
                              );
                              return;
                            }

                            // 중복 확인 (생성 모드일 때만)
                            if (!editingWorker) {
                              const existingWorker = workersList.find(
                                (w) =>
                                  (w.workerId || w.worker_id) ===
                                  workerFormData.worker_id.trim()
                              );
                              if (existingWorker) {
                                showToast(
                                  `이미 존재하는 작업자 ID입니다: ${workerFormData.worker_id}`,
                                  "error"
                                );
                                setWorkerIdError(
                                  `이미 존재하는 작업자 ID입니다: ${workerFormData.worker_id}`
                                );
                                return;
                              }
                            }

                            let response;
                            if (editingWorker) {
                              // 수정
                              response = await api.updateWorker(
                                editingWorker.workerId,
                                {
                                  name: workerFormData.name,
                                  contact: workerFormData.contact,
                                  team: workerFormData.team,
                                  role: workerFormData.role,
                                  blood_type: workerFormData.blood_type,
                                }
                              );
                            } else {
                              // 생성
                              response = await api.createWorker({
                                worker_id: workerFormData.worker_id,
                                name: workerFormData.name,
                                contact: workerFormData.contact,
                                team: workerFormData.team,
                                role: workerFormData.role,
                                blood_type: workerFormData.blood_type,
                              });
                            }

                            if (response.success) {
                              showToast(
                                editingWorker
                                  ? "작업자가 수정되었습니다."
                                  : "작업자가 추가되었습니다.",
                                "success"
                              );
                              setShowWorkerModal(false);
                              setWorkerIdError("");
                              // 목록 새로고침
                              const workersResponse = await api.getWorkers();
                              if (
                                workersResponse.success &&
                                workersResponse.workers
                              ) {
                                setWorkersList(workersResponse.workers);
                                setTotalWorkers(workersResponse.workers.length);
                              }
                            } else {
                              // 중복 오류인 경우 특별 처리
                              if (
                                response.error &&
                                response.error.includes("이미 존재하는")
                              ) {
                                setWorkerIdError(response.error);
                              }
                              showToast(
                                `실패: ${response.error || "알 수 없는 오류"}`,
                                "error"
                              );
                            }
                          } catch (error) {
                            console.error("작업자 저장 오류:", error);
                            showToast(`오류: ${error.message}`, "error");
                          }
                        }}
                      >
                        <div style={{ marginBottom: "15px" }}>
                          <label
                            style={{
                              display: "block",
                              marginBottom: "5px",
                              fontWeight: 500,
                            }}
                          >
                            작업자 ID{" "}
                            {!editingWorker && (
                              <span style={{ color: "#DC2626" }}>*</span>
                            )}
                          </label>
                          <input
                            type="text"
                            value={workerFormData.worker_id}
                            readOnly
                            required={!editingWorker}
                            placeholder="팀을 선택시 자동입력 됩니다"
                            style={{
                              width: "100%",
                              padding: "10px",
                              borderRadius: "6px",
                              border: workerIdError
                                ? "1px solid #DC2626"
                                : "1px solid var(--border-color)",
                              background: "var(--input-disabled-bg, #1f2937)",
                              color: "var(--text-primary)",
                              fontSize: "14px",
                              cursor: "not-allowed",
                            }}
                          />
                          {workerIdError && (
                            <div
                              style={{
                                marginTop: "5px",
                                color: "#DC2626",
                                fontSize: "12px",
                              }}
                            >
                              {workerIdError}
                            </div>
                          )}
                        </div>
                        <div style={{ marginBottom: "15px" }}>
                          <label
                            style={{
                              display: "block",
                              marginBottom: "5px",
                              fontWeight: 500,
                            }}
                          >
                            이름 <span style={{ color: "#DC2626" }}>*</span>
                          </label>
                          <input
                            type="text"
                            value={workerFormData.name}
                            onCompositionStart={() => {
                              isComposingRef.current = true;
                            }}
                            onCompositionEnd={(e) => {
                              isComposingRef.current = false;
                              // 조합이 끝난 후 필터링 적용
                              let newName = e.target.value;
                              newName = newName.replace(
                                /[^가-힣a-zA-Z\s]/g,
                                ""
                              );
                              setWorkerFormData({
                                ...workerFormData,
                                name: newName,
                              });
                            }}
                            onChange={(e) => {
                              let newName = e.target.value;

                              // IME 조합 중일 때는 필터링하지 않음 (한글 입력 문제 해결)
                              if (isComposingRef.current) {
                                setWorkerFormData({
                                  ...workerFormData,
                                  name: newName,
                                });
                                return;
                              }

                              // 한글, 영문, 공백만 허용 (숫자 및 특수문자 방지)
                              newName = newName.replace(
                                /[^가-힣a-zA-Z\s]/g,
                                ""
                              );

                              setWorkerFormData({
                                ...workerFormData,
                                name: newName,
                              });
                            }}
                            required
                            style={{
                              width: "100%",
                              padding: "10px",
                              borderRadius: "6px",
                              border: "1px solid var(--border-color)",
                              background: "var(--card-bg)",
                              color: "var(--text-primary)",
                              fontSize: "14px",
                            }}
                          />
                        </div>
                        <div style={{ marginBottom: "15px" }}>
                          <label
                            style={{
                              display: "block",
                              marginBottom: "5px",
                              fontWeight: 500,
                            }}
                          >
                            연락처
                          </label>
                          <input
                            type="text"
                            value={workerFormData.contact}
                            onChange={(e) => {
                              let newContact = e.target.value;

                              // 숫자와 하이픈만 허용
                              newContact = newContact.replace(/[^0-9-]/g, "");

                              // 하이픈 자동 추가 (010-1234-5678 형식)
                              if (
                                newContact.length > 3 &&
                                newContact[3] !== "-"
                              ) {
                                newContact =
                                  newContact.slice(0, 3) +
                                  "-" +
                                  newContact.slice(3);
                              }
                              if (
                                newContact.length > 8 &&
                                newContact[8] !== "-"
                              ) {
                                newContact =
                                  newContact.slice(0, 8) +
                                  "-" +
                                  newContact.slice(8);
                              }
                              // 최대 길이 제한 (010-1234-5678 = 13자)
                              if (newContact.length > 13) {
                                newContact = newContact.slice(0, 13);
                              }

                              setWorkerFormData({
                                ...workerFormData,
                                contact: newContact,
                              });
                            }}
                            placeholder="010-1234-5678"
                            style={{
                              width: "100%",
                              padding: "10px",
                              borderRadius: "6px",
                              border: "1px solid var(--border-color)",
                              background: "var(--card-bg)",
                              color: "var(--text-primary)",
                              fontSize: "14px",
                            }}
                          />
                        </div>
                        <div style={{ marginBottom: "15px" }}>
                          <label
                            style={{
                              display: "block",
                              marginBottom: "5px",
                              fontWeight: 500,
                            }}
                          >
                            팀
                          </label>
                          <select
                            value={workerFormData.team}
                            onChange={(e) => {
                              const newTeam = e.target.value;

                              // "팀을 선택하세요" 선택 시 작업자 ID도 삭제
                              if (!newTeam || !newTeam.trim()) {
                                setWorkerFormData((prev) => ({
                                  ...prev,
                                  team: "",
                                  worker_id: "",
                                }));
                                setWorkerIdError("");
                                return;
                              }

                              // 생성 모드일 때만 자동 ID 생성
                              if (!editingWorker && newTeam && newTeam.trim()) {
                                const autoGeneratedId =
                                  generateWorkerIdByTeam(newTeam);
                                if (autoGeneratedId) {
                                  // 중복 확인
                                  const existingWorker = workersList.find(
                                    (w) =>
                                      (w.workerId || w.worker_id) ===
                                      autoGeneratedId
                                  );
                                  if (!existingWorker) {
                                    setWorkerFormData((prev) => ({
                                      ...prev,
                                      team: newTeam,
                                      worker_id: autoGeneratedId,
                                    }));
                                    setWorkerIdError("");
                                  } else {
                                    // 중복이면 다음 번호 시도
                                    const baseNumber = parseInt(
                                      autoGeneratedId,
                                      10
                                    );
                                    let nextId = baseNumber + 1;
                                    let found = false;
                                    while (
                                      nextId < baseNumber + 100 &&
                                      !found
                                    ) {
                                      const existing = workersList.find(
                                        (w) =>
                                          (w.workerId || w.worker_id) ===
                                          String(nextId)
                                      );
                                      if (!existing) {
                                        setWorkerFormData((prev) => ({
                                          ...prev,
                                          team: newTeam,
                                          worker_id: String(nextId),
                                        }));
                                        setWorkerIdError("");
                                        found = true;
                                      } else {
                                        nextId++;
                                      }
                                    }
                                    if (!found) {
                                      // 사용 가능한 ID를 찾지 못한 경우
                                      setWorkerFormData((prev) => ({
                                        ...prev,
                                        team: newTeam,
                                      }));
                                      setWorkerIdError(
                                        "해당 팀의 사용 가능한 ID가 없습니다."
                                      );
                                    }
                                  }
                                } else {
                                  setWorkerFormData((prev) => ({
                                    ...prev,
                                    team: newTeam,
                                  }));
                                }
                              } else {
                                setWorkerFormData({
                                  ...workerFormData,
                                  team: newTeam,
                                });
                              }
                            }}
                            style={{
                              width: "100%",
                              padding: "10px",
                              borderRadius: "6px",
                              border: "1px solid var(--border-color)",
                              background: "var(--card-bg)",
                              color: "var(--text-primary)",
                              fontSize: "14px",
                            }}
                          >
                            <option value="">팀을 선택하세요</option>
                            {(() => {
                              const existingTeams = Array.from(
                                new Set(
                                  workersList
                                    .map((w) => w.team)
                                    .filter(
                                      (team) => team && team.trim() !== ""
                                    )
                                )
                              );
                              const defaultTeams = ["A팀", "B팀", "C팀", "D팀"];
                              const allTeams = Array.from(
                                new Set([...existingTeams, ...defaultTeams])
                              ).sort();
                              return allTeams.map((team) => (
                                <option key={team} value={team}>
                                  {team}
                                </option>
                              ));
                            })()}
                          </select>
                        </div>
                        <div style={{ marginBottom: "15px" }}>
                          <label
                            style={{
                              display: "block",
                              marginBottom: "5px",
                              fontWeight: 500,
                            }}
                          >
                            역할
                          </label>
                          <select
                            value={workerFormData.role}
                            onChange={(e) =>
                              setWorkerFormData({
                                ...workerFormData,
                                role: e.target.value,
                              })
                            }
                            style={{
                              width: "100%",
                              padding: "10px",
                              borderRadius: "6px",
                              border: "1px solid var(--border-color)",
                              background: "var(--card-bg)",
                              color: "var(--text-primary)",
                              fontSize: "14px",
                            }}
                          >
                            <option value="worker">작업자</option>
                            <option value="manager">관리자</option>
                          </select>
                        </div>
                        <div style={{ marginBottom: "20px" }}>
                          <label
                            style={{
                              display: "block",
                              marginBottom: "5px",
                              fontWeight: 500,
                            }}
                          >
                            혈액형
                          </label>
                          <select
                            value={workerFormData.blood_type}
                            onChange={(e) =>
                              setWorkerFormData({
                                ...workerFormData,
                                blood_type: e.target.value,
                              })
                            }
                            style={{
                              width: "100%",
                              padding: "10px",
                              borderRadius: "6px",
                              border: "1px solid var(--border-color)",
                              background: "var(--card-bg)",
                              color: "var(--text-primary)",
                              fontSize: "14px",
                            }}
                          >
                            <option value="">선택 안함</option>
                            <option value="A">A</option>
                            <option value="B">B</option>
                            <option value="O">O</option>
                            <option value="AB">AB</option>
                          </select>
                        </div>

                        {/* 오늘 촬영한 이미지 표시 (수정 모드일 때만) */}
                        {editingWorker && (
                          <div
                            style={{
                              marginBottom: "20px",
                              padding: "15px",
                              background: "var(--main-bg)",
                              borderRadius: "8px",
                              border: "1px solid var(--border-color)",
                            }}
                          >
                            <h4
                              style={{
                                margin: 0,
                                fontSize: "16px",
                                marginBottom: "15px",
                              }}
                            >
                              오늘 촬영한 사진
                            </h4>
                            {loadingTodayImages ? (
                              <div
                                style={{
                                  textAlign: "center",
                                  padding: "20px",
                                  color: "var(--text-secondary)",
                                }}
                              >
                                이미지 로딩 중...
                              </div>
                            ) : todayImages.length > 0 ? (
                              <div
                                style={{
                                  display: "grid",
                                  gridTemplateColumns: "repeat(3, 1fr)",
                                  gap: "10px",
                                }}
                              >
                                {todayImages.map((img, idx) => (
                                  <div
                                    key={idx}
                                    style={{
                                      position: "relative",
                                      borderRadius: "6px",
                                      overflow: "hidden",
                                      border: "1px solid var(--border-color)",
                                    }}
                                  >
                                    {img.image_data ? (
                                      <img
                                        src={`data:image/jpeg;base64,${img.image_data}`}
                                        alt={`${img.step} 사진`}
                                        style={{
                                          width: "100%",
                                          height: "120px",
                                          objectFit: "cover",
                                        }}
                                      />
                                    ) : (
                                      <div
                                        style={{
                                          width: "100%",
                                          height: "120px",
                                          background: "var(--main-bg)",
                                          display: "flex",
                                          alignItems: "center",
                                          justifyContent: "center",
                                          color: "var(--text-secondary)",
                                          fontSize: "12px",
                                        }}
                                      >
                                        이미지 없음
                                      </div>
                                    )}
                                    <div
                                      style={{
                                        position: "absolute",
                                        bottom: 0,
                                        left: 0,
                                        right: 0,
                                        background:
                                          "linear-gradient(to top, rgba(0,0,0,0.7), transparent)",
                                        padding: "6px",
                                        fontSize: "11px",
                                        color: "white",
                                        textAlign: "center",
                                      }}
                                    >
                                      {img.step === "front"
                                        ? "정면"
                                        : img.step === "left"
                                        ? "왼쪽"
                                        : img.step === "right"
                                        ? "오른쪽"
                                        : ""}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div
                                style={{
                                  textAlign: "center",
                                  padding: "20px",
                                  color: "var(--text-secondary)",
                                  fontSize: "14px",
                                }}
                              >
                                오늘 촬영한 사진이 없습니다.
                              </div>
                            )}
                          </div>
                        )}

                        <div
                          style={{
                            display: "flex",
                            gap: "10px",
                            justifyContent: "flex-end",
                          }}
                        >
                          {editingWorker && (
                            <button
                              type="button"
                              onClick={() => {
                                showConfirm(
                                  `정말로 "${editingWorker.workerName}" 작업자를 삭제하시겠습니까?`,
                                  async () => {
                                    try {
                                      const response = await api.deleteWorker(
                                        editingWorker.workerId
                                      );
                                      if (response.success) {
                                        showToast(
                                          "작업자가 삭제되었습니다.",
                                          "success"
                                        );
                                        setShowWorkerModal(false);
                                        // 목록 새로고침
                                        const workersResponse =
                                          await api.getWorkers();
                                        if (
                                          workersResponse.success &&
                                          workersResponse.workers
                                        ) {
                                          setWorkersList(
                                            workersResponse.workers
                                          );
                                          setTotalWorkers(
                                            workersResponse.workers.length
                                          );
                                        }
                                      } else {
                                        showToast(
                                          `삭제 실패: ${
                                            response.error || "알 수 없는 오류"
                                          }`,
                                          "error"
                                        );
                                      }
                                    } catch (error) {
                                      console.error("작업자 삭제 오류:", error);
                                      showToast(
                                        `오류: ${error.message}`,
                                        "error"
                                      );
                                    }
                                  }
                                );
                              }}
                              style={{
                                padding: "10px 20px",
                                background:
                                  "linear-gradient(135deg, #DC2626 0%, #B91C1C 100%)",
                                color: "white",
                                border: "none",
                                borderRadius: "6px",
                                cursor: "pointer",
                                fontWeight: 600,
                                fontSize: "14px",
                                display: "flex",
                                alignItems: "center",
                                gap: "6px",
                                boxShadow: "0 2px 8px rgba(220, 38, 38, 0.3)",
                                transition: "all 0.2s ease",
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.transform =
                                  "translateY(-1px)";
                                e.currentTarget.style.boxShadow =
                                  "0 4px 12px rgba(220, 38, 38, 0.4)";
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.transform =
                                  "translateY(0)";
                                e.currentTarget.style.boxShadow =
                                  "0 2px 8px rgba(220, 38, 38, 0.3)";
                              }}
                            >
                              <i className="fas fa-trash"></i>
                              <span>삭제</span>
                            </button>
                          )}
                          <button
                            type="button"
                            onClick={() => setShowWorkerModal(false)}
                            style={{
                              padding: "10px 20px",
                              background: "var(--border-color)",
                              color: "var(--text-primary)",
                              border: "none",
                              borderRadius: "6px",
                              cursor: "pointer",
                              fontWeight: 500,
                            }}
                          >
                            취소
                          </button>
                          <button
                            type="submit"
                            style={{
                              padding: "10px 20px",
                              background: "#3b82f6",
                              color: "white",
                              border: "none",
                              borderRadius: "6px",
                              cursor: "pointer",
                              fontWeight: 500,
                            }}
                          >
                            {editingWorker ? "수정" : "추가"}
                          </button>
                        </div>
                      </form>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      </div>

      {/* 커스텀 확인 다이얼로그 */}
      {showConfirmDialog && confirmDialogData && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(0, 0, 0, 0.7)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 10002,
            animation: "fadeIn 0.2s ease-out",
          }}
          onClick={handleCancel}
        >
          <div
            style={{
              background: "var(--card-bg)",
              borderRadius: "12px",
              padding: "24px",
              minWidth: "400px",
              maxWidth: "500px",
              boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)",
              border: "1px solid var(--border-color)",
              animation: "fadeIn 0.3s ease-out",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              style={{
                marginBottom: "20px",
                color: "var(--text-primary)",
                fontSize: "16px",
                fontWeight: 600,
                lineHeight: "1.5",
              }}
            >
              {confirmDialogData.message}
            </div>
            <div
              style={{
                display: "flex",
                gap: "10px",
                justifyContent: "flex-end",
              }}
            >
              <button
                onClick={handleCancel}
                style={{
                  padding: "10px 20px",
                  background: "var(--border-color)",
                  color: "var(--text-primary)",
                  border: "none",
                  borderRadius: "6px",
                  cursor: "pointer",
                  fontWeight: 500,
                  fontSize: "14px",
                  transition: "all 0.2s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "var(--text-secondary)";
                  e.currentTarget.style.color = "white";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "var(--border-color)";
                  e.currentTarget.style.color = "var(--text-primary)";
                }}
              >
                취소
              </button>
              <button
                onClick={handleConfirm}
                style={{
                  padding: "10px 20px",
                  background: "var(--accent-blue)",
                  color: "white",
                  border: "none",
                  borderRadius: "6px",
                  cursor: "pointer",
                  fontWeight: 600,
                  fontSize: "14px",
                  transition: "all 0.2s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "#2563eb";
                  e.currentTarget.style.transform = "translateY(-1px)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "var(--accent-blue)";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
              >
                확인
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 커스텀 Toast 알림 */}
      {toast && (
        <div
          key={`toast-${toastExiting ? "exiting" : "entering"}`}
          onAnimationEnd={handleToastAnimationEnd}
          style={{
            position: "fixed",
            top: "20px",
            right: "20px",
            zIndex: 10001,
            minWidth: "300px",
            maxWidth: "500px",
            background: "var(--card-bg)",
            border: `1px solid ${
              toast.type === "success"
                ? "var(--accent-green)"
                : toast.type === "error"
                ? "var(--accent-red)"
                : "var(--accent-blue)"
            }`,
            borderRadius: "12px",
            padding: "16px 20px",
            boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
            display: "flex",
            alignItems: "center",
            gap: "12px",
            animation: toastExiting
              ? "slideOutToRight 0.3s ease-out forwards"
              : "slideInFromRight 0.3s ease-out",
          }}
        >
          <div
            style={{
              width: "24px",
              height: "24px",
              borderRadius: "50%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background:
                toast.type === "success"
                  ? "var(--accent-green)"
                  : toast.type === "error"
                  ? "var(--accent-red)"
                  : "var(--accent-blue)",
              color: "white",
              flexShrink: 0,
            }}
          >
            <i
              className={`fas ${
                toast.type === "success"
                  ? "fa-check"
                  : toast.type === "error"
                  ? "fa-exclamation-triangle"
                  : "fa-info-circle"
              }`}
              style={{ fontSize: "12px" }}
            ></i>
          </div>
          <div
            style={{
              flex: 1,
              color: "var(--text-primary)",
              fontSize: "14px",
              fontWeight: 500,
              lineHeight: "1.5",
            }}
          >
            {toast.message}
          </div>
          <button
            onClick={() => {
              setToastExiting(true);
            }}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--text-secondary)",
              cursor: "pointer",
              padding: "4px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "16px",
              opacity: 0.6,
              transition: "opacity 0.2s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.opacity = "1";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.opacity = "0.6";
            }}
          >
            <i className="fas fa-times"></i>
          </button>
        </div>
      )}

      {/* 이미지 모달 */}
      {showImageModal && selectedImagePath && (
        <div
          className="image-modal-overlay"
          onClick={() => {
            setShowImageModal(false);
            setSelectedImagePath(null);
          }}
        >
          <div
            className="image-modal-content"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="image-modal-close"
              onClick={() => {
                setShowImageModal(false);
                setSelectedImagePath(null);
              }}
            >
              <i className="fas fa-times"></i>
            </button>
            <img
              src={`/api/violation-image?path=${encodeURIComponent(
                selectedImagePath
              )}`}
              alt="위반 사진"
              onError={(e) => {
                console.error("이미지 로드 실패:", selectedImagePath);
                // 이미 에러 메시지가 추가되었는지 확인
                const parent = e.target.parentElement;
                if (parent && !parent.querySelector(".image-error-message")) {
                  e.target.style.display = "none";
                  const errorDiv = document.createElement("div");
                  errorDiv.className = "image-error-message";
                  errorDiv.textContent = "이미지를 불러올 수 없습니다.";
                  errorDiv.style.cssText =
                    "padding: 40px; text-align: center; color: var(--text-secondary);";
                  parent.appendChild(errorDiv);
                }
              }}
              onLoad={() => {
                // 이미지 로드 성공 시 에러 메시지 제거
                const parent = document.querySelector(".image-modal-content");
                if (parent) {
                  const errorMsg = parent.querySelector(".image-error-message");
                  if (errorMsg) {
                    errorMsg.remove();
                  }
                }
              }}
            />
          </div>
        </div>
      )}

      {/* Fonts & Icons */}
      <link
        href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap"
        rel="stylesheet"
      />
      <link
        rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
      />
    </div>
  );
}

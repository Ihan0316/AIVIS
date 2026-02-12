const API_BASE_URL = "/api";
const REQUEST_TIMEOUT = 30000; // 30초 타임아웃

// 타임아웃이 있는 fetch 래퍼
async function fetchWithTimeout(url, options = {}, timeout = REQUEST_TIMEOUT) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === "AbortError") {
      throw new Error(`요청 시간이 초과되었습니다. (${timeout / 1000}초)`);
    }
    throw error;
  }
}

// 공통 API 응답 처리 함수
async function handleApiResponse(response) {
  const text = await response.text();
  let responseData = null;

  if (text && text.trim() !== "") {
    try {
      responseData = JSON.parse(text);
    } catch {
      console.warn("응답이 JSON이 아닙니다:", text.substring(0, 100));
      responseData = { error: text };
    }
  } else {
    console.warn("서버에서 빈 응답을 받았습니다.");
    responseData = { error: "서버에서 빈 응답을 받았습니다." };
  }

  if (!response.ok) {
    const errorMessage =
      responseData?.error ||
      responseData?.message ||
      `서버 오류 (${response.status})`;
    const fullError = new Error(
      `서버 오류 (${response.status}): ${errorMessage}`
    );
    fullError.status = response.status;
    fullError.data = responseData;
    console.error("서버 오류 상세:", {
      status: response.status,
      error: errorMessage,
      data: responseData,
    });
    throw fullError;
  }

  if (!responseData) {
    throw new Error(
      "서버에서 빈 응답을 받았습니다. 백엔드 서버가 실행 중인지 확인하세요."
    );
  }

  return responseData;
}

// 공통 에러 처리 함수
function handleApiError(error) {
  if (error.name === "TypeError" && error.message.includes("fetch")) {
    throw new Error(
      "백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
    );
  }
  if (error.message.includes("요청 시간이 초과")) {
    throw error;
  }
  throw error;
}

export const api = {
  async captureFromStream(formData) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/capture-stream`,
        {
          method: "POST",
          body: formData,
        }
      );
      return await handleApiResponse(response);
    } catch (error) {
      handleApiError(error);
      throw error;
    }
  },

  async getWorkers() {
    try {
      console.log(`[API] 작업자 조회 요청: ${API_BASE_URL}/workers`);
      const response = await fetchWithTimeout(`${API_BASE_URL}/workers`, {
        method: "GET",
      });
      const data = await handleApiResponse(response);
      console.log(
        `[API] 작업자 조회 응답: success=${data.success}, count=${
          data.workers?.length || data.count || 0
        }`
      );
      if (data.workers && data.workers.length > 0) {
        console.log(`[API] 작업자 샘플:`, data.workers[0]);
      }
      return data;
    } catch (error) {
      console.error(`[API] 작업자 조회 오류:`, error);
      handleApiError(error);
      throw error;
    }
  },

  async getViolations(limit = null, status = null, days = null) {
    try {
      let url = `${API_BASE_URL}/violations`;
      // limit이 null이면 모든 데이터를 가져오기 위해 limit=0 전달 (백엔드에서 0은 모든 데이터로 처리)
      const limitValue = limit !== null ? limit : 0;
      url += `?limit=${limitValue}`;
      // days가 null이면 모든 데이터를 가져오기 위해 days=0 전달 (백엔드에서 0은 모든 데이터로 처리)
      const daysValue = days !== null ? days : 0;
      url += `&days=${daysValue}`;
      if (status) {
        url += `&status=${status}`;
      }

      const response = await fetchWithTimeout(url, {
        method: "GET",
      });
      return await handleApiResponse(response);
    } catch (error) {
      console.error(`[API] 위반 사항 조회 오류:`, error);
      handleApiError(error);
      throw error;
    }
  },

  async getViolationStats(days = null) {
    try {
      let url = `${API_BASE_URL}/violations/stats`;
      // days가 지정되면 파라미터로 전달 (주간 통계는 7주 = 49일)
      if (days !== null) {
        url += `?days=${days}`;
      }
      console.log(`[API] 통계 조회 요청: ${url}`);
      const response = await fetchWithTimeout(url, {
        method: "GET",
      });
      const data = await handleApiResponse(response);
      console.log(`[API] 통계 조회 응답:`, data);
      if (data.kpi) {
        console.log(`[API] KPI 데이터:`, data.kpi);
      }
      return data;
    } catch (error) {
      console.error(`[API] 통계 조회 오류:`, error);
      handleApiError(error);
      throw error;
    }
  },

  async getGpuUsage() {
    try {
      const response = await fetchWithTimeout(`${API_BASE_URL}/gpu`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await handleApiResponse(response);
      console.log(`[API] GPU 사용량 조회 응답:`, data);
      return data;
    } catch (error) {
      console.error(`[API] GPU 사용량 조회 오류:`, error);
      handleApiError(error);
      throw error;
    }
  },

  async createWorker(workerData) {
    try {
      const response = await fetchWithTimeout(`${API_BASE_URL}/workers`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(workerData),
      });
      return await handleApiResponse(response);
    } catch (error) {
      handleApiError(error);
      throw error;
    }
  },

  async updateWorker(workerId, workerData) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/workers/${workerId}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(workerData),
        }
      );
      return await handleApiResponse(response);
    } catch (error) {
      handleApiError(error);
      throw error;
    }
  },

  async deleteWorker(workerId) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/workers/${workerId}`,
        {
          method: "DELETE",
        }
      );
      return await handleApiResponse(response);
    } catch (error) {
      handleApiError(error);
      throw error;
    }
  },

  async getTodayImages(workerId) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/workers/${workerId}/today-images`,
        {
          method: "GET",
        }
      );
      return await handleApiResponse(response);
    } catch (error) {
      handleApiError(error);
      throw error;
    }
  },

  async updateViolationStatus(workerId, violationDatetime, status = "done") {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/violations/update-status`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            worker_id: workerId,
            violation_datetime: violationDatetime,
            status: status,
          }),
        }
      );
      return await handleApiResponse(response);
    } catch (error) {
      handleApiError(error);
      throw error;
    }
  },

  async getWeather(lat, lon) {
    try {
      const apiKey = import.meta.env.VITE_API_KEY_OPENWEATHERMAP;

      if (!apiKey) {
        throw new Error("OpenWeatherMap API 키가 설정되지 않았습니다.");
      }

      if (!lat || !lon) {
        throw new Error("위도와 경도가 필요합니다.");
      }

      // OpenWeatherMap Current Weather Data API
      // https://api.openweathermap.org/data/2.5/weather
      // 공식 문서: https://openweathermap.org/current
      // URL 형식: https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API key}&units=metric
      const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`;

      const response = await fetchWithTimeout(url, {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
      });

      // OpenWeatherMap API는 직접 호출하는 외부 API이므로 handleApiResponse 대신 직접 처리
      if (!response.ok) {
        const errorText = await response.text();
        console.error("날씨 API 오류 응답:", errorText);
        throw new Error(`날씨 API 오류 (${response.status}): ${errorText}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error("날씨 API 호출 실패:", error);
      handleApiError(error);
      throw error;
    }
  },
};

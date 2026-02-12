export const translateRiskType = (risk, language, t) => {
  if (language === "ko") return risk;
  const riskMap = {
    "안전모 미착용": t.dashboard.unhelmet,
    "안전조끼 미착용": t.dashboard.unvest,
    "넘어짐 감지": t.dashboard.fall,
  };
  return riskMap[risk] || risk;
};

// 날씨 설명을 한국어로 변환
export const translateWeatherDescription = (description) => {
  const weatherMap = {
    // Clear
    "clear sky": "맑음",
    clear: "맑음",

    // Clouds
    "few clouds": "구름 조금",
    "scattered clouds": "구름 많음",
    "broken clouds": "구름 많음",
    "overcast clouds": "흐림",
    clouds: "구름",

    // Rain
    "light rain": "가벼운 비",
    "moderate rain": "보통 비",
    "heavy intensity rain": "강한 비",
    "very heavy rain": "매우 강한 비",
    "extreme rain": "극심한 비",
    "freezing rain": "어는 비",
    "light intensity shower rain": "가벼운 소나기",
    "shower rain": "소나기",
    "heavy intensity shower rain": "강한 소나기",
    "ragged shower rain": "불규칙한 소나기",
    rain: "비",

    // Thunderstorm
    "thunderstorm with light rain": "가벼운 비를 동반한 천둥번개",
    "thunderstorm with rain": "비를 동반한 천둥번개",
    "thunderstorm with heavy rain": "강한 비를 동반한 천둥번개",
    "light thunderstorm": "약한 천둥번개",
    thunderstorm: "천둥번개",
    "heavy thunderstorm": "강한 천둥번개",
    "ragged thunderstorm": "불규칙한 천둥번개",
    "thunderstorm with light drizzle": "가벼운 이슬비를 동반한 천둥번개",
    "thunderstorm with drizzle": "이슬비를 동반한 천둥번개",
    "thunderstorm with heavy drizzle": "강한 이슬비를 동반한 천둥번개",

    // Snow
    "light snow": "가벼운 눈",
    snow: "눈",
    "heavy snow": "강한 눈",
    sleet: "진눈깨비",
    "light shower sleet": "가벼운 소나기 진눈깨비",
    "shower sleet": "소나기 진눈깨비",
    "light rain and snow": "가벼운 비와 눈",
    "rain and snow": "비와 눈",
    "light shower snow": "가벼운 소나기 눈",
    "shower snow": "소나기 눈",
    "heavy shower snow": "강한 소나기 눈",

    // Drizzle
    "light intensity drizzle": "가벼운 이슬비",
    drizzle: "이슬비",
    "heavy intensity drizzle": "강한 이슬비",
    "light intensity drizzle rain": "가벼운 이슬비 비",
    "drizzle rain": "이슬비 비",
    "heavy intensity drizzle rain": "강한 이슬비 비",
    "shower rain and drizzle": "소나기와 이슬비",
    "heavy shower rain and drizzle": "강한 소나기와 이슬비",

    // Atmosphere
    mist: "안개",
    smoke: "연기",
    haze: "실안개",
    "sand/ dust whirls": "모래/먼지 회오리",
    fog: "안개",
    sand: "모래",
    dust: "먼지",
    "volcanic ash": "화산재",
    squalls: "돌풍",
    tornado: "토네이도",
  };

  // 소문자로 변환하여 매핑
  const lowerDescription = description?.toLowerCase() || "";
  return weatherMap[lowerDescription] || description;
};

export const getWeekLabel = (weeksAgo, language, translations) => {
  if (weeksAgo === 0) return translations[language].dashboard.thisWeek;
  const today = new Date();
  const weekStart = new Date(today);
  weekStart.setDate(today.getDate() - today.getDay() - weeksAgo * 7); // 주의 시작일(일요일)
  const weekEnd = new Date(weekStart);
  weekEnd.setDate(weekStart.getDate() + 6); // 주의 종료일(토요일)

  const startMonth = weekStart.getMonth() + 1;
  const startDate = weekStart.getDate();
  const endMonth = weekEnd.getMonth() + 1;
  const endDate = weekEnd.getDate();

  if (startMonth === endMonth) {
    return `${startMonth}/${startDate}-${endDate}`;
  }
  return `${startMonth}/${startDate}-${endMonth}/${endDate}`;
};

// 얼굴 인식 영문 이름 -> 한글 이름 매핑
const FACE_NAME_MAPPING = {
  ihan: "조이한",
  junsung: "정준성",
  seungwon: "유승원",
  donghun: "강동훈",
  donghyeon: "이동현",
  dongbin: "손동빈",
  hyukjun: "양혁준",
  juheung: "김주형",
  soyeon: "윤소연",
  sunggeong: "김성경",
  taeyeun: "김태윤",
};

/**
 * 영문 이름을 한글로 매핑하는 함수
 * @param {string} name - 매핑할 이름 (영문 또는 한글)
 * @returns {string} - 매핑된 한글 이름 또는 원본 이름
 */
export const mapWorkerName = (name) => {
  if (!name || typeof name !== "string") {
    return name || "";
  }

  // 이미 한글이 포함되어 있으면 그대로 반환
  if (/[가-힣]/.test(name)) {
    return name;
  }

  // 대소문자 구분 없이 매핑 (소문자로 변환하여 검색)
  const lowerName = name.toLowerCase().trim();
  const mappedName = FACE_NAME_MAPPING[lowerName];

  return mappedName || name;
};

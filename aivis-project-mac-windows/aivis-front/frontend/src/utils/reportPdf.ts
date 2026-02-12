import { PDFDocument, rgb } from 'pdf-lib';
import fontkit from '@pdf-lib/fontkit';
import { saveAs } from 'file-saver';

/**
 * 리포트 PDF 생성 함수
 * 템플릿 PDF를 로드하고 발행일과 기간을 동적으로 채워 다운로드합니다.
 * 
 * @param startDate 시작 날짜 (Date 객체)
 * @param endDate 종료 날짜 (Date 객체)
 * @param selectedDates 선택된 날짜 배열 (Date 객체 배열)
 */
export async function generateSummaryPdf(
  startDate: Date | null,
  endDate: Date | null,
  selectedDates: Date[]
): Promise<void> {
  try {
    // 1. 템플릿 PDF 로드
    const templateResponse = await fetch('/report/aivis-report.pdf');
    if (!templateResponse.ok) {
      throw new Error('템플릿 PDF를 로드할 수 없습니다.');
    }
    const templateBytes = await templateResponse.arrayBuffer();

    // 2. PDF 문서 로드
    const pdfDoc = await PDFDocument.load(templateBytes);

    // 3. fontkit 등록
    pdfDoc.registerFontkit(fontkit);

    // 4. 한글 폰트 로드
    let font;
    let boldFont;
    try {
      const fontResponse = await fetch('/NanumGothic.ttf');
      if (!fontResponse.ok) {
        throw new Error('한글 폰트를 로드할 수 없습니다.');
      }
      const fontBytes = await fontResponse.arrayBuffer();
      font = await pdfDoc.embedFont(fontBytes);
      boldFont = font; // NanumGothic은 기본적으로 볼드가 없으므로 동일한 폰트 사용
    } catch (fontError) {
      console.warn('한글 폰트 로드 실패, 기본 폰트 사용:', fontError);
      // 기본 폰트 사용 (한글이 제대로 표시되지 않을 수 있음)
      font = await pdfDoc.embedFont('Helvetica');
      boldFont = await pdfDoc.embedFont('Helvetica-Bold');
    }

    // 5. 첫 번째 페이지 가져오기
    const pages = pdfDoc.getPages();
    if (pages.length === 0) {
      throw new Error('템플릿 PDF에 페이지가 없습니다.');
    }
    const firstPage = pages[0];
    const { width, height } = firstPage.getSize();

    // 6. 발행일 계산 (오늘 날짜)
    const today = new Date();
    const issueDate = `${today.getFullYear()}.${String(today.getMonth() + 1).padStart(2, '0')}.${String(today.getDate()).padStart(2, '0')}`;

    // 7. 기간 계산
    let periodText = '';
    if (selectedDates.length > 0) {
      // 선택된 날짜가 있는 경우
      const sortedDates = [...selectedDates].sort((a, b) => a.getTime() - b.getTime());
      const minDate = sortedDates[0];
      const maxDate = sortedDates[sortedDates.length - 1];
      const startStr = `${minDate.getFullYear()}.${String(minDate.getMonth() + 1).padStart(2, '0')}.${String(minDate.getDate()).padStart(2, '0')}`;
      const endStr = `${maxDate.getFullYear()}.${String(maxDate.getMonth() + 1).padStart(2, '0')}.${String(maxDate.getDate()).padStart(2, '0')}`;
      periodText = `${startStr} ~ ${endStr} (${selectedDates.length}일)`;
    } else if (startDate && endDate) {
      // 시작일과 종료일이 있는 경우
      const startStr = `${startDate.getFullYear()}.${String(startDate.getMonth() + 1).padStart(2, '0')}.${String(startDate.getDate()).padStart(2, '0')}`;
      const endStr = `${endDate.getFullYear()}.${String(endDate.getMonth() + 1).padStart(2, '0')}.${String(endDate.getDate()).padStart(2, '0')}`;
      const daysDiff = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)) + 1;
      periodText = `${startStr} ~ ${endStr} (${daysDiff}일)`;
    } else {
      throw new Error('선택된 기간이 없습니다.');
    }

    // 8. 좌표 계산 (A4 기준 595 x 842 포인트를 기준으로 비율 계산)
    // PDF 좌표계: (0, 0)은 왼쪽 아래 모서리
    // 발행일: X=62, Y=130 (템플릿 기준)
    const issueDateX = (62 / 595) * width;
    // Y 좌표는 아래에서 위로 계산되므로 height에서 빼야 함
    const issueDateY = height - ((111 / 842) * height);
    
    // 기간: X=33, Y=105 (템플릿 기준)
    const periodX = (33 / 595) * width;
    const periodY = height - ((137 / 842) * height);

    // 9. 텍스트 그리기
    const fontSize = 9;
    const textColor = rgb(1, 1, 1); // 흰색

    // 발행일 그리기
    firstPage.drawText(issueDate, {
      x: issueDateX,
      y: issueDateY,
      size: fontSize,
      font: boldFont,
      color: textColor,
    });

    // 기간 그리기
    firstPage.drawText(periodText, {
      x: periodX,
      y: periodY,
      size: fontSize,
      font: boldFont,
      color: textColor,
    });

    // 10. PDF 생성 및 다운로드
    const pdfBytes = await pdfDoc.save();
    const fileName = `AIVIS_리포트_${issueDate}.pdf`;
    const blob = new Blob([pdfBytes as BlobPart], { type: 'application/pdf' });
    saveAs(blob, fileName);

    console.log('[PDF 생성] 리포트 PDF 생성 완료:', fileName);
  } catch (error) {
    console.error('[PDF 생성] 오류:', error);
    alert(`PDF 생성 중 오류가 발생했습니다: ${error instanceof Error ? error.message : String(error)}`);
    throw error;
  }
}


"""
aiohttp 미들웨어
전역 에러 처리, CORS, 로깅, 인증
"""
import traceback
import time
from aiohttp import web
from typing import Callable
from .logger import get_logger

logger = get_logger('aivis.middleware')


@web.middleware
async def error_handler(request: web.Request, handler: Callable):
    """전역 에러 핸들러 미들웨어"""
    try:
        response = await handler(request)
        return response

    except web.HTTPException as e:
        # HTTP 예외는 그대로 전달
        raise

    except Exception as e:
        # 모든 예외를 로그하고 500 에러 반환
        logger.log_error(e, {
            'method': request.method,
            'path': request.path,
            'client_ip': request.remote,
            'headers': dict(request.headers)
        })

        return web.json_response({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc() if request.app.get('debug', False) else None,
            'message': '서버 오류가 발생했습니다. 관리자에게 문의하세요.'
        }, status=500)


@web.middleware
async def logging_middleware(request: web.Request, handler: Callable):
    """요청/응답 로깅 미들웨어"""
    start_time = time.time()

    # 요청 로그
    logger.log_event('http_request', {
        'method': request.method,
        'path': request.path,
        'client_ip': request.remote,
        'user_agent': request.headers.get('User-Agent', 'Unknown')
    }, 'DEBUG')

    # 핸들러 실행
    response = await handler(request)

    # 응답 로그
    duration_ms = (time.time() - start_time) * 1000
    logger.log_performance('http_request', duration_ms, {
        'method': request.method,
        'path': request.path,
        'status': response.status
    })

    return response


@web.middleware
async def cors_middleware(request: web.Request, handler: Callable):
    """CORS 미들웨어 (환경 변수 기반 도메인 제한)"""
    import os
    
    # 환경 변수에서 허용된 도메인 가져오기
    allowed_origins_str = os.getenv('ALLOWED_ORIGINS', '')
    origin = request.headers.get('Origin', '')
    
    if allowed_origins_str:
        # 프로덕션 모드: 허용된 도메인만 허용
        allowed_origins = [o.strip() for o in allowed_origins_str.split(',') if o.strip()]
        
        # Origin 헤더 확인
        if origin and origin in allowed_origins:
            allowed_origin = origin
        else:
            # 허용되지 않은 도메인은 첫 번째 허용 도메인 사용 (또는 빈 문자열)
            allowed_origin = allowed_origins[0] if allowed_origins else ''
    else:
        # 개발 모드: 모든 도메인 허용
        allowed_origin = '*'
    
    # OPTIONS 요청 (프리플라이트)
    if request.method == 'OPTIONS':
        return web.Response(
            status=200,
            headers={
                'Access-Control-Allow-Origin': allowed_origin,
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Access-Control-Max-Age': '3600'
            }
        )
    
    # 일반 요청
    response = await handler(request)
    
    # CORS 헤더 추가
    response.headers['Access-Control-Allow-Origin'] = allowed_origin
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    return response


@web.middleware
async def timeout_middleware(request: web.Request, handler: Callable):
    """요청 타임아웃 미들웨어"""
    import asyncio

    # WebSocket 요청은 타임아웃 제외
    if request.path.startswith('/ws'):
        return await handler(request)

    try:
        # 30초 타임아웃
        return await asyncio.wait_for(handler(request), timeout=30.0)
    except asyncio.TimeoutError:
        logger.log_event('timeout', {
            'method': request.method,
            'path': request.path
        }, 'WARNING')

        return web.json_response({
            'error': 'Request timeout',
            'message': '요청 처리 시간이 초과되었습니다.'
        }, status=504)


def setup_middlewares(app: web.Application):
    """미들웨어 설정"""
    app.middlewares.append(error_handler)
    app.middlewares.append(logging_middleware)
    app.middlewares.append(cors_middleware)
    app.middlewares.append(timeout_middleware)

    app['debug'] = app.get('debug', False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자체 서명 SSL 인증서 생성 스크립트
개발 환경용 HTTPS/WSS 지원을 위한 인증서 생성

사용법:
    python generate_ssl_cert.py
"""

import os
import sys
import ipaddress
from datetime import datetime, timedelta

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def generate_self_signed_cert():
    """자체 서명 인증서 생성"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        print("❌ cryptography 모듈이 설치되지 않았습니다.")
        print("설치 방법: pip install cryptography")
        return False

    # 현재 스크립트 디렉토리에 인증서 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cert_file = os.path.join(script_dir, 'cert.pem')
    key_file = os.path.join(script_dir, 'key.pem')

    # 이미 인증서가 있으면 삭제 확인
    if os.path.exists(cert_file) or os.path.exists(key_file):
        response = input("⚠️ 기존 인증서 파일이 존재합니다. 덮어쓰시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("취소되었습니다.")
            return False

    print("🔐 자체 서명 SSL 인증서 생성 중...")

    # RSA 개인 키 생성
    print("  1. RSA 개인 키 생성 중 (4096 bit)...")
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
    )

    # 인증서 주체 정보
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "KR"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Seoul"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Seoul"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AIVIS Development"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    # 인증서 생성
    print("  2. X.509 인증서 생성 중...")
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
        ]),
        critical=False,
    ).sign(private_key, hashes.SHA256())

    # 개인 키 파일 저장
    print(f"  3. 개인 키 저장 중: {key_file}")
    with open(key_file, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # 인증서 파일 저장
    print(f"  4. 인증서 저장 중: {cert_file}")
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print()
    print("✅ SSL 인증서 생성 완료!")
    print(f"   - 인증서: {cert_file}")
    print(f"   - 개인 키: {key_file}")
    print(f"   - 유효 기간: 365일")
    print()
    print("ℹ️  브라우저에서 '안전하지 않음' 경고가 나타날 수 있습니다.")
    print("   개발 환경에서는 '고급' > '계속 진행'을 클릭하여 접속하세요.")
    print()
    print("🚀 이제 백엔드 서버를 시작하면 HTTPS/WSS가 활성화됩니다.")

    return True


if __name__ == "__main__":
    try:
        success = generate_self_signed_cert()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

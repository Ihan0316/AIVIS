"""
얼굴 임베딩 시각화 스크립트
- t-SNE와 UMAP을 사용하여 512차원 임베딩을 2D로 시각화
- 인물별로 색상을 구분하여 클러스터링 품질 확인
"""

import os
import sys
import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings

# 한글 폰트 설정
import matplotlib
if sys.platform == 'win32':
    matplotlib.rc('font', family='Malgun Gothic')
elif sys.platform == 'darwin':
    matplotlib.rc('font', family='AppleGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 프로젝트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FACE_DATA_DIR = os.path.join(PROJECT_ROOT, "face", "data")
FAISS_INDEX_FILE = os.path.join(FACE_DATA_DIR, "face_index.faiss")
FAISS_LABELS_FILE = os.path.join(FACE_DATA_DIR, "face_index.faiss.labels.npy")
EMBEDDINGS_FILE = os.path.join(FACE_DATA_DIR, "embeddings", "face_embeddings.npy")


def load_embeddings_from_faiss():
    """FAISS 인덱스에서 임베딩과 레이블 로드"""
    print("\n" + "=" * 60)
    print("📊 얼굴 임베딩 시각화 도구")
    print("=" * 60)
    
    # FAISS 인덱스 로드
    if not os.path.exists(FAISS_INDEX_FILE):
        print(f"❌ FAISS 인덱스 파일을 찾을 수 없습니다: {FAISS_INDEX_FILE}")
        return None, None
    
    print(f"\n📂 FAISS 인덱스 로드 중: {FAISS_INDEX_FILE}")
    index = faiss.read_index(FAISS_INDEX_FILE)
    
    # 임베딩 추출
    n_embeddings = index.ntotal
    embedding_dim = index.d
    print(f"✅ 로드된 임베딩 수: {n_embeddings}")
    print(f"✅ 임베딩 차원: {embedding_dim}")
    
    # FAISS 인덱스에서 임베딩 벡터 추출
    embeddings = np.zeros((n_embeddings, embedding_dim), dtype=np.float32)
    for i in range(n_embeddings):
        embeddings[i] = index.reconstruct(i)
    
    # 레이블 로드
    if not os.path.exists(FAISS_LABELS_FILE):
        print(f"❌ 레이블 파일을 찾을 수 없습니다: {FAISS_LABELS_FILE}")
        labels = np.array([f"Person_{i}" for i in range(n_embeddings)])
    else:
        labels = np.load(FAISS_LABELS_FILE, allow_pickle=True)
        print(f"✅ 로드된 레이블 수: {len(labels)}")
    
    return embeddings, labels


def visualize_with_tsne(embeddings, labels, perplexity=30, n_iter=1000):
    """t-SNE를 사용한 임베딩 시각화"""
    print("\n🔄 t-SNE 변환 중...")
    
    # 임베딩 수에 따라 perplexity 조정
    n_samples = len(embeddings)
    adjusted_perplexity = min(perplexity, n_samples - 1)
    
    tsne = TSNE(
        n_components=2, 
        perplexity=adjusted_perplexity, 
        n_iter=n_iter, 
        random_state=42,
        learning_rate='auto',
        init='pca'
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print(f"✅ t-SNE 변환 완료 (perplexity={adjusted_perplexity})")
    return embeddings_2d


def visualize_with_umap(embeddings, labels, n_neighbors=15, min_dist=0.1):
    """UMAP을 사용한 임베딩 시각화 (UMAP 설치 필요)"""
    try:
        import umap
    except ImportError:
        print("⚠️  UMAP이 설치되지 않았습니다. 설치하려면: pip install umap-learn")
        return None
    
    print("\n🔄 UMAP 변환 중...")
    
    # 임베딩 수에 따라 n_neighbors 조정
    n_samples = len(embeddings)
    adjusted_n_neighbors = min(n_neighbors, n_samples - 1)
    
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=adjusted_n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    embeddings_2d = reducer.fit_transform(embeddings)
    
    print(f"✅ UMAP 변환 완료 (n_neighbors={adjusted_n_neighbors})")
    return embeddings_2d


def plot_embeddings(embeddings_2d, labels, title="임베딩 시각화", save_path=None):
    """2D 임베딩 플롯"""
    # 고유 인물 목록
    unique_labels = list(sorted(set(labels)))
    n_persons = len(unique_labels)
    
    print(f"\n👥 등록된 인물 수: {n_persons}")
    for person in unique_labels:
        count = np.sum(labels == person)
        print(f"   - {person}: {count}개 임베딩")
    
    # 색상 맵 생성
    if n_persons <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_persons]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_persons, 20)))
    
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # 플롯 생성
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 각 인물별로 포인트 그리기
    for person in unique_labels:
        mask = labels == person
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color_map[person]],
            label=f"{person} ({np.sum(mask)}개)",
            s=80,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5
        )
    
    # 각 클러스터 중심에 레이블 표시
    for person in unique_labels:
        mask = labels == person
        center_x = np.mean(embeddings_2d[mask, 0])
        center_y = np.mean(embeddings_2d[mask, 1])
        ax.annotate(
            person,
            (center_x, center_y),
            fontsize=11,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
        )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('차원 1', fontsize=12)
    ax.set_ylabel('차원 2', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n💾 이미지 저장됨: {save_path}")
    
    return fig


def compute_cluster_statistics(embeddings, labels):
    """클러스터 통계 계산"""
    print("\n📊 클러스터 통계:")
    print("-" * 50)
    
    unique_labels = list(sorted(set(labels)))
    
    stats = []
    for person in unique_labels:
        mask = labels == person
        person_embeddings = embeddings[mask]
        
        # 클러스터 중심 (평균)
        centroid = np.mean(person_embeddings, axis=0)
        
        # 중심까지의 평균 거리 (분산 측정)
        distances = np.linalg.norm(person_embeddings - centroid, axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # 클래스 내 코사인 유사도
        norms = np.linalg.norm(person_embeddings, axis=1, keepdims=True)
        normalized = person_embeddings / (norms + 1e-8)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        cos_similarities = np.dot(normalized, centroid_norm)
        avg_cos_sim = np.mean(cos_similarities)
        
        stats.append({
            'person': person,
            'count': np.sum(mask),
            'mean_dist': mean_dist,
            'std_dist': std_dist,
            'avg_cos_sim': avg_cos_sim
        })
        
        print(f"👤 {person}:")
        print(f"   - 임베딩 수: {np.sum(mask)}")
        print(f"   - 클러스터 반경 (평균): {mean_dist:.4f}")
        print(f"   - 클러스터 반경 (표준편차): {std_dist:.4f}")
        print(f"   - 중심과의 평균 코사인 유사도: {avg_cos_sim:.4f}")
        print()
    
    # 클래스 간 분리도 계산
    if len(unique_labels) > 1:
        print("\n🔍 클래스 간 분리도:")
        print("-" * 50)
        
        centroids = {}
        for person in unique_labels:
            mask = labels == person
            centroids[person] = np.mean(embeddings[mask], axis=0)
        
        for i, p1 in enumerate(unique_labels):
            for p2 in unique_labels[i+1:]:
                # 유클리드 거리
                dist = np.linalg.norm(centroids[p1] - centroids[p2])
                # 코사인 유사도
                c1_norm = centroids[p1] / (np.linalg.norm(centroids[p1]) + 1e-8)
                c2_norm = centroids[p2] / (np.linalg.norm(centroids[p2]) + 1e-8)
                cos_sim = np.dot(c1_norm, c2_norm)
                
                print(f"   {p1} ↔ {p2}:")
                print(f"      - 유클리드 거리: {dist:.4f}")
                print(f"      - 코사인 유사도: {cos_sim:.4f} (낮을수록 분리 잘됨)")
    
    return stats


def main():
    # 임베딩 및 레이블 로드
    embeddings, labels = load_embeddings_from_faiss()
    
    if embeddings is None:
        print("\n❌ 임베딩을 로드할 수 없습니다.")
        return
    
    if len(embeddings) < 3:
        print("\n❌ 시각화하기에 임베딩이 너무 적습니다 (최소 3개 필요)")
        return
    
    # 클러스터 통계 출력
    compute_cluster_statistics(embeddings, labels)
    
    # t-SNE 시각화
    embeddings_tsne = visualize_with_tsne(embeddings, labels)
    fig_tsne = plot_embeddings(
        embeddings_tsne, 
        labels, 
        title="얼굴 임베딩 시각화 (t-SNE)",
        save_path=os.path.join(PROJECT_ROOT, "embedding_visualization_tsne.png")
    )
    
    # UMAP 시각화 (설치된 경우)
    embeddings_umap = visualize_with_umap(embeddings, labels)
    if embeddings_umap is not None:
        fig_umap = plot_embeddings(
            embeddings_umap, 
            labels, 
            title="얼굴 임베딩 시각화 (UMAP)",
            save_path=os.path.join(PROJECT_ROOT, "embedding_visualization_umap.png")
        )
    
    print("\n" + "=" * 60)
    print("✅ 시각화 완료!")
    print("=" * 60)
    
    # 이미지 표시
    plt.show()


if __name__ == "__main__":
    main()


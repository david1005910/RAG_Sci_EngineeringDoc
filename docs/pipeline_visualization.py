#!/usr/bin/env python3
"""
RAG Pipeline Visualization Script
Generates step-by-step pipeline diagrams using matplotlib and graphviz

Usage:
    python pipeline_visualization.py

Output:
    - docs/images/rag_pipeline.png (static diagram)
    - docs/images/pipeline_steps/ (step-by-step images for animation)
"""

import os
from pathlib import Path

# Create output directories
OUTPUT_DIR = Path(__file__).parent / "images"
STEPS_DIR = OUTPUT_DIR / "pipeline_steps"
OUTPUT_DIR.mkdir(exist_ok=True)
STEPS_DIR.mkdir(exist_ok=True)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Install with: pip install matplotlib")

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("graphviz not installed. Install with: pip install graphviz")


# =============================================================================
# Pipeline Step Definitions
# =============================================================================

PIPELINE_STEPS = [
    {
        "id": 1,
        "name": "Data Collection",
        "name_ko": "데이터 수집",
        "description": "PubMed API에서 논문 메타데이터 수집",
        "color": "#3B82F6",  # Blue
        "icon": "📥",
        "details": [
            "PubMed E-utilities API 호출",
            "논문 제목, 초록, 저자 추출",
            "Rate Limit: 10 req/sec"
        ]
    },
    {
        "id": 2,
        "name": "Text Preprocessing",
        "name_ko": "텍스트 전처리",
        "description": "텍스트 정제 및 청킹",
        "color": "#10B981",  # Green
        "icon": "🔧",
        "details": [
            "특수문자 제거",
            "참조번호 정규화",
            "512 토큰 단위 청킹"
        ]
    },
    {
        "id": 3,
        "name": "Embedding Generation",
        "name_ko": "임베딩 생성",
        "description": "OpenAI API로 벡터 임베딩 생성",
        "color": "#8B5CF6",  # Purple
        "icon": "🧮",
        "details": [
            "text-embedding-3-small 모델",
            "1536 차원 벡터",
            "배치 처리 (100개씩)"
        ]
    },
    {
        "id": 4,
        "name": "Vector Storage",
        "name_ko": "벡터 저장",
        "description": "Qdrant에 벡터 인덱싱",
        "color": "#F59E0B",  # Amber
        "icon": "💾",
        "details": [
            "Qdrant Vector DB",
            "HNSW 인덱스",
            "메타데이터 저장"
        ]
    },
    {
        "id": 5,
        "name": "Query Processing",
        "name_ko": "쿼리 처리",
        "description": "사용자 질문 처리 및 임베딩",
        "color": "#EC4899",  # Pink
        "icon": "❓",
        "details": [
            "한글 → 영어 번역",
            "쿼리 임베딩 생성",
            "검색 파라미터 설정"
        ]
    },
    {
        "id": 6,
        "name": "Hybrid Search",
        "name_ko": "하이브리드 검색",
        "description": "Dense + Sparse 검색 융합",
        "color": "#06B6D4",  # Cyan
        "icon": "🔍",
        "details": [
            "Dense: 의미 유사도 (70%)",
            "Sparse: 키워드 매칭 (30%)",
            "Score Fusion"
        ]
    },
    {
        "id": 7,
        "name": "Context Building",
        "name_ko": "컨텍스트 구성",
        "description": "검색 결과로 프롬프트 구성",
        "color": "#EF4444",  # Red
        "icon": "📋",
        "details": [
            "Top-K 문서 선택",
            "관련성 점수 기반 정렬",
            "프롬프트 템플릿 적용"
        ]
    },
    {
        "id": 8,
        "name": "LLM Generation",
        "name_ko": "LLM 응답 생성",
        "description": "GPT-4로 답변 생성",
        "color": "#22C55E",  # Green
        "icon": "🤖",
        "details": [
            "GPT-4 API 호출",
            "컨텍스트 기반 응답",
            "출처 인용 포함"
        ]
    }
]


# =============================================================================
# Matplotlib Visualization
# =============================================================================

def create_matplotlib_pipeline(highlight_step=None, show_all=True):
    """Create pipeline diagram using matplotlib"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return None

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#0f172a')  # Dark background
    fig.patch.set_facecolor('#0f172a')

    # Title
    ax.text(8, 9.5, 'Bio-RAG Pipeline', fontsize=24, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(8, 9.0, '데이터 수집부터 AI 응답까지', fontsize=14,
            ha='center', va='center', color='#94a3b8')

    # Calculate positions for steps (2 rows of 4)
    positions = [
        (2, 7), (5, 7), (8, 7), (11, 7),     # Top row
        (2, 3), (5, 3), (8, 3), (11, 3),     # Bottom row
    ]

    box_width = 2.5
    box_height = 1.8

    for i, step in enumerate(PIPELINE_STEPS):
        if not show_all and highlight_step is not None and i >= highlight_step:
            continue

        x, y = positions[i]

        # Determine alpha based on highlight
        alpha = 1.0
        if highlight_step is not None and i != highlight_step - 1:
            alpha = 0.3 if show_all else 0.0

        # Draw box
        color = step['color']
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color if alpha == 1.0 else '#1e293b',
            edgecolor=color,
            linewidth=2,
            alpha=alpha
        )
        ax.add_patch(box)

        # Draw icon and text
        text_alpha = alpha
        ax.text(x, y + 0.5, step['icon'], fontsize=20, ha='center', va='center',
                alpha=text_alpha)
        ax.text(x, y, step['name_ko'], fontsize=11, fontweight='bold',
                ha='center', va='center', color='white', alpha=text_alpha)
        ax.text(x, y - 0.4, f"Step {step['id']}", fontsize=9,
                ha='center', va='center', color='#94a3b8', alpha=text_alpha)

        # Draw details if highlighted
        if highlight_step is not None and i == highlight_step - 1:
            details_y = y - 1.5
            for j, detail in enumerate(step['details'][:3]):
                ax.text(x, details_y - j*0.25, f"• {detail}", fontsize=8,
                        ha='center', va='center', color='#e2e8f0')

    # Draw arrows
    arrow_style = "Simple,tail_width=0.5,head_width=4,head_length=8"
    for i in range(len(PIPELINE_STEPS) - 1):
        if not show_all and highlight_step is not None and i >= highlight_step - 1:
            continue

        start_x, start_y = positions[i]
        end_x, end_y = positions[i + 1]

        alpha = 0.3 if highlight_step is not None else 0.7
        if highlight_step is not None and i == highlight_step - 1:
            alpha = 1.0

        # Determine arrow path
        if i == 3:  # From step 4 to step 5 (different row)
            # Draw curved arrow going down
            ax.annotate('', xy=(end_x - box_width/2 - 0.2, end_y),
                       xytext=(start_x + box_width/2 + 0.2, start_y),
                       arrowprops=dict(arrowstyle='->', color='#64748b',
                                      connectionstyle='arc3,rad=0.3',
                                      alpha=alpha, lw=2))
        else:
            # Horizontal arrow
            ax.annotate('', xy=(end_x - box_width/2 - 0.1, end_y),
                       xytext=(start_x + box_width/2 + 0.1, start_y),
                       arrowprops=dict(arrowstyle='->', color='#64748b',
                                      alpha=alpha, lw=2))

    # Add legend for highlighted step
    if highlight_step is not None:
        step = PIPELINE_STEPS[highlight_step - 1]
        legend_text = f"현재 단계: {step['name']} - {step['description']}"
        ax.text(8, 0.5, legend_text, fontsize=12, ha='center', va='center',
                color='white', style='italic')

    plt.tight_layout()
    return fig


def generate_step_images():
    """Generate individual images for each pipeline step"""
    if not HAS_MATPLOTLIB:
        return

    print("Generating step-by-step images...")

    # Generate full pipeline image
    fig = create_matplotlib_pipeline(highlight_step=None, show_all=True)
    if fig:
        fig.savefig(OUTPUT_DIR / "rag_pipeline_full.png", dpi=150,
                   facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: rag_pipeline_full.png")

    # Generate individual step images
    for i in range(1, len(PIPELINE_STEPS) + 1):
        fig = create_matplotlib_pipeline(highlight_step=i, show_all=True)
        if fig:
            fig.savefig(STEPS_DIR / f"step_{i:02d}.png", dpi=150,
                       facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: step_{i:02d}.png")

    print(f"Generated {len(PIPELINE_STEPS) + 1} images")


# =============================================================================
# Graphviz Visualization
# =============================================================================

def create_graphviz_pipeline():
    """Create pipeline diagram using graphviz"""
    if not HAS_GRAPHVIZ:
        print("graphviz not available")
        return None

    dot = Digraph(comment='Bio-RAG Pipeline')
    dot.attr(rankdir='TB', bgcolor='#0f172a', fontcolor='white')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica',
             fontsize='12', fontcolor='white', margin='0.3')
    dot.attr('edge', color='#64748b', penwidth='2')

    # Create subgraphs for rows
    with dot.subgraph() as s:
        s.attr(rank='same')
        for step in PIPELINE_STEPS[:4]:
            s.node(f"step{step['id']}",
                   f"{step['icon']} {step['name_ko']}\n{step['name']}",
                   fillcolor=step['color'])

    with dot.subgraph() as s:
        s.attr(rank='same')
        for step in PIPELINE_STEPS[4:]:
            s.node(f"step{step['id']}",
                   f"{step['icon']} {step['name_ko']}\n{step['name']}",
                   fillcolor=step['color'])

    # Add edges
    for i in range(len(PIPELINE_STEPS) - 1):
        dot.edge(f"step{i+1}", f"step{i+2}")

    return dot


def generate_graphviz_image():
    """Generate pipeline image using graphviz"""
    if not HAS_GRAPHVIZ:
        return

    print("Generating graphviz diagram...")
    dot = create_graphviz_pipeline()
    if dot:
        output_path = OUTPUT_DIR / "rag_pipeline_graphviz"
        dot.render(output_path, format='png', cleanup=True)
        print(f"  Saved: {output_path}.png")


# =============================================================================
# JSON Output for Frontend Animation
# =============================================================================

def generate_json_data():
    """Generate JSON data for frontend animation"""
    import json

    output_data = {
        "pipeline_name": "Bio-RAG Pipeline",
        "description": "AI 기반 바이오메디컬 연구 플랫폼 파이프라인",
        "steps": PIPELINE_STEPS,
        "connections": [
            {"from": i+1, "to": i+2} for i in range(len(PIPELINE_STEPS) - 1)
        ]
    }

    json_path = OUTPUT_DIR / "pipeline_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Generated JSON data: {json_path}")
    return output_data


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Bio-RAG Pipeline Visualization Generator")
    print("=" * 60)
    print()

    # Generate matplotlib images
    if HAS_MATPLOTLIB:
        generate_step_images()
        print()

    # Generate graphviz image
    if HAS_GRAPHVIZ:
        generate_graphviz_image()
        print()

    # Generate JSON for frontend
    generate_json_data()
    print()

    print("=" * 60)
    print("Generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

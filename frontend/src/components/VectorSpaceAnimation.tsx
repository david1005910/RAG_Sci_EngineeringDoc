import { useRef, useState, useMemo, useEffect, useCallback } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Text, OrbitControls, Line, Sphere, Html } from '@react-three/drei'
import * as THREE from 'three'
import { Play, Pause, Zap, Search, Info, Layers, Loader2, Database, AlertCircle } from 'lucide-react'
import { searchApi, vectordbApi } from '@/services/api'

// Multi-level semantic similarity data
// Level 1-6: Hierarchical relations to search term

interface WordRelation {
  word: string
  similarity: number
  category: string
  children?: WordRelation[] // Next level relations
}

interface SearchTermData {
  centerWord: string
  description: string
  relations: WordRelation[]
  isFromApi?: boolean
}

// API response types
interface PaperResult {
  pmid: string
  title: string
  keywords: string[]
  authors: string[]
  journal: string
  relevanceScore: number
}

interface VectorDBPaper {
  id?: string
  pmid: string
  title: string
  abstract?: string
  keywords?: string[]
  authors?: string[]
  journal?: string
  indexed_at?: string
}

const SEMANTIC_DATA: Record<string, SearchTermData> = {
  cancer: {
    centerWord: 'cancer',
    description: '암 관련 6단계 의미적 유사도 네트워크',
    relations: [
      {
        word: 'tumor',
        similarity: 0.95,
        category: '질환',
        children: [
          {
            word: 'malignant',
            similarity: 0.91,
            category: '특성',
            children: [
              {
                word: 'aggressive',
                similarity: 0.85,
                category: '특성',
                children: [
                  {
                    word: 'grade III',
                    similarity: 0.82,
                    category: '등급',
                    children: [
                      {
                        word: 'poorly differentiated',
                        similarity: 0.78,
                        category: '분화',
                        children: [
                          { word: 'anaplastic', similarity: 0.74, category: '형태' },
                        ],
                      },
                    ],
                  },
                ],
              },
              {
                word: 'invasive',
                similarity: 0.83,
                category: '특성',
                children: [
                  {
                    word: 'infiltrating',
                    similarity: 0.79,
                    category: '형태',
                    children: [
                      {
                        word: 'stromal invasion',
                        similarity: 0.75,
                        category: '병리',
                        children: [
                          { word: 'desmoplasia', similarity: 0.71, category: '병리' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            word: 'benign',
            similarity: 0.88,
            category: '특성',
            children: [
              {
                word: 'non-cancerous',
                similarity: 0.82,
                category: '특성',
                children: [
                  {
                    word: 'adenoma',
                    similarity: 0.78,
                    category: '종양',
                    children: [
                      {
                        word: 'polyp',
                        similarity: 0.74,
                        category: '병변',
                        children: [
                          { word: 'hyperplasia', similarity: 0.70, category: '병리' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
      {
        word: 'metastasis',
        similarity: 0.89,
        category: '진행',
        children: [
          {
            word: 'spread',
            similarity: 0.87,
            category: '과정',
            children: [
              {
                word: 'lymph node',
                similarity: 0.79,
                category: '해부',
                children: [
                  {
                    word: 'sentinel node',
                    similarity: 0.76,
                    category: '진단',
                    children: [
                      {
                        word: 'biopsy',
                        similarity: 0.72,
                        category: '검사',
                        children: [
                          { word: 'fine needle aspiration', similarity: 0.68, category: '기법' },
                        ],
                      },
                    ],
                  },
                ],
              },
              {
                word: 'distant organ',
                similarity: 0.76,
                category: '해부',
                children: [
                  {
                    word: 'liver metastasis',
                    similarity: 0.73,
                    category: '전이',
                    children: [
                      {
                        word: 'hepatectomy',
                        similarity: 0.69,
                        category: '수술',
                        children: [
                          { word: 'ablation', similarity: 0.65, category: '치료' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            word: 'invasion',
            similarity: 0.84,
            category: '과정',
            children: [
              {
                word: 'EMT',
                similarity: 0.80,
                category: '기전',
                children: [
                  {
                    word: 'E-cadherin',
                    similarity: 0.76,
                    category: '단백질',
                    children: [
                      {
                        word: 'cell adhesion',
                        similarity: 0.72,
                        category: '기능',
                        children: [
                          { word: 'tight junction', similarity: 0.68, category: '구조' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
      {
        word: 'oncogene',
        similarity: 0.85,
        category: '유전자',
        children: [
          {
            word: 'KRAS',
            similarity: 0.88,
            category: '유전자',
            children: [
              {
                word: 'G12D mutation',
                similarity: 0.82,
                category: '변이',
                children: [
                  {
                    word: 'RAS pathway',
                    similarity: 0.78,
                    category: '경로',
                    children: [
                      {
                        word: 'MEK inhibitor',
                        similarity: 0.74,
                        category: '약물',
                        children: [
                          { word: 'trametinib', similarity: 0.70, category: '약물' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            word: 'HER2',
            similarity: 0.86,
            category: '유전자',
            children: [
              {
                word: 'amplification',
                similarity: 0.82,
                category: '변이',
                children: [
                  {
                    word: 'trastuzumab',
                    similarity: 0.78,
                    category: '약물',
                    children: [
                      {
                        word: 'Herceptin',
                        similarity: 0.74,
                        category: '제품',
                        children: [
                          { word: 'biosimilar', similarity: 0.70, category: '약물' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            word: 'EGFR',
            similarity: 0.84,
            category: '유전자',
            children: [
              {
                word: 'T790M',
                similarity: 0.80,
                category: '변이',
                children: [
                  {
                    word: 'osimertinib',
                    similarity: 0.76,
                    category: '약물',
                    children: [
                      {
                        word: 'Tagrisso',
                        similarity: 0.72,
                        category: '제품',
                        children: [
                          { word: 'third-generation TKI', similarity: 0.68, category: '분류' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
      {
        word: 'chemotherapy',
        similarity: 0.78,
        category: '치료',
        children: [
          {
            word: 'cytotoxic',
            similarity: 0.85,
            category: '기전',
            children: [
              {
                word: 'cell death',
                similarity: 0.80,
                category: '효과',
                children: [
                  {
                    word: 'necrosis',
                    similarity: 0.76,
                    category: '기전',
                    children: [
                      {
                        word: 'inflammation',
                        similarity: 0.72,
                        category: '반응',
                        children: [
                          { word: 'cytokine storm', similarity: 0.68, category: '부작용' },
                        ],
                      },
                    ],
                  },
                ],
              },
              {
                word: 'apoptosis',
                similarity: 0.78,
                category: '기전',
                children: [
                  {
                    word: 'caspase',
                    similarity: 0.74,
                    category: '효소',
                    children: [
                      {
                        word: 'Bcl-2',
                        similarity: 0.70,
                        category: '단백질',
                        children: [
                          { word: 'anti-apoptotic', similarity: 0.66, category: '기능' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            word: 'drug resistance',
            similarity: 0.76,
            category: '문제',
            children: [
              {
                word: 'MDR1',
                similarity: 0.72,
                category: '유전자',
                children: [
                  {
                    word: 'P-glycoprotein',
                    similarity: 0.68,
                    category: '단백질',
                    children: [
                      {
                        word: 'efflux pump',
                        similarity: 0.64,
                        category: '기전',
                        children: [
                          { word: 'ABC transporter', similarity: 0.60, category: '분류' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
  },
  immunotherapy: {
    centerWord: 'immunotherapy',
    description: '면역치료 6단계 의미적 유사도 네트워크',
    relations: [
      {
        word: 'checkpoint inhibitor',
        similarity: 0.94,
        category: '약물',
        children: [
          {
            word: 'PD-1',
            similarity: 0.92,
            category: '타겟',
            children: [
              {
                word: 'pembrolizumab',
                similarity: 0.88,
                category: '약물',
                children: [
                  {
                    word: 'Keytruda',
                    similarity: 0.84,
                    category: '제품',
                    children: [
                      {
                        word: 'melanoma',
                        similarity: 0.80,
                        category: '적응증',
                        children: [
                          { word: 'skin cancer', similarity: 0.76, category: '질환' },
                        ],
                      },
                    ],
                  },
                ],
              },
              {
                word: 'nivolumab',
                similarity: 0.86,
                category: '약물',
                children: [
                  {
                    word: 'Opdivo',
                    similarity: 0.82,
                    category: '제품',
                    children: [
                      {
                        word: 'lung cancer',
                        similarity: 0.78,
                        category: '적응증',
                        children: [
                          { word: 'NSCLC', similarity: 0.74, category: '질환' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            word: 'CTLA-4',
            similarity: 0.89,
            category: '타겟',
            children: [
              {
                word: 'ipilimumab',
                similarity: 0.85,
                category: '약물',
                children: [
                  {
                    word: 'Yervoy',
                    similarity: 0.81,
                    category: '제품',
                    children: [
                      {
                        word: 'combination therapy',
                        similarity: 0.77,
                        category: '치료',
                        children: [
                          { word: 'nivo-ipi', similarity: 0.73, category: '병용' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
      {
        word: 'CAR-T',
        similarity: 0.88,
        category: '세포치료',
        children: [
          {
            word: 'chimeric antigen',
            similarity: 0.91,
            category: '구조',
            children: [
              {
                word: 'scFv',
                similarity: 0.84,
                category: '도메인',
                children: [
                  {
                    word: 'antigen binding',
                    similarity: 0.80,
                    category: '기능',
                    children: [
                      {
                        word: 'affinity',
                        similarity: 0.76,
                        category: '특성',
                        children: [
                          { word: 'avidity', similarity: 0.72, category: '특성' },
                        ],
                      },
                    ],
                  },
                ],
              },
              {
                word: 'costimulatory',
                similarity: 0.82,
                category: '도메인',
                children: [
                  {
                    word: 'CD28',
                    similarity: 0.78,
                    category: '단백질',
                    children: [
                      {
                        word: '4-1BB',
                        similarity: 0.74,
                        category: '단백질',
                        children: [
                          { word: 'T cell persistence', similarity: 0.70, category: '효과' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            word: 'CD19',
            similarity: 0.83,
            category: '타겟',
            children: [
              {
                word: 'B-ALL',
                similarity: 0.79,
                category: '적응증',
                children: [
                  {
                    word: 'Kymriah',
                    similarity: 0.75,
                    category: '제품',
                    children: [
                      {
                        word: 'tisagenlecleucel',
                        similarity: 0.71,
                        category: '약물',
                        children: [
                          { word: 'Novartis', similarity: 0.67, category: '제조사' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
      {
        word: 'tumor microenvironment',
        similarity: 0.82,
        category: '기전',
        children: [
          {
            word: 'immune suppression',
            similarity: 0.85,
            category: '기전',
            children: [
              {
                word: 'Treg',
                similarity: 0.79,
                category: '세포',
                children: [
                  {
                    word: 'Foxp3',
                    similarity: 0.75,
                    category: '마커',
                    children: [
                      {
                        word: 'immunosuppression',
                        similarity: 0.71,
                        category: '기전',
                        children: [
                          { word: 'TGF-beta', similarity: 0.67, category: '사이토카인' },
                        ],
                      },
                    ],
                  },
                ],
              },
              {
                word: 'MDSC',
                similarity: 0.77,
                category: '세포',
                children: [
                  {
                    word: 'myeloid',
                    similarity: 0.73,
                    category: '기원',
                    children: [
                      {
                        word: 'arginase',
                        similarity: 0.69,
                        category: '효소',
                        children: [
                          { word: 'T cell inhibition', similarity: 0.65, category: '기전' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
  },
  crispr: {
    centerWord: 'CRISPR',
    description: 'CRISPR 유전자 편집 6단계 의미적 유사도 네트워크',
    relations: [
      {
        word: 'Cas9',
        similarity: 0.96,
        category: '효소',
        children: [
          {
            word: 'nuclease',
            similarity: 0.92,
            category: '기능',
            children: [
              {
                word: 'DNA cleavage',
                similarity: 0.88,
                category: '기전',
                children: [
                  {
                    word: 'DSB repair',
                    similarity: 0.84,
                    category: '과정',
                    children: [
                      {
                        word: 'NHEJ',
                        similarity: 0.80,
                        category: '경로',
                        children: [
                          { word: 'indel formation', similarity: 0.76, category: '결과' },
                        ],
                      },
                    ],
                  },
                ],
              },
              {
                word: 'double-strand break',
                similarity: 0.86,
                category: '효과',
                children: [
                  {
                    word: 'HDR',
                    similarity: 0.82,
                    category: '경로',
                    children: [
                      {
                        word: 'template DNA',
                        similarity: 0.78,
                        category: '재료',
                        children: [
                          { word: 'precise insertion', similarity: 0.74, category: '결과' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            word: 'HNH domain',
            similarity: 0.84,
            category: '구조',
            children: [
              {
                word: 'target strand',
                similarity: 0.80,
                category: '기능',
                children: [
                  {
                    word: 'cleavage site',
                    similarity: 0.76,
                    category: '위치',
                    children: [
                      {
                        word: 'PAM proximal',
                        similarity: 0.72,
                        category: '특성',
                        children: [
                          { word: '3bp upstream', similarity: 0.68, category: '위치' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
      {
        word: 'guide RNA',
        similarity: 0.94,
        category: '구성요소',
        children: [
          {
            word: 'sgRNA',
            similarity: 0.95,
            category: '종류',
            children: [
              {
                word: 'spacer',
                similarity: 0.88,
                category: '구조',
                children: [
                  {
                    word: '20nt sequence',
                    similarity: 0.84,
                    category: '특성',
                    children: [
                      {
                        word: 'target recognition',
                        similarity: 0.80,
                        category: '기능',
                        children: [
                          { word: 'complementarity', similarity: 0.76, category: '기전' },
                        ],
                      },
                    ],
                  },
                ],
              },
              {
                word: 'scaffold',
                similarity: 0.85,
                category: '구조',
                children: [
                  {
                    word: 'Cas9 binding',
                    similarity: 0.81,
                    category: '기능',
                    children: [
                      {
                        word: 'stem loops',
                        similarity: 0.77,
                        category: '구조',
                        children: [
                          { word: 'nexus', similarity: 0.73, category: '구조' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
      {
        word: 'base editing',
        similarity: 0.82,
        category: '변형기술',
        children: [
          {
            word: 'ABE',
            similarity: 0.88,
            category: '종류',
            children: [
              {
                word: 'A-to-G',
                similarity: 0.84,
                category: '변환',
                children: [
                  {
                    word: 'adenine deaminase',
                    similarity: 0.80,
                    category: '효소',
                    children: [
                      {
                        word: 'TadA',
                        similarity: 0.76,
                        category: '효소',
                        children: [
                          { word: 'evolved variant', similarity: 0.72, category: '개선' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            word: 'prime editing',
            similarity: 0.84,
            category: '기술',
            children: [
              {
                word: 'pegRNA',
                similarity: 0.80,
                category: '구성요소',
                children: [
                  {
                    word: 'RT template',
                    similarity: 0.76,
                    category: '구조',
                    children: [
                      {
                        word: 'reverse transcriptase',
                        similarity: 0.72,
                        category: '효소',
                        children: [
                          { word: 'M-MLV', similarity: 0.68, category: '효소' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
  },
  protein: {
    centerWord: 'protein',
    description: '단백질 6단계 의미적 유사도 네트워크',
    relations: [
      {
        word: 'structure',
        similarity: 0.93,
        category: '구조',
        children: [
          {
            word: 'folding',
            similarity: 0.91,
            category: '과정',
            children: [
              { word: 'chaperone', similarity: 0.84, category: '분자' },
              { word: 'misfolding', similarity: 0.82, category: '문제' },
            ],
          },
          { word: 'alpha helix', similarity: 0.87, category: '2차구조' },
          { word: 'beta sheet', similarity: 0.85, category: '2차구조' },
        ],
      },
      {
        word: 'amino acid',
        similarity: 0.91,
        category: '구성',
        children: [
          {
            word: 'peptide bond',
            similarity: 0.89,
            category: '결합',
            children: [
              { word: 'backbone', similarity: 0.82, category: '구조' },
              { word: 'side chain', similarity: 0.80, category: '구조' },
            ],
          },
          { word: 'sequence', similarity: 0.86, category: '특성' },
          { word: 'residue', similarity: 0.84, category: '단위' },
        ],
      },
      {
        word: 'AlphaFold',
        similarity: 0.85,
        category: '예측',
        children: [
          {
            word: 'structure prediction',
            similarity: 0.92,
            category: '응용',
            children: [
              { word: 'pLDDT', similarity: 0.78, category: '지표' },
              { word: 'MSA', similarity: 0.76, category: '입력' },
            ],
          },
          { word: 'deep learning', similarity: 0.84, category: '방법' },
          { word: 'ESMFold', similarity: 0.82, category: '도구' },
        ],
      },
      {
        word: 'enzyme',
        similarity: 0.82,
        category: '기능',
        children: [
          { word: 'catalysis', similarity: 0.88, category: '기능' },
          { word: 'active site', similarity: 0.85, category: '구조' },
          { word: 'substrate', similarity: 0.82, category: '상호작용' },
        ],
      },
    ],
  },
  rna: {
    centerWord: 'RNA',
    description: 'RNA 6단계 의미적 유사도 네트워크',
    relations: [
      {
        word: 'mRNA',
        similarity: 0.95,
        category: '종류',
        children: [
          {
            word: 'translation',
            similarity: 0.91,
            category: '과정',
            children: [
              { word: 'ribosome', similarity: 0.87, category: '기구' },
              { word: 'codon', similarity: 0.85, category: '단위' },
            ],
          },
          { word: 'cap', similarity: 0.84, category: '구조' },
          { word: 'poly-A tail', similarity: 0.82, category: '구조' },
        ],
      },
      {
        word: 'transcription',
        similarity: 0.92,
        category: '과정',
        children: [
          {
            word: 'RNA polymerase',
            similarity: 0.90,
            category: '효소',
            children: [
              { word: 'promoter', similarity: 0.84, category: '서열' },
              { word: 'elongation', similarity: 0.82, category: '단계' },
            ],
          },
          { word: 'splicing', similarity: 0.86, category: '처리' },
          { word: 'pre-mRNA', similarity: 0.84, category: '중간체' },
        ],
      },
      {
        word: 'siRNA',
        similarity: 0.88,
        category: '종류',
        children: [
          {
            word: 'RNA interference',
            similarity: 0.92,
            category: '기전',
            children: [
              { word: 'RISC', similarity: 0.86, category: '복합체' },
              { word: 'Argonaute', similarity: 0.84, category: '단백질' },
            ],
          },
          { word: 'gene silencing', similarity: 0.88, category: '효과' },
          { word: 'knockdown', similarity: 0.85, category: '응용' },
        ],
      },
      {
        word: 'mRNA vaccine',
        similarity: 0.82,
        category: '응용',
        children: [
          { word: 'LNP', similarity: 0.88, category: '전달' },
          { word: 'Moderna', similarity: 0.84, category: '회사' },
          { word: 'spike protein', similarity: 0.80, category: '항원' },
        ],
      },
    ],
  },
  covid: {
    centerWord: 'COVID-19',
    description: 'COVID-19 6단계 의미적 유사도 네트워크',
    relations: [
      {
        word: 'SARS-CoV-2',
        similarity: 0.97,
        category: '바이러스',
        children: [
          {
            word: 'spike protein',
            similarity: 0.93,
            category: '구조',
            children: [
              { word: 'RBD', similarity: 0.89, category: '도메인' },
              { word: 'S1/S2', similarity: 0.86, category: '서브유닛' },
            ],
          },
          { word: 'ACE2', similarity: 0.90, category: '수용체' },
          { word: 'viral entry', similarity: 0.87, category: '과정' },
        ],
      },
      {
        word: 'variant',
        similarity: 0.89,
        category: '변이',
        children: [
          {
            word: 'Omicron',
            similarity: 0.91,
            category: '변이주',
            children: [
              { word: 'BA.5', similarity: 0.85, category: '하위변이' },
              { word: 'immune escape', similarity: 0.83, category: '특성' },
            ],
          },
          { word: 'Delta', similarity: 0.88, category: '변이주' },
          { word: 'mutation', similarity: 0.85, category: '기전' },
        ],
      },
      {
        word: 'mRNA vaccine',
        similarity: 0.86,
        category: '백신',
        children: [
          {
            word: 'Pfizer',
            similarity: 0.92,
            category: '제조사',
            children: [
              { word: 'BNT162b2', similarity: 0.88, category: '제품' },
              { word: 'efficacy', similarity: 0.82, category: '지표' },
            ],
          },
          { word: 'Moderna', similarity: 0.90, category: '제조사' },
          { word: 'booster', similarity: 0.84, category: '접종' },
        ],
      },
      {
        word: 'treatment',
        similarity: 0.78,
        category: '치료',
        children: [
          { word: 'Paxlovid', similarity: 0.86, category: '약물' },
          { word: 'remdesivir', similarity: 0.82, category: '약물' },
          { word: 'monoclonal antibody', similarity: 0.79, category: '약물' },
        ],
      },
    ],
  },
  alzheimer: {
    centerWord: 'Alzheimer',
    description: '알츠하이머 6단계 의미적 유사도 네트워크',
    relations: [
      {
        word: 'amyloid beta',
        similarity: 0.93,
        category: '병리',
        children: [
          {
            word: 'plaque',
            similarity: 0.91,
            category: '병변',
            children: [
              { word: 'aggregation', similarity: 0.86, category: '과정' },
              { word: 'oligomer', similarity: 0.84, category: '형태' },
            ],
          },
          { word: 'APP', similarity: 0.87, category: '전구체' },
          { word: 'secretase', similarity: 0.84, category: '효소' },
        ],
      },
      {
        word: 'tau',
        similarity: 0.91,
        category: '병리',
        children: [
          {
            word: 'neurofibrillary tangle',
            similarity: 0.89,
            category: '병변',
            children: [
              { word: 'hyperphosphorylation', similarity: 0.85, category: '기전' },
              { word: 'paired helical', similarity: 0.82, category: '구조' },
            ],
          },
          { word: 'microtubule', similarity: 0.84, category: '구조' },
          { word: 'spreading', similarity: 0.81, category: '과정' },
        ],
      },
      {
        word: 'neurodegeneration',
        similarity: 0.88,
        category: '병리',
        children: [
          { word: 'neuron loss', similarity: 0.86, category: '효과' },
          { word: 'synaptic dysfunction', similarity: 0.83, category: '기전' },
          { word: 'inflammation', similarity: 0.79, category: '기전' },
        ],
      },
      {
        word: 'treatment',
        similarity: 0.75,
        category: '치료',
        children: [
          { word: 'lecanemab', similarity: 0.88, category: '약물' },
          { word: 'aducanumab', similarity: 0.85, category: '약물' },
          { word: 'anti-amyloid', similarity: 0.82, category: '기전' },
        ],
      },
    ],
  },
  diabetes: {
    centerWord: 'diabetes',
    description: '당뇨병 6단계 의미적 유사도 네트워크',
    relations: [
      {
        word: 'insulin',
        similarity: 0.95,
        category: '호르몬',
        children: [
          {
            word: 'beta cell',
            similarity: 0.91,
            category: '세포',
            children: [
              { word: 'pancreas', similarity: 0.87, category: '장기' },
              { word: 'islet', similarity: 0.85, category: '구조' },
            ],
          },
          { word: 'secretion', similarity: 0.88, category: '과정' },
          { word: 'receptor', similarity: 0.84, category: '분자' },
        ],
      },
      {
        word: 'glucose',
        similarity: 0.93,
        category: '대사',
        children: [
          {
            word: 'blood sugar',
            similarity: 0.92,
            category: '지표',
            children: [
              { word: 'HbA1c', similarity: 0.88, category: '검사' },
              { word: 'fasting glucose', similarity: 0.86, category: '검사' },
            ],
          },
          { word: 'glycolysis', similarity: 0.82, category: '대사' },
          { word: 'gluconeogenesis', similarity: 0.79, category: '대사' },
        ],
      },
      {
        word: 'insulin resistance',
        similarity: 0.88,
        category: '기전',
        children: [
          {
            word: 'type 2',
            similarity: 0.91,
            category: '유형',
            children: [
              { word: 'obesity', similarity: 0.84, category: '위험인자' },
              { word: 'metabolic syndrome', similarity: 0.82, category: '관련질환' },
            ],
          },
          { word: 'GLUT4', similarity: 0.83, category: '분자' },
          { word: 'signaling', similarity: 0.80, category: '기전' },
        ],
      },
      {
        word: 'treatment',
        similarity: 0.80,
        category: '치료',
        children: [
          { word: 'metformin', similarity: 0.89, category: '약물' },
          { word: 'GLP-1 agonist', similarity: 0.86, category: '약물' },
          { word: 'SGLT2 inhibitor', similarity: 0.84, category: '약물' },
        ],
      },
    ],
  },
  'deep learning': {
    centerWord: 'deep learning',
    description: '딥러닝 바이오 응용 6단계 의미적 유사도 네트워크',
    relations: [
      {
        word: 'neural network',
        similarity: 0.95,
        category: '모델',
        children: [
          {
            word: 'transformer',
            similarity: 0.91,
            category: '아키텍처',
            children: [
              { word: 'attention', similarity: 0.89, category: '기법' },
              { word: 'self-attention', similarity: 0.87, category: '기법' },
            ],
          },
          { word: 'CNN', similarity: 0.88, category: '아키텍처' },
          { word: 'GNN', similarity: 0.85, category: '아키텍처' },
        ],
      },
      {
        word: 'AlphaFold',
        similarity: 0.92,
        category: '응용',
        children: [
          {
            word: 'protein structure',
            similarity: 0.94,
            category: '예측',
            children: [
              { word: 'folding', similarity: 0.88, category: '과정' },
              { word: 'contact map', similarity: 0.84, category: '특성' },
            ],
          },
          { word: 'ESM', similarity: 0.86, category: '모델' },
          { word: 'pLDDT', similarity: 0.82, category: '지표' },
        ],
      },
      {
        word: 'drug discovery',
        similarity: 0.85,
        category: '응용',
        children: [
          {
            word: 'molecular generation',
            similarity: 0.88,
            category: '기술',
            children: [
              { word: 'VAE', similarity: 0.82, category: '모델' },
              { word: 'diffusion', similarity: 0.80, category: '모델' },
            ],
          },
          { word: 'virtual screening', similarity: 0.84, category: '기술' },
          { word: 'QSAR', similarity: 0.81, category: '방법' },
        ],
      },
      {
        word: 'embedding',
        similarity: 0.82,
        category: '기법',
        children: [
          { word: 'representation', similarity: 0.88, category: '개념' },
          { word: 'latent space', similarity: 0.85, category: '개념' },
          { word: 'feature learning', similarity: 0.82, category: '과정' },
        ],
      },
    ],
  },
  stem_cell: {
    centerWord: 'stem cell',
    description: '줄기세포 6단계 의미적 유사도 네트워크',
    relations: [
      {
        word: 'pluripotency',
        similarity: 0.94,
        category: '특성',
        children: [
          {
            word: 'differentiation',
            similarity: 0.91,
            category: '과정',
            children: [
              { word: 'lineage', similarity: 0.86, category: '결과' },
              { word: 'commitment', similarity: 0.84, category: '단계' },
            ],
          },
          { word: 'self-renewal', similarity: 0.89, category: '특성' },
          { word: 'potency', similarity: 0.86, category: '능력' },
        ],
      },
      {
        word: 'iPSC',
        similarity: 0.92,
        category: '유형',
        children: [
          {
            word: 'reprogramming',
            similarity: 0.90,
            category: '기술',
            children: [
              { word: 'Yamanaka factors', similarity: 0.88, category: '인자' },
              { word: 'Oct4', similarity: 0.86, category: '전사인자' },
            ],
          },
          { word: 'somatic cell', similarity: 0.84, category: '기원' },
          { word: 'patient-derived', similarity: 0.82, category: '특성' },
        ],
      },
      {
        word: 'organoid',
        similarity: 0.85,
        category: '응용',
        children: [
          {
            word: '3D culture',
            similarity: 0.88,
            category: '기술',
            children: [
              { word: 'matrigel', similarity: 0.82, category: '재료' },
              { word: 'self-organization', similarity: 0.80, category: '과정' },
            ],
          },
          { word: 'mini-organ', similarity: 0.85, category: '결과' },
          { word: 'disease modeling', similarity: 0.82, category: '응용' },
        ],
      },
      {
        word: 'cell therapy',
        similarity: 0.82,
        category: '응용',
        children: [
          { word: 'regenerative medicine', similarity: 0.88, category: '분야' },
          { word: 'transplantation', similarity: 0.85, category: '방법' },
          { word: 'GMP', similarity: 0.78, category: '규제' },
        ],
      },
    ],
  },
}

const CATEGORY_COLORS: Record<string, string> = {
  질환: '#f43f5e', 진행: '#fb923c', 유전자: '#8b5cf6', 치료: '#22c55e',
  진단: '#06b6d4', 예후: '#eab308', 약물: '#ec4899', 타겟: '#8b5cf6',
  세포치료: '#06b6d4', 세포: '#f59e0b', 기전: '#a855f7', 분자: '#14b8a6',
  연구: '#6366f1', 효과: '#22c55e', 부작용: '#ef4444', 효소: '#8b5cf6',
  기술: '#06b6d4', 구성요소: '#22c55e', 수복: '#f59e0b', 문제점: '#ef4444',
  변형기술: '#ec4899', 전달: '#14b8a6', 응용: '#22c55e', 적용: '#6366f1',
  구성: '#f59e0b', 구조: '#8b5cf6', 예측: '#06b6d4', 기능: '#22c55e',
  종류: '#ec4899', 상호작용: '#a855f7', 분석: '#14b8a6', 생성: '#f59e0b',
  기구: '#6366f1', 처리: '#eab308', 바이러스: '#ef4444', 백신: '#22c55e',
  변이: '#fb923c', 후유증: '#f43f5e', 병리: '#8b5cf6', 증상: '#f59e0b',
  해부: '#14b8a6', 호르몬: '#ec4899', 대사: '#f59e0b', 지표: '#06b6d4',
  유형: '#8b5cf6', 장기: '#14b8a6', 위험인자: '#fb923c', 합병증: '#ef4444',
  모델: '#8b5cf6', 아키텍처: '#06b6d4', 기법: '#22c55e', 학습: '#f59e0b',
  개념: '#a855f7', 특성: '#8b5cf6', 과정: '#06b6d4', 전사인자: '#f59e0b',
  '2차구조': '#ec4899', 결합: '#14b8a6', 단위: '#6366f1', 방법: '#22c55e',
  도구: '#f59e0b', 입력: '#a855f7', 병변: '#f43f5e', 전구체: '#fb923c',
  도메인: '#8b5cf6', 복합체: '#14b8a6', 단백질: '#f59e0b', 검사: '#06b6d4',
  관련질환: '#ef4444', 변이주: '#fb923c', 하위변이: '#f43f5e', 제조사: '#22c55e',
  제품: '#ec4899', 접종: '#06b6d4', 서열: '#8b5cf6', 단계: '#f59e0b',
  중간체: '#a855f7', 수용체: '#14b8a6', 능력: '#22c55e', 인자: '#ec4899',
  기원: '#fb923c', 재료: '#6366f1', 결과: '#22c55e', 분야: '#8b5cf6',
  규제: '#ef4444', 문제: '#ef4444', 개선: '#22c55e', 서브유닛: '#a855f7',
  default: '#64748b',
}

const LEVEL_COLORS = ['#fbbf24', '#22c55e', '#06b6d4', '#a855f7', '#ec4899', '#f97316', '#6366f1'] // Center, L1, L2, L3, L4, L5, L6

// Similarity-based colors for API mode
const SIMILARITY_COLORS = {
  center: '#fbbf24',     // 검색어 - yellow/gold
  high: '#22c55e',       // 고유사 (>80%) - green
  medium: '#06b6d4',     // 중유사 (60-80%) - cyan
  low: '#a855f7',        // 저유사 (<60%) - purple
}

// Get color based on similarity score
function getSimilarityColor(similarity: number, isCenter: boolean = false): string {
  if (isCenter) return SIMILARITY_COLORS.center
  if (similarity >= 0.8) return SIMILARITY_COLORS.high
  if (similarity >= 0.6) return SIMILARITY_COLORS.medium
  return SIMILARITY_COLORS.low
}

// Get similarity label
function getSimilarityLabel(similarity: number): string {
  if (similarity >= 0.8) return '고유사'
  if (similarity >= 0.6) return '중유사'
  return '저유사'
}

interface WordNode {
  id: string
  text: string
  position: THREE.Vector3
  targetPosition: THREE.Vector3
  color: string
  similarity: number
  category: string
  level: number
  parentId?: string
}

interface Connection {
  from: string
  to: string
  similarity: number
  level: number
}

// Flatten hierarchical data into nodes and connections
function flattenRelations(
  data: SearchTermData,
  maxLevel: number
): { nodes: WordNode[]; connections: Connection[] } {
  const nodes: WordNode[] = []
  const connections: Connection[] = []
  const useSimilarityColors = data.isFromApi === true

  // Center node
  const centerNode: WordNode = {
    id: data.centerWord,
    text: data.centerWord,
    position: new THREE.Vector3(
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10
    ),
    targetPosition: new THREE.Vector3(0, 0, 0),
    color: getSimilarityColor(1, true),
    similarity: 1,
    category: '검색어',
    level: 0,
  }
  nodes.push(centerNode)

  // Recursive function to add nodes
  function addNodes(
    relations: WordRelation[],
    parentId: string,
    level: number,
    parentAngle: number,
    _parentDistance: number
  ) {
    if (level > maxLevel) return

    const angleStep = (Math.PI * 2) / relations.length
    const baseDistance = 1.5 + level * 1.2

    relations.forEach((rel, i) => {
      const angle = parentAngle + angleStep * i - (angleStep * (relations.length - 1)) / 2
      const distance = baseDistance + (1 - rel.similarity) * 2
      const heightVar = ((i % 3) - 1) * 0.8 * level

      const targetX = Math.cos(angle) * distance
      const targetY = heightVar
      const targetZ = Math.sin(angle) * distance

      const nodeId = `${rel.word}-${level}`
      // Use similarity-based colors for API mode, category colors otherwise
      const color = useSimilarityColors
        ? getSimilarityColor(rel.similarity)
        : (CATEGORY_COLORS[rel.category] || LEVEL_COLORS[Math.min(level, 6)])

      // Update category to similarity label for API mode
      const category = useSimilarityColors
        ? getSimilarityLabel(rel.similarity)
        : rel.category

      nodes.push({
        id: nodeId,
        text: rel.word,
        position: new THREE.Vector3(
          (Math.random() - 0.5) * 12,
          (Math.random() - 0.5) * 12,
          (Math.random() - 0.5) * 12
        ),
        targetPosition: new THREE.Vector3(targetX, targetY, targetZ),
        color,
        similarity: rel.similarity,
        category,
        level,
        parentId,
      })

      connections.push({
        from: parentId,
        to: nodeId,
        similarity: rel.similarity,
        level,
      })

      // Recurse for children
      if (rel.children && rel.children.length > 0) {
        addNodes(rel.children, nodeId, level + 1, angle, distance)
      }
    })
  }

  addNodes(data.relations, data.centerWord, 1, 0, 0)

  return { nodes, connections }
}

// 3D Word Node Component
function WordNode3D({
  node,
  isHovered,
  onHover,
  isPlaying,
  showSimilarity,
  visibleLevel,
}: {
  node: WordNode
  isHovered: boolean
  onHover: (id: string | null) => void
  isPlaying: boolean
  showSimilarity: boolean
  visibleLevel: number
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const textRef = useRef<THREE.Mesh>(null)

  const isVisible = node.level <= visibleLevel
  const baseSize = node.level === 0 ? 0.35 : 0.12 + (1 - node.level * 0.15) * 0.15

  useFrame(() => {
    if (!meshRef.current || !isPlaying) return

    const current = meshRef.current.position
    current.lerp(node.targetPosition, 0.025)

    if (textRef.current) {
      textRef.current.position.copy(current)
      textRef.current.position.y += baseSize + 0.12
    }

    node.position.copy(current)
  })

  if (!isVisible) return null

  return (
    <group>
      <Sphere
        ref={meshRef}
        args={[isHovered ? baseSize * 1.4 : baseSize, 24, 24]}
        position={node.position}
        onPointerOver={() => onHover(node.id)}
        onPointerOut={() => onHover(null)}
      >
        <meshStandardMaterial
          color={node.color}
          emissive={node.color}
          emissiveIntensity={isHovered ? 0.7 : node.level === 0 ? 0.5 : 0.25}
          transparent
          opacity={isHovered ? 1 : 0.9 - node.level * 0.1}
        />
      </Sphere>

      <Text
        ref={textRef}
        position={[node.position.x, node.position.y + baseSize + 0.12, node.position.z]}
        fontSize={node.level === 0 ? 0.24 : isHovered ? 0.18 : 0.13 - node.level * 0.01}
        color="white"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.02}
        outlineColor="#000000"
      >
        {node.text}
      </Text>

      {isHovered && node.level > 0 && showSimilarity && (
        <Html position={[node.position.x, node.position.y - baseSize - 0.25, node.position.z]} center>
          <div className="px-2 py-1 bg-black/90 rounded text-xs whitespace-nowrap border border-cyan-400/50">
            <span className="text-cyan-400">{(node.similarity * 100).toFixed(0)}%</span>
            <span className="text-white/50 ml-1">L{node.level}</span>
          </div>
        </Html>
      )}
    </group>
  )
}

// Connection Lines
function ConnectionLines({
  connections,
  nodes,
  hoveredWord,
  visibleLevel,
}: {
  connections: Connection[]
  nodes: WordNode[]
  hoveredWord: string | null
  visibleLevel: number
}) {
  return (
    <>
      {connections.map((conn, i) => {
        if (conn.level > visibleLevel) return null

        const fromNode = nodes.find((n) => n.id === conn.from)
        const toNode = nodes.find((n) => n.id === conn.to)
        if (!fromNode || !toNode) return null

        const isHighlighted = hoveredWord === conn.from || hoveredWord === conn.to
        const opacity = isHighlighted ? 0.9 : 0.15 + conn.similarity * 0.25
        const lineWidth = isHighlighted ? 2.5 : 0.8 + conn.similarity * 1.2

        const hue = conn.similarity * 180
        const color = isHighlighted ? '#06b6d4' : `hsl(${hue}, 65%, 55%)`

        return (
          <group key={i}>
            <Line
              points={[fromNode.position, toNode.position]}
              color={color}
              lineWidth={lineWidth}
              transparent
              opacity={opacity}
            />
            {isHighlighted && (
              <Html
                position={[
                  (fromNode.position.x + toNode.position.x) / 2,
                  (fromNode.position.y + toNode.position.y) / 2 + 0.15,
                  (fromNode.position.z + toNode.position.z) / 2,
                ]}
                center
              >
                <div className="px-2 py-0.5 bg-cyan-500/90 rounded text-[10px] text-white font-bold">
                  {(conn.similarity * 100).toFixed(0)}%
                </div>
              </Html>
            )}
          </group>
        )
      })}
    </>
  )
}

function Axes() {
  return (
    <group>
      <Line points={[[-5, 0, 0], [5, 0, 0]]} color="#ff6b6b" lineWidth={1} transparent opacity={0.15} />
      <Line points={[[0, -5, 0], [0, 5, 0]]} color="#51cf66" lineWidth={1} transparent opacity={0.15} />
      <Line points={[[0, 0, -5], [0, 0, 5]]} color="#339af0" lineWidth={1} transparent opacity={0.15} />
      <gridHelper args={[10, 10, '#333333', '#222222']} />
    </group>
  )
}

function Scene({
  nodes,
  connections,
  hoveredWord,
  setHoveredWord,
  isPlaying,
  showConnections,
  showSimilarity,
  visibleLevel,
}: {
  nodes: WordNode[]
  connections: Connection[]
  hoveredWord: string | null
  setHoveredWord: (id: string | null) => void
  isPlaying: boolean
  showConnections: boolean
  showSimilarity: boolean
  visibleLevel: number
}) {
  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} />

      <Axes />

      {showConnections && (
        <ConnectionLines
          connections={connections}
          nodes={nodes}
          hoveredWord={hoveredWord}
          visibleLevel={visibleLevel}
        />
      )}

      {nodes.map((node) => (
        <WordNode3D
          key={node.id}
          node={node}
          isHovered={hoveredWord === node.id}
          onHover={setHoveredWord}
          isPlaying={isPlaying}
          showSimilarity={showSimilarity}
          visibleLevel={visibleLevel}
        />
      ))}

      <OrbitControls enableDamping dampingFactor={0.05} rotateSpeed={0.5} minDistance={3} maxDistance={18} />
    </>
  )
}

// Biomedical semantic knowledge base for generating related terms
const BIOMEDICAL_ONTOLOGY: Record<string, { related: string[]; synonyms: string[]; broader: string[]; narrower: string[] }> = {
  // Diseases & Conditions
  cancer: { related: ['tumor', 'oncology', 'metastasis', 'carcinoma', 'malignancy'], synonyms: ['neoplasm', 'malignant tumor'], broader: ['disease', 'pathology'], narrower: ['lung cancer', 'breast cancer', 'leukemia', 'lymphoma'] },
  tumor: { related: ['cancer', 'growth', 'mass', 'lesion'], synonyms: ['neoplasm', 'tumour'], broader: ['pathology'], narrower: ['benign tumor', 'malignant tumor'] },
  diabetes: { related: ['insulin', 'glucose', 'metabolism', 'hyperglycemia'], synonyms: ['diabetes mellitus'], broader: ['metabolic disease'], narrower: ['type 1 diabetes', 'type 2 diabetes', 'gestational diabetes'] },
  alzheimer: { related: ['dementia', 'neurodegeneration', 'amyloid', 'tau', 'cognitive'], synonyms: ['Alzheimer disease', 'AD'], broader: ['neurodegenerative disease'], narrower: ['early-onset AD', 'late-onset AD'] },
  covid: { related: ['SARS-CoV-2', 'coronavirus', 'pandemic', 'vaccine', 'infection'], synonyms: ['COVID-19', 'coronavirus disease'], broader: ['viral infection'], narrower: ['long COVID', 'acute COVID'] },
  infection: { related: ['pathogen', 'bacteria', 'virus', 'immune response'], synonyms: ['infectious disease'], broader: ['disease'], narrower: ['viral infection', 'bacterial infection'] },
  inflammation: { related: ['immune response', 'cytokine', 'swelling', 'pain'], synonyms: ['inflammatory response'], broader: ['immune response'], narrower: ['acute inflammation', 'chronic inflammation'] },

  // Technologies & Methods
  crispr: { related: ['gene editing', 'Cas9', 'guide RNA', 'genome'], synonyms: ['CRISPR-Cas9', 'CRISPR system'], broader: ['gene editing technology'], narrower: ['base editing', 'prime editing', 'CRISPRi'] },
  sequencing: { related: ['DNA', 'RNA', 'genome', 'NGS', 'reads'], synonyms: ['DNA sequencing'], broader: ['genomics'], narrower: ['RNA-seq', 'whole genome sequencing', 'single-cell sequencing'] },
  pcr: { related: ['amplification', 'DNA', 'primer', 'thermal cycling'], synonyms: ['polymerase chain reaction'], broader: ['molecular biology technique'], narrower: ['qPCR', 'RT-PCR', 'digital PCR'] },
  microscopy: { related: ['imaging', 'visualization', 'magnification', 'optics'], synonyms: ['microscopic imaging'], broader: ['imaging technique'], narrower: ['electron microscopy', 'fluorescence microscopy', 'confocal microscopy'] },

  // Molecules & Proteins
  protein: { related: ['amino acid', 'enzyme', 'structure', 'folding'], synonyms: ['polypeptide'], broader: ['biomolecule'], narrower: ['enzyme', 'antibody', 'receptor', 'transcription factor'] },
  dna: { related: ['gene', 'genome', 'nucleotide', 'replication'], synonyms: ['deoxyribonucleic acid'], broader: ['nucleic acid'], narrower: ['genomic DNA', 'mitochondrial DNA', 'cDNA'] },
  rna: { related: ['transcription', 'gene expression', 'nucleotide'], synonyms: ['ribonucleic acid'], broader: ['nucleic acid'], narrower: ['mRNA', 'tRNA', 'rRNA', 'miRNA', 'siRNA'] },
  mrna: { related: ['transcription', 'translation', 'protein synthesis', 'vaccine'], synonyms: ['messenger RNA'], broader: ['RNA'], narrower: ['mature mRNA', 'pre-mRNA'] },
  antibody: { related: ['antigen', 'immune response', 'B cell', 'immunoglobulin'], synonyms: ['immunoglobulin', 'Ab'], broader: ['protein'], narrower: ['IgG', 'IgM', 'IgA', 'monoclonal antibody'] },
  enzyme: { related: ['catalysis', 'substrate', 'reaction', 'kinetics'], synonyms: ['biocatalyst'], broader: ['protein'], narrower: ['kinase', 'protease', 'polymerase', 'ligase'] },

  // Cells & Biology
  cell: { related: ['nucleus', 'membrane', 'organelle', 'division'], synonyms: ['cellular'], broader: ['biology'], narrower: ['stem cell', 'T cell', 'neuron', 'epithelial cell'] },
  'stem cell': { related: ['differentiation', 'pluripotency', 'regeneration'], synonyms: ['progenitor cell'], broader: ['cell'], narrower: ['ESC', 'iPSC', 'adult stem cell', 'HSC'] },
  neuron: { related: ['synapse', 'axon', 'dendrite', 'neurotransmitter'], synonyms: ['nerve cell'], broader: ['cell'], narrower: ['motor neuron', 'sensory neuron', 'interneuron'] },

  // Treatments & Therapies
  immunotherapy: { related: ['immune system', 'checkpoint', 'T cell', 'cancer'], synonyms: ['immune therapy'], broader: ['therapy'], narrower: ['CAR-T', 'checkpoint inhibitor', 'cancer vaccine'] },
  chemotherapy: { related: ['cancer', 'cytotoxic', 'drug', 'treatment'], synonyms: ['chemo'], broader: ['cancer treatment'], narrower: ['adjuvant chemotherapy', 'neoadjuvant chemotherapy'] },
  vaccine: { related: ['immunization', 'antigen', 'antibody', 'immunity'], synonyms: ['vaccination'], broader: ['preventive medicine'], narrower: ['mRNA vaccine', 'viral vector vaccine', 'subunit vaccine'] },
  therapy: { related: ['treatment', 'intervention', 'cure', 'medicine'], synonyms: ['treatment'], broader: ['medicine'], narrower: ['gene therapy', 'cell therapy', 'radiation therapy'] },
  drug: { related: ['pharmaceutical', 'compound', 'molecule', 'target'], synonyms: ['medication', 'pharmaceutical'], broader: ['treatment'], narrower: ['small molecule', 'biologic', 'antibody drug'] },

  // AI & Computational
  'deep learning': { related: ['neural network', 'machine learning', 'AI', 'training'], synonyms: ['DL'], broader: ['machine learning'], narrower: ['CNN', 'RNN', 'transformer', 'GAN'] },
  'machine learning': { related: ['algorithm', 'prediction', 'model', 'data'], synonyms: ['ML'], broader: ['artificial intelligence'], narrower: ['supervised learning', 'unsupervised learning', 'reinforcement learning'] },
  ai: { related: ['algorithm', 'automation', 'intelligence', 'prediction'], synonyms: ['artificial intelligence'], broader: ['computer science'], narrower: ['machine learning', 'deep learning', 'NLP'] },
  bioinformatics: { related: ['computational biology', 'sequence analysis', 'genomics'], synonyms: ['computational biology'], broader: ['data science'], narrower: ['sequence alignment', 'structural bioinformatics', 'systems biology'] },

  // General biomedical terms
  gene: { related: ['DNA', 'expression', 'mutation', 'heredity'], synonyms: ['genetic element'], broader: ['genome'], narrower: ['oncogene', 'tumor suppressor', 'housekeeping gene'] },
  mutation: { related: ['variant', 'polymorphism', 'DNA change', 'genetic'], synonyms: ['genetic variant'], broader: ['genetic change'], narrower: ['point mutation', 'deletion', 'insertion', 'SNP'] },
  expression: { related: ['transcription', 'gene', 'protein', 'regulation'], synonyms: ['gene expression'], broader: ['molecular biology'], narrower: ['overexpression', 'downregulation', 'differential expression'] },
  pathway: { related: ['signaling', 'metabolism', 'cascade', 'regulation'], synonyms: ['biological pathway'], broader: ['systems biology'], narrower: ['metabolic pathway', 'signaling pathway', 'regulatory pathway'] },
  receptor: { related: ['ligand', 'binding', 'signal', 'membrane'], synonyms: ['cellular receptor'], broader: ['protein'], narrower: ['GPCR', 'tyrosine kinase receptor', 'nuclear receptor'] },
  biomarker: { related: ['diagnosis', 'prognosis', 'detection', 'marker'], synonyms: ['biological marker'], broader: ['diagnostic'], narrower: ['protein biomarker', 'genetic biomarker', 'imaging biomarker'] },
}

// Generate semantic network from query using ontology
function generateSemanticNetwork(query: string): SearchTermData {
  const queryLower = query.toLowerCase().trim()
  const words = queryLower.split(/\s+/)

  const relations: WordRelation[] = []
  const addedWords = new Set<string>()

  // Find matching ontology entries
  let matchedEntries: Array<{ term: string; data: typeof BIOMEDICAL_ONTOLOGY[string]; matchScore: number }> = []

  for (const [term, data] of Object.entries(BIOMEDICAL_ONTOLOGY)) {
    // Exact match
    if (queryLower === term || queryLower.includes(term) || term.includes(queryLower)) {
      matchedEntries.push({ term, data, matchScore: queryLower === term ? 1.0 : 0.9 })
    }
    // Word match
    else if (words.some(w => term.includes(w) || w.includes(term))) {
      matchedEntries.push({ term, data, matchScore: 0.7 })
    }
    // Synonym match
    else if (data.synonyms.some(s => s.toLowerCase().includes(queryLower) || queryLower.includes(s.toLowerCase()))) {
      matchedEntries.push({ term, data, matchScore: 0.85 })
    }
  }

  // Sort by match score
  matchedEntries = matchedEntries.sort((a, b) => b.matchScore - a.matchScore).slice(0, 3)

  // If no matches, create basic network from query words
  if (matchedEntries.length === 0) {
    // Create nodes from query words
    words.forEach((word, idx) => {
      if (word.length > 2 && !addedWords.has(word)) {
        addedWords.add(word)
        relations.push({
          word: word,
          similarity: 0.85 - idx * 0.05,
          category: '검색어',
          children: [
            { word: `${word} research`, similarity: 0.75, category: '연구' },
            { word: `${word} analysis`, similarity: 0.72, category: '분석' },
            { word: `${word} study`, similarity: 0.70, category: '연구' },
          ]
        })
      }
    })

    return {
      centerWord: query,
      description: `"${query}" 의미적 유사어 네트워크 (일반 검색)`,
      relations,
      isFromApi: true
    }
  }

  // Build network from matched ontology entries
  matchedEntries.forEach((entry, entryIdx) => {
    const { term, data, matchScore } = entry

    // Level 1: Related terms
    const level1Children: WordRelation[] = []

    // Add related terms as Level 2
    data.related.slice(0, 4).forEach((related, idx) => {
      if (!addedWords.has(related)) {
        addedWords.add(related)

        // Level 3: Get sub-relations if available
        const subOntology = BIOMEDICAL_ONTOLOGY[related.toLowerCase()]
        const level2Children: WordRelation[] = []

        if (subOntology) {
          subOntology.related.slice(0, 3).forEach((subRelated, subIdx) => {
            if (!addedWords.has(subRelated)) {
              addedWords.add(subRelated)

              // Level 4: Even deeper relations
              const deepOntology = BIOMEDICAL_ONTOLOGY[subRelated.toLowerCase()]
              const level3Children: WordRelation[] = []

              if (deepOntology) {
                deepOntology.related.slice(0, 2).forEach((deepRelated, deepIdx) => {
                  if (!addedWords.has(deepRelated)) {
                    addedWords.add(deepRelated)
                    level3Children.push({
                      word: deepRelated,
                      similarity: 0.55 - deepIdx * 0.05,
                      category: '확장연관',
                      children: deepOntology.narrower?.slice(0, 2).map((n, ni) => ({
                        word: n,
                        similarity: 0.45 - ni * 0.05,
                        category: '세부개념'
                      }))
                    })
                  }
                })
              }

              level2Children.push({
                word: subRelated,
                similarity: 0.68 - subIdx * 0.04,
                category: '연관개념',
                children: level3Children.length > 0 ? level3Children : undefined
              })
            }
          })
        }

        level1Children.push({
          word: related,
          similarity: 0.82 - idx * 0.04,
          category: '관련어',
          children: level2Children.length > 0 ? level2Children : undefined
        })
      }
    })

    // Add synonyms
    data.synonyms.slice(0, 2).forEach((syn, idx) => {
      if (!addedWords.has(syn)) {
        addedWords.add(syn)
        level1Children.push({
          word: syn,
          similarity: 0.92 - idx * 0.03,
          category: '동의어'
        })
      }
    })

    // Add narrower terms with their own children
    data.narrower.slice(0, 3).forEach((narrow, idx) => {
      if (!addedWords.has(narrow)) {
        addedWords.add(narrow)
        const narrowOntology = BIOMEDICAL_ONTOLOGY[narrow.toLowerCase()]
        const narrowChildren: WordRelation[] = []

        if (narrowOntology) {
          narrowOntology.related.slice(0, 2).forEach((nr, nrIdx) => {
            if (!addedWords.has(nr)) {
              addedWords.add(nr)
              narrowChildren.push({
                word: nr,
                similarity: 0.65 - nrIdx * 0.05,
                category: '세부관련'
              })
            }
          })
        }

        level1Children.push({
          word: narrow,
          similarity: 0.78 - idx * 0.05,
          category: '하위개념',
          children: narrowChildren.length > 0 ? narrowChildren : undefined
        })
      }
    })

    // Main entry node
    if (entryIdx === 0 && term.toLowerCase() !== queryLower) {
      // If the matched term is different from query, add it as a related concept
      relations.push({
        word: term,
        similarity: matchScore,
        category: '핵심개념',
        children: level1Children
      })
    } else if (entryIdx === 0) {
      // Direct children for main query
      relations.push(...level1Children)
    } else {
      // Additional matched entries
      relations.push({
        word: term,
        similarity: matchScore - entryIdx * 0.1,
        category: '관련개념',
        children: level1Children.slice(0, 3)
      })
    }

    // Add broader terms
    data.broader.slice(0, 2).forEach((broad, idx) => {
      if (!addedWords.has(broad)) {
        addedWords.add(broad)
        relations.push({
          word: broad,
          similarity: 0.70 - idx * 0.05,
          category: '상위개념'
        })
      }
    })
  })

  return {
    centerWord: query,
    description: `"${query}" 의미적 유사어 네트워크 (${relations.length}개 연관어)`,
    relations,
    isFromApi: true
  }
}

// Function to extract key terms from text chunk
function extractKeyTerms(text: string): string[] {
  if (!text || text.length === 0) return []

  // Biomedical stopwords to filter out
  const stopwords = new Set([
    'the', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
    'was', 'were', 'been', 'have', 'has', 'had', 'is', 'are', 'be', 'this', 'that',
    'which', 'who', 'what', 'where', 'when', 'how', 'why', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very',
    'can', 'will', 'just', 'may', 'should', 'could', 'would', 'might', 'must',
    'a', 'an', 'as', 'but', 'if', 'not', 'only', 'same', 'so', 'also', 'into',
    'study', 'studies', 'results', 'conclusion', 'methods', 'background', 'objective',
    'using', 'used', 'use', 'based', 'showed', 'found', 'suggest', 'suggests',
    'included', 'including', 'associated', 'significant', 'significantly'
  ])

  // Biomedical terms patterns to prioritize
  const biomedicalPatterns = /\b(gene|protein|receptor|enzyme|mutation|variant|pathway|cell|tumor|cancer|therapy|treatment|drug|inhibitor|antibody|biomarker|expression|level|activity|patient|disease|syndrome|disorder|infection|virus|bacteria|immune|inflammatory|clinical|trial|outcome|survival|response|resistance|diagnosis|prognosis)\b/gi

  // Extract words, prioritizing biomedical terms
  const words = text
    .toLowerCase()
    .replace(/[^\w\s-]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 3 && !stopwords.has(word))

  // Count word frequency
  const wordCount = new Map<string, number>()
  words.forEach(word => {
    wordCount.set(word, (wordCount.get(word) || 0) + 1)
  })

  // Find biomedical terms in original text
  const bioTerms = new Set<string>()
  let match
  while ((match = biomedicalPatterns.exec(text)) !== null) {
    bioTerms.add(match[0].toLowerCase())
  }

  // Sort by frequency and biomedical relevance
  const sortedWords = Array.from(wordCount.entries())
    .sort((a, b) => {
      const aIsBio = bioTerms.has(a[0]) ? 10 : 0
      const bIsBio = bioTerms.has(b[0]) ? 10 : 0
      return (b[1] + bIsBio) - (a[1] + aIsBio)
    })
    .map(([word]) => word)

  return sortedWords.slice(0, 5)
}

// Function to build network data from VectorDB and API search results
async function buildNetworkFromSearch(query: string): Promise<SearchTermData | null> {
  try {
    const relations: WordRelation[] = []
    const addedWords = new Set<string>()
    let vectorDbUsed = false
    let paperCount = 0

    // 1. PRIMARY: Try VectorDB search first (indexed papers with embeddings)
    try {
      console.log('Searching VectorDB for:', query)
      const vectorResponse = await vectordbApi.search(query, 10, 'hybrid', 0.7)
      const vectorResults = vectorResponse.results

      if (vectorResults && vectorResults.length > 0) {
        vectorDbUsed = true
        paperCount = vectorResults.length
        console.log(`Found ${vectorResults.length} results in VectorDB`)

        // Get full paper metadata for richer information
        let papersMetadata: VectorDBPaper[] = []
        try {
          const metadataResponse = await vectordbApi.getMetadata()
          papersMetadata = metadataResponse.papers || []
        } catch {
          console.log('Could not get paper metadata')
        }

        // Build network from vector search results
        vectorResults.slice(0, 6).forEach((result, idx) => {
          const paperTitle = result.title.length > 45
            ? result.title.substring(0, 45) + '...'
            : result.title

          if (addedWords.has(paperTitle.toLowerCase())) return
          addedWords.add(paperTitle.toLowerCase())

          // Find full paper metadata
          const paperMeta = papersMetadata.find(p => p.pmid === result.pmid)
          const keywords = paperMeta?.keywords || []

          // Extract key terms from text chunk
          const textTerms = extractKeyTerms(result.text)

          // Level 2: Keywords and text terms
          const level2Children: WordRelation[] = []

          // Add keywords
          keywords.slice(0, 3).forEach((kw, kwIdx) => {
            if (!addedWords.has(kw.toLowerCase())) {
              addedWords.add(kw.toLowerCase())

              // Level 3: Get ontology connections for keywords
              const ontologyData = BIOMEDICAL_ONTOLOGY[kw.toLowerCase()]
              const level3Children: WordRelation[] = []

              if (ontologyData) {
                ontologyData.related.slice(0, 2).forEach((rel, relIdx) => {
                  if (!addedWords.has(rel.toLowerCase())) {
                    addedWords.add(rel.toLowerCase())
                    level3Children.push({
                      word: rel,
                      similarity: 0.65 - relIdx * 0.05,
                      category: '연관개념'
                    })
                  }
                })
              }

              level2Children.push({
                word: kw,
                similarity: 0.78 - kwIdx * 0.04,
                category: '키워드',
                children: level3Children.length > 0 ? level3Children : undefined
              })
            }
          })

          // Add extracted text terms
          textTerms.slice(0, 3).forEach((term, tIdx) => {
            if (!addedWords.has(term.toLowerCase())) {
              addedWords.add(term.toLowerCase())
              level2Children.push({
                word: term,
                similarity: 0.70 - tIdx * 0.05,
                category: '추출어'
              })
            }
          })

          // Calculate display score (normalize to 0-1 if needed)
          const displayScore = result.score > 1 ? result.score / 100 : result.score

          relations.push({
            word: paperTitle,
            similarity: Math.min(0.98, displayScore + 0.1 - idx * 0.03),
            category: '벡터DB논문',
            children: level2Children.length > 0 ? level2Children : undefined
          })

          // Add score info as metadata
          if (result.dense_score !== undefined && result.sparse_score !== undefined) {
            const scoreInfo = `Dense: ${(result.dense_score * 100).toFixed(0)}% / Sparse: ${(result.sparse_score * 100).toFixed(0)}%`
            if (!addedWords.has(scoreInfo)) {
              addedWords.add(scoreInfo)
              // Add as child of paper showing score breakdown
            }
          }
        })
      }
    } catch (err) {
      console.log('VectorDB search failed:', err)
    }

    // 2. SECONDARY: If VectorDB has few results, supplement with PubMed API
    if (relations.length < 3) {
      try {
        console.log('Supplementing with PubMed API search')
        const searchResponse = await searchApi.search(query, 10, undefined, 'pubmed')
        const papers = searchResponse.results as unknown as PaperResult[]

        if (papers && papers.length > 0) {
          papers.slice(0, Math.max(3, 5 - relations.length)).forEach((paper, idx) => {
            const paperTitle = paper.title.length > 45
              ? paper.title.substring(0, 45) + '...'
              : paper.title

            if (addedWords.has(paperTitle.toLowerCase())) return
            addedWords.add(paperTitle.toLowerCase())

            const paperKeywords = paper.keywords || []
            const level2Children: WordRelation[] = paperKeywords.slice(0, 3).map((kw, kwIdx) => {
              if (addedWords.has(kw.toLowerCase())) return null
              addedWords.add(kw.toLowerCase())
              return {
                word: kw,
                similarity: 0.75 - kwIdx * 0.04,
                category: '키워드'
              }
            }).filter((c): c is WordRelation => c !== null)

            relations.push({
              word: paperTitle,
              similarity: 0.85 - idx * 0.06,
              category: 'PubMed논문',
              children: level2Children.length > 0 ? level2Children : undefined
            })
            paperCount++
          })
        }
      } catch (err) {
        console.log('PubMed API search failed:', err)
      }
    }

    // 3. ALWAYS: Add semantic ontology connections for query terms
    const semanticData = generateSemanticNetwork(query)
    if (semanticData && semanticData.relations) {
      // Add top semantic relations that aren't already present
      semanticData.relations.slice(0, 4).forEach((rel) => {
        if (!addedWords.has(rel.word.toLowerCase())) {
          addedWords.add(rel.word.toLowerCase())
          relations.push({
            ...rel,
            similarity: rel.similarity * 0.9, // Slightly lower than direct matches
            category: rel.category === '핵심개념' ? '의미연관' : rel.category
          })
        }
      })
    }

    // If still no results, return full semantic network
    if (relations.length === 0) {
      console.log('No results from any source, using semantic network')
      return generateSemanticNetwork(query)
    }

    // Build description
    let description = `"${query}" `
    if (vectorDbUsed) {
      description += `벡터DB 검색 결과 (${paperCount}편, Hybrid Search)`
    } else if (paperCount > 0) {
      description += `PubMed 검색 결과 (${paperCount}편)`
    } else {
      description += '의미적 유사어 네트워크'
    }

    return {
      centerWord: query,
      description,
      relations,
      isFromApi: true
    }
  } catch (error) {
    console.error('Failed to build network from search, using semantic fallback:', error)
    // Fallback to semantic network generation
    return generateSemanticNetwork(query)
  }
}

export default function VectorSpaceAnimation() {
  const [searchQuery, setSearchQuery] = useState('')
  const [activeQuery, setActiveQuery] = useState('cancer')
  const [isPlaying, setIsPlaying] = useState(true)
  const [showConnections, setShowConnections] = useState(true)
  const [showSimilarity, setShowSimilarity] = useState(true)
  const [hoveredWord, setHoveredWord] = useState<string | null>(null)
  const [visibleLevel, setVisibleLevel] = useState(6)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [apiData, setApiData] = useState<SearchTermData | null>(null)
  const [useApiMode, setUseApiMode] = useState(true)

  // Use API data if available, otherwise fall back to sample data
  const currentData = useMemo(() => {
    if (useApiMode && apiData) {
      return apiData
    }
    return SEMANTIC_DATA[activeQuery.toLowerCase()] || SEMANTIC_DATA.cancer
  }, [apiData, activeQuery, useApiMode])

  const { nodes, connections } = useMemo(() => {
    return flattenRelations(currentData, 6)
  }, [currentData])

  // Fetch data from API when query changes
  const fetchApiData = useCallback(async (query: string) => {
    if (!useApiMode) return

    setIsLoading(true)
    setError(null)

    try {
      const data = await buildNetworkFromSearch(query)
      if (data) {
        setApiData(data)
        // Show info message if using semantic network instead of API
        if (data.description.includes('의미적 유사어')) {
          setError(null) // Clear error - semantic network is valid
        }
      } else {
        // This case should not happen anymore as we have semantic fallback
        setError('의미적 유사어 네트워크를 생성합니다.')
        setApiData(generateSemanticNetwork(query))
      }
    } catch (err) {
      console.error('fetchApiData error:', err)
      // Use semantic network as fallback
      setApiData(generateSemanticNetwork(query))
      setError(null)
    } finally {
      setIsLoading(false)
    }
  }, [useApiMode])

  useEffect(() => {
    nodes.forEach((node) => {
      node.position.set(
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 12
      )
    })
  }, [activeQuery, nodes, apiData])

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    const query = searchQuery.trim()
    if (!query) return

    setActiveQuery(query.toLowerCase())

    if (useApiMode) {
      await fetchApiData(query)
    }
  }

  const handlePresetClick = async (query: string) => {
    setActiveQuery(query)
    setSearchQuery(query)

    if (useApiMode) {
      await fetchApiData(query)
    } else {
      setApiData(null)
    }
  }

  const handleRandomize = () => {
    nodes.forEach((node) => {
      node.position.set(
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 12
      )
    })
  }

  const presetQueries = Object.keys(SEMANTIC_DATA)

  // Count nodes per level
  const levelCounts = [0, 0, 0, 0, 0, 0, 0]
  nodes.forEach((n) => {
    if (n.level < 7) levelCounts[n.level]++
  })

  return (
    <div className="glossy-panel p-6">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-4">
        <div>
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <span className="text-2xl">🧬</span>
            논문 검색 벡터 스페이스
            {currentData.isFromApi && (
              <span className="px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded-full border border-green-400/30">
                API
              </span>
            )}
          </h2>
          <p className="text-white/60 text-sm mt-1">{currentData.description}</p>
        </div>
        <div className="flex items-center gap-2">
          {/* API Mode Toggle */}
          <button
            onClick={() => {
              setUseApiMode(!useApiMode)
              if (!useApiMode) {
                setApiData(null)
              }
            }}
            className={`p-2 rounded-lg transition-all flex items-center gap-1 ${
              useApiMode
                ? 'bg-green-500/20 text-green-400 border border-green-400/30'
                : 'bg-white/10 text-white/50 border border-white/20'
            }`}
            title={useApiMode ? 'API 모드 (실시간 검색)' : '샘플 데이터 모드'}
          >
            <Database size={18} />
            <span className="text-xs">{useApiMode ? 'API' : '샘플'}</span>
          </button>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`p-2 rounded-lg transition-all ${
              isPlaying ? 'bg-orange-500/20 text-orange-400 border border-orange-400/30'
                : 'bg-green-500/20 text-green-400 border border-green-400/30'
            }`}
          >
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <button onClick={handleRandomize} className="p-2 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-400/30 hover:bg-purple-500/30 transition-all">
            <Zap size={20} />
          </button>
          <button onClick={() => setShowConnections(!showConnections)} className={`px-3 py-2 rounded-lg text-sm transition-all ${showConnections ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-400/30' : 'bg-white/10 text-white/70 border border-white/20'}`}>
            연결선
          </button>
          <button onClick={() => setShowSimilarity(!showSimilarity)} className={`px-3 py-2 rounded-lg text-sm transition-all ${showSimilarity ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-400/30' : 'bg-white/10 text-white/70 border border-white/20'}`}>
            유사도%
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-4 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30 flex items-center gap-2">
          <AlertCircle size={18} className="text-yellow-400" />
          <span className="text-yellow-400 text-sm">{error}</span>
        </div>
      )}

      {/* Level Control */}
      <div className="flex items-center gap-2 mb-4 p-3 rounded-lg bg-white/5 border border-white/10 flex-wrap">
        <Layers size={18} className="text-white/60" />
        <span className="text-sm text-white/70">표시 단계:</span>
        {[1, 2, 3, 4, 5, 6].map((level) => (
          <button
            key={level}
            onClick={() => setVisibleLevel(level)}
            className={`px-2 py-1 rounded-lg text-xs font-medium transition-all ${
              visibleLevel >= level
                ? 'text-white'
                : 'bg-white/5 text-white/40 border border-white/10'
            }`}
            style={{
              backgroundColor: visibleLevel >= level ? LEVEL_COLORS[level] + '40' : undefined,
              borderColor: visibleLevel >= level ? LEVEL_COLORS[level] : undefined,
            }}
          >
            L{level} ({levelCounts[level]})
          </button>
        ))}
        <span className="text-xs text-white/40 ml-2">총 {nodes.length}개 노드</span>
      </div>

      <form onSubmit={handleSearch} className="mb-4">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-white/50" size={18} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={useApiMode ? "논문 검색어 입력 (예: CRISPR, cancer, immunotherapy)..." : "샘플 검색어 선택..."}
              className="w-full pl-10 pr-4 py-2 rounded-lg bg-white/10 border border-white/20 text-white placeholder-white/40 focus:outline-none focus:border-cyan-400/50"
              disabled={isLoading}
            />
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className="px-4 py-2 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-400/30 hover:bg-cyan-500/30 transition-all disabled:opacity-50 flex items-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                검색중...
              </>
            ) : (
              '검색'
            )}
          </button>
        </div>
      </form>

      <div className="flex flex-wrap gap-2 mb-4">
        <span className="text-xs text-white/40 py-1.5">샘플:</span>
        {presetQueries.map((query) => (
          <button
            key={query}
            onClick={() => handlePresetClick(query)}
            disabled={isLoading}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all disabled:opacity-50 ${
              activeQuery === query
                ? 'bg-cyan-500/30 text-cyan-400 border border-cyan-400/50'
                : 'bg-white/5 text-white/60 border border-white/10 hover:bg-white/10'
            }`}
          >
            {query}
          </button>
        ))}
      </div>

      <div className="relative w-full h-[520px] rounded-xl overflow-hidden bg-slate-900/80 border border-white/10">
        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-900/90 backdrop-blur-sm">
            <div className="text-center">
              <Loader2 size={48} className="animate-spin text-cyan-400 mx-auto mb-4" />
              <p className="text-white/80 font-medium">논문 검색 중...</p>
              <p className="text-white/50 text-sm mt-1">벡터 스페이스 구성 중</p>
            </div>
          </div>
        )}

        <Canvas camera={{ position: [6, 4, 6], fov: 50 }}>
          <Scene
            nodes={nodes}
            connections={connections}
            hoveredWord={hoveredWord}
            setHoveredWord={setHoveredWord}
            isPlaying={isPlaying}
            showConnections={showConnections}
            showSimilarity={showSimilarity}
            visibleLevel={visibleLevel}
          />
        </Canvas>

        {hoveredWord && (
          <div className="absolute top-4 left-4 p-3 rounded-lg bg-black/85 backdrop-blur-sm border border-cyan-400/30">
            <div className="text-cyan-400 font-medium">{nodes.find((n) => n.id === hoveredWord)?.text}</div>
            {nodes.find((n) => n.id === hoveredWord)?.level !== 0 && (
              <div className="text-white/70 text-sm mt-1">
                유사도: <span className="text-yellow-400 font-bold">{((nodes.find((n) => n.id === hoveredWord)?.similarity || 0) * 100).toFixed(0)}%</span>
                <span className="text-white/40 ml-2">Level {nodes.find((n) => n.id === hoveredWord)?.level}</span>
              </div>
            )}
            <div className="text-white/50 text-xs mt-1">
              {nodes.find((n) => n.id === hoveredWord)?.category}
            </div>
          </div>
        )}

        {/* Data Source Indicator */}
        <div className="absolute top-4 right-4 p-2 rounded-lg bg-black/70 backdrop-blur-sm border border-white/10">
          <div className="flex items-center gap-2 text-xs">
            {currentData.isFromApi ? (
              <>
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-green-400">실시간 API</span>
              </>
            ) : (
              <>
                <div className="w-2 h-2 rounded-full bg-yellow-400" />
                <span className="text-yellow-400">샘플 데이터</span>
              </>
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 right-4 p-2 rounded-lg bg-black/70 backdrop-blur-sm border border-white/10">
          <div className="text-xs text-white/50 mb-1 flex items-center gap-1">
            <Info size={12} /> 유사도 기반 노드
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
            {currentData.isFromApi ? (
              <>
                <div className="flex items-center gap-1 text-[9px]">
                  <div className="w-2 h-2 rounded-full bg-yellow-400" />
                  <span className="text-white/70">검색어</span>
                </div>
                <div className="flex items-center gap-1 text-[9px]">
                  <div className="w-2 h-2 rounded-full bg-green-400" />
                  <span className="text-white/70">고유사 (≥80%)</span>
                </div>
                <div className="flex items-center gap-1 text-[9px]">
                  <div className="w-2 h-2 rounded-full bg-cyan-400" />
                  <span className="text-white/70">중유사 (60-80%)</span>
                </div>
                <div className="flex items-center gap-1 text-[9px]">
                  <div className="w-2 h-2 rounded-full bg-purple-400" />
                  <span className="text-white/70">저유사 (&lt;60%)</span>
                </div>
              </>
            ) : (
              ['중심', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6'].map((label, i) => (
                <div key={i} className="flex items-center gap-1 text-[9px]">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: LEVEL_COLORS[i] }} />
                  <span className="text-white/70">{label}</span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="absolute bottom-4 left-4 p-2 rounded-lg bg-black/50 text-xs text-white/50">
          드래그: 회전 | 스크롤: 확대
        </div>
      </div>

      <div className="mt-4 p-4 rounded-xl bg-white/5 border border-white/10">
        <h3 className="text-sm font-medium text-white mb-2 flex items-center gap-2">
          {currentData.isFromApi ? '논문 검색 벡터 스페이스' : '6단계 유사도 네트워크'}
          {currentData.isFromApi && (
            <span className="px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded border border-green-400/30">
              PubMed API
            </span>
          )}
        </h3>
        <div className="grid grid-cols-7 gap-1 mb-3">
          {['중심', '1차', '2차', '3차', '4차', '5차', '6차'].map((label, i) => (
            <div key={i} className="text-center p-1.5 rounded-lg" style={{ backgroundColor: LEVEL_COLORS[i] + '20' }}>
              <div className="text-sm font-bold" style={{ color: LEVEL_COLORS[i] }}>{levelCounts[i]}</div>
              <div className="text-[8px] text-white/60">{label}</div>
            </div>
          ))}
        </div>
        <p className="text-xs text-white/60">
          {currentData.isFromApi
            ? '검색어와 유사도를 기반으로 벡터 스페이스를 시각화합니다. 녹색(고유사 ≥80%) → 청록색(중유사 60-80%) → 보라색(저유사 <60%) 순으로 유사도를 나타냅니다.'
            : '중심 단어에서 시작하여 6단계까지의 의미적 연관 관계를 시각화합니다. 가까울수록 유사도가 높으며, 단계가 깊어질수록 간접적인 연관 관계를 나타냅니다.'
          }
        </p>
      </div>
    </div>
  )
}

// API Response Types
export interface UserProfile {
  id: string;
  username: string;
  displayName: string;
  bio: string;
  profileImageUrl: string;
  location?: string;
  joinDate: string;
  followersCount: number;
  followingCount: number;
  postsCount: number;
  verified: boolean;
}

export interface Post {
  id: string;
  rank: number;
  tag: string;
  tagColor: string;
  timestamp: string;
  content: string;
  likes: number;
  shares: number;
  replies: number;
  relevanceScore: number;
  sentimentScore: number;
  concerns: string[];
}

export interface MentalHealthAssessment {
  id: string;
  userId: string;
  username: string;
  overallRiskLevel: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  confidenceScore: number;
  analysisDate: string;
  keyFindings: string[];
  riskFactors: RiskFactor[];
  recommendations: string[];
  posts: Post[];
}

export interface RiskFactor {
  category: string;
  severity: 'LOW' | 'MODERATE' | 'HIGH';
  description: string;
  evidence: string[];
  frequency: number;
}

export interface AIReport {
  id: string;
  assessmentId: string;
  executiveSummary: string;
  detailedAnalysis: {
    behavioralPatterns: string;
    languageAnalysis: string;
    temporalPatterns: string;
    socialInteractionAnalysis: string;
  };
  clinicalRecommendations: string[];
  followUpActions: string[];
  generatedAt: string;
}

export interface AnalysisRequest {
  username: string;
  analysisType?: 'standard' | 'detailed';
  timeframe?: '7d' | '30d' | '90d';
}

export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}

export interface StreamingResponse {
  chunk: string;
  isComplete: boolean;
  totalChunks?: number;
  currentChunk?: number;
}
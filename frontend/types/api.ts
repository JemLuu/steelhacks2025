// API Response Types matching backend
export interface UserProfile {
  username: string;
  icon_url: string;
  total_karma: number;
  bio: string;
  cake_day: string;
}

export interface AssessmentItem {
  type: 'post' | 'comment';
  subreddit: string;
  permalink: string;
  title: string;
  content: string;
  created_at: string;
  indicators: string[];
  relevance_score: number;
}

export interface MentalHealthAssessment {
  username: string;
  model: string;
  executive_summary: string;
  confidence_score: number;
  mental_health_score: number;
  key_points: string[];
  items: AssessmentItem[];
}

// Legacy interface - keeping for display logic
export interface Post {
  id: string;
  rank: number;
  tag: string;
  tagColor: string;
  timestamp: string;
  title?: string;
  content: string;
  subreddit: string;
  upvotes: number;
  downvotes: number;
  score: number;
  comments: number;
  relevanceScore: number;
  sentimentScore: number;
  concerns: string[];
  postType: 'text' | 'link' | 'image';
}

// Request/Response interfaces
export interface AnalysisRequest {
  username: string;
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
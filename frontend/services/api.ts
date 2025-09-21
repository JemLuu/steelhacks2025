import {
  UserProfile,
  MentalHealthAssessment,
  AssessmentItem,
  AnalysisRequest,
  APIResponse,
  Post
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

// Helper function to format relative time
function formatRelativeTime(dateString: string): string {
  // Add debugging and fallback
  if (!dateString) {
    console.warn('No date string provided to formatRelativeTime');
    return 'Unknown';
  }

  const date = new Date(dateString);

  // Check if date is valid
  if (isNaN(date.getTime())) {
    console.warn('Invalid date string:', dateString);
    return 'Unknown';
  }

  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  // Handle future dates or negative differences
  if (diffInSeconds < 0) {
    return 'Just now';
  }

  if (diffInSeconds < 60) {
    return 'Just now';
  }

  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) {
    return `${diffInMinutes}m ago`;
  }

  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) {
    return `${diffInHours}h ago`;
  }

  const diffInDays = Math.floor(diffInHours / 24);
  if (diffInDays < 30) {
    return `${diffInDays}d ago`;
  }

  const diffInMonths = Math.floor(diffInDays / 30);
  if (diffInMonths < 12) {
    return `${diffInMonths}mo ago`;
  }

  const diffInYears = Math.floor(diffInMonths / 12);
  return `${diffInYears}y ago`;
}

// Helper function to transform backend assessment to display format
function transformAssessmentToDisplayFormat(backendAssessment: MentalHealthAssessment): {
  assessment: any;
  posts: Post[];
} {
  // Extract risk level from mental health score (backend sends 0-100)
  const getRiskLevel = (mental_health_score: number): 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL' => {
    if (mental_health_score >= 90) return 'CRITICAL';
    if (mental_health_score >= 70) return 'HIGH';
    if (mental_health_score >= 40) return 'MODERATE';
    return 'LOW';
  };

  // Transform assessment items to posts for display
  // Sort items by relevance score (highest first) before transforming
  const sortedItems = [...backendAssessment.items].sort((a, b) => b.relevance_score - a.relevance_score);

  const posts: Post[] = sortedItems.map((item, index) => {
    // Use the title from the API response directly
    const title = item.title || `${item.type === 'post' ? 'Post' : 'Comment'} in r/${item.subreddit}`;
    const content = item.content;

    return {
      id: `${item.type}_${index}`,
      rank: index + 1, // Rank based on sorted order
      tag: item.indicators[0] || 'general',
      tagColor: item.relevance_score > 8 ? 'bg-red-500' :
                item.relevance_score > 6 ? 'bg-orange-500' : 'bg-yellow-500',
      timestamp: formatRelativeTime(item.created_at),
      title,
      content,
      subreddit: `r/${item.subreddit}`,
      upvotes: Math.floor(Math.random() * 100) + 10, // Placeholder
      downvotes: Math.floor(Math.random() * 20),
      score: Math.floor(Math.random() * 80) + 10,
      comments: Math.floor(Math.random() * 50) + 5,
      relevanceScore: item.relevance_score,
      sentimentScore: item.score.overall_score, // Use actual overall score from API
      concerns: item.indicators,
      postType: item.type === 'post' ? 'text' : 'text'
    };
  });

  // Create display assessment object
  const assessment = {
    id: backendAssessment.username + '_assessment_' + Date.now(),
    userId: 'user_' + Date.now(),
    username: backendAssessment.username,
    executiveSummary: backendAssessment.executive_summary,
    overallRiskLevel: getRiskLevel(backendAssessment.mental_health_score),
    confidenceScore: Math.round(backendAssessment.confidence_score),
    mental_health_score: Math.round(backendAssessment.mental_health_score),
    analysisDate: new Date().toISOString(),
    keyFindings: [
      ...backendAssessment.executive_summary.split('.').filter(s => s.trim().length > 0).slice(0, 2).map(s => s.trim() + '.'),
      `Analysis identified ${backendAssessment.items.length} relevant posts/comments`,
    ],
    riskFactors: backendAssessment.items
      .filter(item => item.relevance_score > 6)
      .slice(0, 3)
      .map(item => ({
        category: item.indicators[0] || 'Mental Health Concern',
        severity: item.relevance_score > 8 ? 'HIGH' : 'MODERATE',
        description: item.content.substring(0, 100) + '...',
        evidence: item.indicators,
        frequency: Math.round(item.relevance_score * 10)
      })),
    recommendations: [
      'Consider professional mental health consultation',
      'Monitor social media usage patterns',
      'Encourage healthy coping mechanisms'
    ],
    posts: posts
  };

  return { assessment, posts };
}

// Transform backend profile to display format
function transformProfileToDisplayFormat(backendProfile: UserProfile): any {
  return {
    id: 'user_' + Date.now(),
    username: backendProfile.username,
    displayName: backendProfile.username,
    bio: backendProfile.bio || 'No bio available',
    profileImageUrl: backendProfile.icon_url || 'https://picsum.photos/400/400',
    location: undefined,
    joinDate: backendProfile.cake_day,
    karma: backendProfile.total_karma,
    postKarma: Math.floor(backendProfile.total_karma * 0.6),
    commentKarma: Math.floor(backendProfile.total_karma * 0.4),
    postsCount: Math.floor(Math.random() * 500) + 50, // Placeholder
    verified: false
  };
}

// API Service Functions
export const apiService = {
  // User Profile API
  async getUserProfile(username: string): Promise<APIResponse<any>> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/reddit/profile/${username}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const backendProfile: UserProfile = await response.json();
      const displayProfile = transformProfileToDisplayFormat(backendProfile);

      return {
        success: true,
        data: displayProfile
      };
    } catch (error) {
      console.error('Profile fetch error:', error);
      return {
        success: false,
        error: {
          code: 'USER_NOT_FOUND',
          message: 'User profile could not be retrieved',
          details: error
        }
      };
    }
  },

  // Mental Health Assessment API
  async createAssessment(request: AnalysisRequest): Promise<APIResponse<any>> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/reddit/assessment/${request.username}`);

        if (!response.ok) {
          const errorText = await response.text().catch(() => '[Unable to read response body]');
          console.error(
            '[Assessment API Error]',
            {
          url: `${API_BASE_URL}/api/reddit/${request.username}/assessment`,
          status: response.status,
          statusText: response.statusText,
          headers: Object.fromEntries(response.headers.entries()),
          body: errorText
            }
          );
          throw new Error(
            `HTTP ${response.status}: ${response.statusText}\n` +
            `URL: ${API_BASE_URL}/api/reddit/assessment/${request.username}\n` +
            `Response Body: ${errorText}`
          );
        }


      const backendAssessment: MentalHealthAssessment = await response.json();
      const { assessment } = transformAssessmentToDisplayFormat(backendAssessment);

      return {
        success: true,
        data: assessment
      };
    } catch (error) {
      console.error('Assessment fetch error:', error);

      // Check if this is a rate limit error and preserve the detailed message
      const errorMessage = error instanceof Error ? error.message : '';
      const isRateLimit = errorMessage.includes('429') ||
                         errorMessage.includes('502') ||
                         errorMessage.toLowerCase().includes('rate limit') ||
                         errorMessage.toLowerCase().includes('too many requests') ||
                         errorMessage.toLowerCase().includes('rate_limit_error') ||
                         errorMessage.toLowerCase().includes('claude api error: 429');

      return {
        success: false,
        error: {
          code: isRateLimit ? 'RATE_LIMIT_ERROR' : 'ANALYSIS_FAILED',
          message: isRateLimit ? errorMessage : 'Mental health assessment could not be completed',
          details: error
        }
      };
    }
  },

  // AI Report API - now uses assessment data
  async getAIReport(assessmentId: string, executiveSummary: string, onChunk?: (chunk: any) => void): Promise<APIResponse<any>> {
    try {
      if (onChunk) {
        // Stream the executive summary directly without additional API call
        const fullText = executiveSummary;
        const chunkSize = 3;

        for (let i = 0; i < fullText.length; i += chunkSize) {
          const chunk = fullText.slice(i, i + chunkSize);
          const currentChunk = Math.floor(i / chunkSize) + 1;
          const isComplete = i + chunkSize >= fullText.length;

          onChunk({
            chunk,
            isComplete,
            totalChunks: Math.ceil(fullText.length / chunkSize),
            currentChunk
          });

          // Simulate streaming delay
          await new Promise(resolve => setTimeout(resolve, 25));
        }

        // Return structured AI report
        return {
          success: true,
          data: {
            id: 'report_' + Date.now(),
            assessmentId: assessmentId,
            executiveSummary: executiveSummary,
            detailedAnalysis: {
              behavioralPatterns: 'Analysis completed based on user activity patterns.',
              languageAnalysis: 'Text analysis and sentiment evaluation performed.',
              temporalPatterns: 'Activity timeline reviewed.',
              socialInteractionAnalysis: 'Social engagement patterns analyzed.'
            },
            clinicalRecommendations: [
              'Professional mental health evaluation recommended',
              'Consider monitoring social media usage patterns',
              'Encourage seeking support from mental health professionals'
            ],
            followUpActions: [
              'Schedule follow-up assessment in 2-4 weeks',
              'Monitor for escalation of concerning indicators',
              'Consider family/support system involvement'
            ],
            generatedAt: new Date().toISOString()
          }
        };
      }

      return {
        success: true,
        data: {
          id: 'report_' + Date.now(),
          assessmentId: assessmentId,
          executiveSummary: 'Unable to generate streaming report.',
          detailedAnalysis: {},
          clinicalRecommendations: [],
          followUpActions: [],
          generatedAt: new Date().toISOString()
        }
      };
    } catch (error) {
      console.error('AI Report error:', error);
      return {
        success: false,
        error: {
          code: 'REPORT_GENERATION_FAILED',
          message: 'AI report could not be generated',
          details: error
        }
      };
    }
  },

  // Get existing assessment
  async getAssessment(assessmentId: string): Promise<APIResponse<any>> {
    try {
      // Extract username from assessmentId or use default
      const username = assessmentId.split('_')[0] || 'testuser';

      const response = await fetch(`${API_BASE_URL}/api/reddit/assessment/${username}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const backendAssessment: MentalHealthAssessment = await response.json();
      const { assessment } = transformAssessmentToDisplayFormat(backendAssessment);

      return {
        success: true,
        data: assessment
      };
    } catch (error) {
      console.error('Get Assessment error:', error);
      return {
        success: false,
        error: {
          code: 'ASSESSMENT_NOT_FOUND',
          message: 'Assessment could not be retrieved',
          details: error
        }
      };
    }
  }
};
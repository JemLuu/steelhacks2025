import {
  UserProfile,
  MentalHealthAssessment,
  AIReport,
  AnalysisRequest,
  APIResponse,
  StreamingResponse,
  Post
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

// Mock data - replace with actual API calls later
const mockUserProfile: UserProfile = {
  id: 'user_123',
  username: 'joncoulter',
  displayName: 'Jonathan Coulter',
  bio: 'Just trying to make sense of the world, one post at a time. Coffee enthusiast and weekend philosopher.',
  profileImageUrl: 'https://picsum.photos/400/400',
  location: 'Pittsburgh, PA',
  joinDate: '2020-03-15T00:00:00Z',
  followersCount: 421,
  followingCount: 892,
  postsCount: 350,
  verified: false
};

const mockPosts: Post[] = [
  {
    id: 'post_1',
    rank: 1,
    tag: 'concerning',
    tagColor: 'bg-orange-500',
    timestamp: '2h ago',
    content: 'Another sleepless night. Can\'t seem to turn my brain off. The thoughts just keep racing and I\'m exhausted. #insomnia #mentalhealth',
    likes: 12,
    shares: 3,
    replies: 7,
    relevanceScore: 92,
    sentimentScore: -0.7,
    concerns: ['sleep_disturbance', 'racing_thoughts', 'exhaustion']
  },
  {
    id: 'post_2',
    rank: 2,
    tag: 'mood indicator',
    tagColor: 'bg-red-500',
    timestamp: '5h ago',
    content: 'Feeling so disconnected from everyone lately. Even in a room full of people, I feel completely alone. Is this what my life has become?',
    likes: 8,
    shares: 1,
    replies: 4,
    relevanceScore: 89,
    sentimentScore: -0.8,
    concerns: ['social_isolation', 'loneliness', 'existential_questioning']
  },
  {
    id: 'post_3',
    rank: 3,
    tag: 'behavioral',
    tagColor: 'bg-yellow-500',
    timestamp: '1d ago',
    content: 'Canceled plans again tonight. Just don\'t have the energy to pretend everything is fine. My friends probably think I\'m flaky.',
    likes: 15,
    shares: 2,
    replies: 9,
    relevanceScore: 85,
    sentimentScore: -0.6,
    concerns: ['social_withdrawal', 'energy_depletion', 'masking_behavior']
  },
  {
    id: 'post_4',
    rank: 4,
    tag: 'sleep pattern',
    tagColor: 'bg-orange-500',
    timestamp: '2d ago',
    content: '3 AM and still wide awake. This has been my routine for weeks now. Coffee is basically a food group at this point ☕️',
    likes: 6,
    shares: 0,
    replies: 2,
    relevanceScore: 78,
    sentimentScore: -0.4,
    concerns: ['insomnia', 'sleep_cycle_disruption', 'caffeine_dependency']
  },
  {
    id: 'post_5',
    rank: 5,
    tag: 'emotional',
    tagColor: 'bg-red-500',
    timestamp: '3d ago',
    content: 'Some days I wonder if anyone would notice if I just disappeared. Not in a scary way, just... would my absence make a difference?',
    likes: 4,
    shares: 1,
    replies: 12,
    relevanceScore: 94,
    sentimentScore: -0.9,
    concerns: ['existential_thoughts', 'self_worth', 'passive_ideation']
  }
];

const mockAssessment: MentalHealthAssessment = {
  id: 'assessment_123',
  userId: 'user_123',
  username: 'joncoulter',
  overallRiskLevel: 'MODERATE',
  confidenceScore: 78,
  analysisDate: new Date().toISOString(),
  keyFindings: [
    'Sleep disturbance patterns observed in 78% of analyzed posts over the past month',
    'Social withdrawal behaviors indicated through frequent mentions of canceled plans and isolation',
    'Elevated risk markers for depression based on language sentiment analysis and emotional expression patterns'
  ],
  riskFactors: [
    {
      category: 'Sleep Disruption',
      severity: 'HIGH',
      description: 'Chronic insomnia and irregular sleep patterns',
      evidence: ['Multiple mentions of sleepless nights', 'Late-night posting patterns', 'Fatigue-related content'],
      frequency: 78
    },
    {
      category: 'Social Isolation',
      severity: 'MODERATE',
      description: 'Withdrawal from social activities and relationships',
      evidence: ['Canceled social plans', 'Feelings of disconnection', 'Loneliness expressions'],
      frequency: 65
    }
  ],
  recommendations: [
    'Recommend sleep hygiene evaluation and possible sleep study',
    'Consider referral to mental health professional for depression screening',
    'Suggest social support interventions and gradual re-engagement strategies'
  ],
  posts: mockPosts
};

const mockAIReport: AIReport = {
  id: 'report_123',
  assessmentId: 'assessment_123',
  executiveSummary: 'Based on comprehensive analysis of @joncoulter\'s recent Reddit activity, our AI has identified several patterns indicative of potential mental health concerns. The analysis reveals consistent themes around sleep disturbances, social withdrawal, and emotional distress spanning multiple weeks. Language sentiment analysis shows a 73% negative emotional valence in recent posts, with particular emphasis on isolation and fatigue-related keywords. Temporal posting patterns indicate irregular sleep cycles, with 68% of posts occurring during late-night hours (11 PM - 4 AM). The user frequently expresses feelings of disconnection, cancels social commitments, and demonstrates classic symptoms of depression including anhedonia and social isolation.',
  detailedAnalysis: {
    behavioralPatterns: 'Analysis reveals consistent patterns of social avoidance and activity withdrawal. The user has mentioned canceling social plans in 40% of relevant posts, indicating potential behavioral activation deficits commonly associated with depressive episodes.',
    languageAnalysis: 'Linguistic markers show increased use of first-person singular pronouns, absolutist language, and negative emotional expressions. Sentiment analysis indicates a 73% negative valence with particular clustering around themes of exhaustion, isolation, and existential questioning.',
    temporalPatterns: 'Posting patterns show significant circadian rhythm disruption, with 68% of posts occurring between 11 PM and 4 AM. This suggests potential sleep disorders or mood-related sleep disturbances requiring clinical attention.',
    socialInteractionAnalysis: 'Decreased engagement with others\' content and reduced reciprocal social interactions. The user\'s posts receive emotional support responses, indicating social network awareness of distress signals.'
  },
  clinicalRecommendations: [
    'Immediate sleep hygiene assessment and potential sleep study referral',
    'Depression screening using validated instruments (PHQ-9, GAD-7)',
    'Consideration for cognitive behavioral therapy for insomnia (CBT-I)',
    'Social support system evaluation and intervention planning'
  ],
  followUpActions: [
    'Schedule follow-up assessment in 2-4 weeks',
    'Monitor for escalation of suicidal ideation or self-harm indicators',
    'Coordinate with primary care provider for comprehensive evaluation',
    'Consider family/social support involvement in treatment planning'
  ],
  generatedAt: new Date().toISOString()
};

// API Service Functions
export const apiService = {
  // User Profile API
  async getUserProfile(username: string): Promise<APIResponse<UserProfile>> {
    try {
      // TODO: Replace with actual API call
      // const response = await fetch(`${API_BASE_URL}/users/${username}`);
      // const data = await response.json();

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      return {
        success: true,
        data: mockUserProfile
      };
    } catch (error) {
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
  async createAssessment(request: AnalysisRequest): Promise<APIResponse<MentalHealthAssessment>> {
    try {
      // TODO: Replace with actual API call
      // const response = await fetch(`${API_BASE_URL}/assessments`, {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(request)
      // });
      // const data = await response.json();

      // Simulate API delay for analysis
      await new Promise(resolve => setTimeout(resolve, 2000));

      return {
        success: true,
        data: mockAssessment
      };
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'ANALYSIS_FAILED',
          message: 'Mental health assessment could not be completed',
          details: error
        }
      };
    }
  },

  // AI Report API with streaming simulation
  async getAIReport(assessmentId: string, onChunk?: (chunk: StreamingResponse) => void): Promise<APIResponse<AIReport>> {
    try {
      // TODO: Replace with actual streaming API call
      // const response = await fetch(`${API_BASE_URL}/reports/${assessmentId}/stream`);
      // const reader = response.body?.getReader();

      // Simulate streaming response
      if (onChunk) {
        const fullText = mockAIReport.executiveSummary;
        const chunkSize = 3;
        const totalChunks = Math.ceil(fullText.length / chunkSize);

        for (let i = 0; i < fullText.length; i += chunkSize) {
          const chunk = fullText.slice(i, i + chunkSize);
          const currentChunk = Math.floor(i / chunkSize) + 1;
          const isComplete = i + chunkSize >= fullText.length;

          onChunk({
            chunk,
            isComplete,
            totalChunks,
            currentChunk
          });

          // Simulate streaming delay
          await new Promise(resolve => setTimeout(resolve, 50));
        }
      }

      return {
        success: true,
        data: mockAIReport
      };
    } catch (error) {
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
  async getAssessment(assessmentId: string): Promise<APIResponse<MentalHealthAssessment>> {
    try {
      // TODO: Replace with actual API call
      // const response = await fetch(`${API_BASE_URL}/assessments/${assessmentId}`);
      // const data = await response.json();

      await new Promise(resolve => setTimeout(resolve, 500));

      return {
        success: true,
        data: mockAssessment
      };
    } catch (error) {
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
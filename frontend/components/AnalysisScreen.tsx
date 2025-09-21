import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ArrowLeft, MapPin, Calendar, Users, MessageCircle, ArrowUp, ArrowDown, Share, Brain, AlertTriangle, Activity, X, CheckCircle, Clock } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { apiService } from '../services/api';
import { UserProfile, MentalHealthAssessment, AIReport, Post, StreamingResponse } from '../types/api';

interface AnalysisScreenProps {
  redditHandle: string;
  onBack: () => void;
}

export default function AnalysisScreen({ redditHandle, onBack }: AnalysisScreenProps) {
  const [searchHandle, setSearchHandle] = useState(redditHandle);
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const [streamedText, setStreamedText] = useState('');
  const [showFindings, setShowFindings] = useState(false);
  const [findingIndex, setFindingIndex] = useState(0);
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);
  const [isScrollingPaused, setIsScrollingPaused] = useState(false);
  const [isAutoScrolling, setIsAutoScrolling] = useState(true);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<any>(null);

  // API Data State
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [postCount, setPostCount] = useState<number | null>(null);
  const [commentCount, setCommentCount] = useState<number | null>(null);
  const [assessment, setAssessment] = useState<MentalHealthAssessment | null>(null);
  const [aiReport, setAiReport] = useState<AIReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Rate Limit State
  const [isRateLimited, setIsRateLimited] = useState(false);
  const [rateLimitTimer, setRateLimitTimer] = useState(0);
  const [hasSearchedOnce, setHasSearchedOnce] = useState(false);

  // Loading text cycling
  const [loadingTextIndex, setLoadingTextIndex] = useState(0);
  const loadingTexts = [
    'Parsing post history',
    'Analyzing social media patterns',
    'Processing communication styles',
    'Evaluating behavioral indicators',
    'Identifying mental health signals',
    'Assessing language patterns',
  ];

  // Start rate limit timer
  const startRateLimitTimer = () => {
    setIsRateLimited(true);
    setRateLimitTimer(60);
  };

  // Load initial data
  useEffect(() => {
    const abortController = new AbortController();

    // Start rate limit timer for initial search
    if (!hasSearchedOnce) {
      setHasSearchedOnce(true);
      startRateLimitTimer();
    }

    loadAnalysisData(abortController.signal);

    return () => {
      abortController.abort();
    };
  }, [redditHandle]);

  const loadAnalysisData = async (signal?: AbortSignal, username?: string) => {
    const targetUsername = username || redditHandle;
    try {
      setLoading(true);
      setError(null);
      setStreamedText('');

      // Step 1: Get user profile
      const userResponse = await apiService.getUserProfile(targetUsername);
      if (signal?.aborted) return;

      if (!userResponse.success) {
        throw new Error(userResponse.error?.message || 'Failed to load user profile');
      }
      setUserProfile(userResponse.data!);

      // Step 1.5: Get post count (async, don't block)
      apiService.getPostCount(targetUsername).then(postCountResponse => {
        if (!signal?.aborted && postCountResponse.success) {
          setPostCount(postCountResponse.data!.post_count);
          setCommentCount(postCountResponse.data!.comment_count);
        }
      }).catch(err => {
        console.warn('Post count fetch failed:', err);
        setPostCount(0); // Default fallback
        setCommentCount(0);
      });

      // Step 2: Create mental health assessment
      if (signal?.aborted) return;

      const assessmentResponse = await apiService.createAssessment({
        username: targetUsername,
        analysisType: 'detailed',
        timeframe: '30d'
      });

      if (!assessmentResponse.success) {
        throw new Error(assessmentResponse.error?.message || 'Failed to create assessment');
      }

      setAssessment(assessmentResponse.data!);
      setIsAnalyzing(false);

      // Check if aborted before streaming
      if (signal?.aborted) return;

      // Step 3: Get AI report with streaming
      const reportResponse = await apiService.getAIReport(
        assessmentResponse.data!.id,
        assessmentResponse.data!.executiveSummary,
        (chunk: StreamingResponse) => {
          // Check if aborted before updating state
          if (signal?.aborted) return;

          setStreamedText(prev => prev + chunk.chunk);

          if (chunk.isComplete) {
            setTimeout(() => {
              if (!signal?.aborted) {
                setShowFindings(true);
              }
            }, 500);
          }
        }
      );

      if (!reportResponse.success) {
        throw new Error(reportResponse.error?.message || 'Failed to generate AI report');
      }

      setAiReport(reportResponse.data!);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unexpected error occurred';

      // Check if it's a rate limit error (can come as 429, 502, or in message text)
      if (errorMessage.includes('429') ||
          errorMessage.includes('502') ||
          errorMessage.toLowerCase().includes('rate limit') ||
          errorMessage.toLowerCase().includes('too many requests') ||
          errorMessage.toLowerCase().includes('rate_limit_error') ||
          errorMessage.toLowerCase().includes('claude api error: 429')) {
        startRateLimitTimer();
        setError('Rate limit exceeded. API usage limit reached. Please wait before trying again.');
      } else {
        setError(errorMessage);
      }
      setIsAnalyzing(false);
    } finally {
      setLoading(false);
    }
  };

  // Animate findings appearing
  useEffect(() => {
    if (showFindings && assessment && findingIndex < assessment.keyFindings.length) {
      const timer = setTimeout(() => {
        setFindingIndex(findingIndex + 1);
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [showFindings, findingIndex, assessment?.keyFindings.length]);

  // Autoscroll logic
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAutoScrolling && scrollContainerRef.current) {
      interval = setInterval(() => {
        if (scrollContainerRef.current) {
          if (scrollContainerRef.current.scrollTop + scrollContainerRef.current.clientHeight >= scrollContainerRef.current.scrollHeight) {
            scrollContainerRef.current.scrollTop = 0;
          } else {
            scrollContainerRef.current.scrollTop += 1;
          }
        }
      }, 50);
    }
    return () => clearInterval(interval);
  }, [isAutoScrolling, assessment]);

  // Rate limit timer effect
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRateLimited && rateLimitTimer > 0) {
      interval = setInterval(() => {
        setRateLimitTimer(prev => {
          if (prev <= 1) {
            setIsRateLimited(false);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRateLimited, rateLimitTimer]);

  // Loading text cycling effect
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAnalyzing) {
      interval = setInterval(() => {
        setLoadingTextIndex(prev => (prev + 1) % loadingTexts.length);
      }, 3000); // Change text every 3 seconds
    }
    return () => clearInterval(interval);
  }, [isAnalyzing, loadingTexts.length]);

  const handleSearchSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isRateLimited) return;

    if (searchHandle.trim() && searchHandle !== redditHandle) {
      // Start cooldown timer if this is the first search or after initial search
      if (!hasSearchedOnce) {
        setHasSearchedOnce(true);
        startRateLimitTimer();
      }

      // Reset all state and reload with new handle
      setUserProfile(null);
      setPostCount(null);
      setCommentCount(null);
      setAssessment(null);
      setAiReport(null);
      setStreamedText('');
      setShowFindings(false);
      setFindingIndex(0);
      setSelectedPost(null);
      setIsAnalyzing(true);
      setError(null);

      // In a real app, you'd update the URL/route here
      await loadAnalysisData(undefined, searchHandle);
    }
  };

  const handlePostClick = (post: Post) => {
    try {
      setSelectedPost(post);
      setIsScrollingPaused(true);
    } catch (error) {
      console.warn('Error opening post modal:', error);
    }
  };

  const handleCloseModal = () => {
    try {
      setSelectedPost(null);
      setIsScrollingPaused(false);
    } catch (error) {
      console.warn('Error closing post modal:', error);
    }
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'LOW': return 'bg-green-100 text-green-800';
      case 'MODERATE': return 'bg-orange-100 text-orange-800';
      case 'HIGH': return 'bg-red-100 text-red-800';
      case 'CRITICAL': return 'bg-red-200 text-red-900';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading && !userProfile) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading user profile...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="p-6 max-w-md mx-auto">
          <div className="text-center">
            <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Analysis Error</h3>
            <p className="text-gray-600 mb-4">{error}</p>
            <Button onClick={onBack} variant="outline">
              Back to Landing
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Button
                variant="ghost"
                onClick={onBack}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <ArrowLeft className="w-4 h-4" />
              </Button>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">MindScope AI</h1>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              {isRateLimited && (
                <div className="flex items-center space-x-2 px-3 py-2 bg-orange-50 border border-orange-200 rounded-lg">
                  <Clock className="w-4 h-4 text-orange-600" />
                  <span className="text-sm font-medium text-orange-800">
                    Cooldown: {Math.floor(rateLimitTimer / 60)}:{(rateLimitTimer % 60).toString().padStart(2, '0')}
                  </span>
                </div>
              )}

              <form onSubmit={handleSearchSubmit} className="flex space-x-2">
                <Input
                  type="text"
                  placeholder="Username"
                  value={searchHandle}
                  onChange={(e) => setSearchHandle(e.target.value)}
                  className="w-48 h-9 px-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={isRateLimited}
                />
                <Button
                  type="submit"
                  className="px-4 h-9 bg-gray-900 hover:bg-gray-800 text-white rounded-lg text-sm disabled:bg-gray-400 disabled:cursor-not-allowed"
                  disabled={loading || isRateLimited}
                >
                  Analyze
                </Button>
              </form>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-4">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column - User Profile & Posts */}
          <div className="space-y-6">
            {/* User Profile Card */}
            {userProfile && (
              <Card className="p-4 bg-white border border-gray-200 rounded-lg">
                <div className="flex items-start space-x-3">
                  <ImageWithFallback
                    src={userProfile.profileImageUrl}
                    alt="User profile"
                    className="w-12 h-12 rounded-full object-cover"
                  />
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900">{userProfile.displayName}</h3>
                    <p className="text-gray-600">@{userProfile.username}</p>
                    <p className="text-gray-700 mt-2 text-sm">{userProfile.bio}</p>

                    <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600">
                      {userProfile.location && (
                        <div className="flex items-center space-x-1">
                          <MapPin className="w-4 h-4" />
                          <span>{userProfile.location}</span>
                        </div>
                      )}
                      <div className="flex items-center space-x-1">
                        <Calendar className="w-4 h-4" />
                        <span>{new Date(userProfile.joinDate).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</span>
                      </div>
                    </div>

                    <div className="flex items-center space-x-6 mt-2 text-sm">
                      <span><strong>{userProfile.karma.toLocaleString()}</strong> <span className="text-gray-600">Karma</span></span>
                      <span><strong>{postCount !== null ? postCount.toLocaleString() : '...'}</strong> <span className="text-gray-600">Posts</span></span>
                      <span><strong>{commentCount !== null ? commentCount.toLocaleString() : '...'}</strong> <span className="text-gray-600">Comments</span></span>
                    </div>
                  </div>
                </div>
              </Card>
            )}

            {/* Relevant Tweets */}
            {assessment && (
              <Card className="bg-white border border-gray-200 rounded-lg">
                <div className="p-6 border-b border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-900">Relevant Posts</h3>
                  <p className="text-gray-600">{assessment.posts.length} posts</p>
                </div>

                <div
                  ref={scrollContainerRef}
                  className={`h-[440px] overflow-y-auto overflow-x-hidden cursor-pointer ${isAutoScrolling ? 'no-scrollbar' : ''}`}
                  onWheel={() => setIsAutoScrolling(false)}
                  onTouchStart={() => setIsAutoScrolling(false)}
                  onMouseDown={() => setIsAutoScrolling(false)}
                  onClick={() => {
                      if (!isAutoScrolling) {
                          setIsAutoScrolling(true);
                      }
                  }}
                >
                  <div className="space-y-0">
                    {assessment.posts.map((post, index) => (
                      <div
                        key={`${post.id}-${index}`}
                        className="p-4 border-b border-gray-100 hover:bg-gray-50 transition-colors cursor-pointer min-h-[120px]"
                        onClick={(e) => {
                          e.stopPropagation();
                          handlePostClick(post);
                        }}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <span className="text-sm font-medium text-gray-500">#{post.rank}</span>
                            <Badge className={`${post.tagColor} text-white text-xs px-2 py-1 rounded`}>
                              {post.tag}
                            </Badge>
                          </div>
                          <span className="text-sm text-gray-500">{post.timestamp}</span>
                        </div>

                        <div className="mb-2">
                          <span className="text-xs text-blue-600 font-medium">{post.subreddit}</span>
                          {post.title && (
                            <h4 className="text-sm font-semibold text-gray-900 mt-1 break-words">{post.title}</h4>
                          )}
                        </div>

                        <p className="text-gray-700 text-sm mb-3 break-words">{post.content}</p>

                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4 text-gray-500">
                            <div className="flex items-center space-x-1">
                              <ArrowUp className="w-4 h-4 text-orange-500" />
                              <span className="text-xs font-medium">{post.score}</span>
                              <ArrowDown className="w-4 h-4" />
                            </div>
                            <div className="flex items-center space-x-1">
                              <MessageCircle className="w-4 h-4" />
                              <span className="text-xs">{post.comments}</span>
                            </div>
                          </div>
                          <span className="text-xs text-blue-600 font-medium">
                            Relevance: {post.relevanceScore}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            )}
          </div>

          {/* Right Column - AI Analysis */}
          <div className="space-y-6">
            {/* Analysis Header */}
            {assessment && (
              <Card className="p-6 bg-white border border-gray-200 rounded-lg">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="flex items-center justify-center w-10 h-10 bg-blue-100 rounded-lg">
                    <Brain className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">AI Mental Health Analysis</h3>
                    <div className="space-y-2 mt-1">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm text-gray-600">Mental Risk:</span>
                        <div className="flex-1 max-w-24">
                          <Progress value={assessment.mental_health_score} className="h-2" />
                        </div>
                        <span className="text-sm font-medium text-gray-900">{assessment.mental_health_score}%</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm text-gray-600">Confidence:</span>
                        <div className="flex-1 max-w-24">
                          <Progress value={assessment.confidenceScore} className="h-2" />
                        </div>
                        <span className="text-sm font-medium text-gray-900">{assessment.confidenceScore}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                <Badge className={`${getRiskLevelColor(assessment.overallRiskLevel)} px-3 py-1 rounded-full text-sm font-medium`}>
                  {assessment.overallRiskLevel} RISK
                </Badge>
              </Card>
            )}


            {/* Mental Health Evaluation Panel */}
            {assessment && (
              <>
                {assessment.overallRiskLevel === 'LOW' && (
                  <Card className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-start space-x-3">
                      <CheckCircle className="w-6 h-6 text-green-600 mt-1" />
                      <div>
                        <h4 className="font-semibold text-green-900">
                          Positive Mental Health Indicators
                        </h4>
                        <p className="text-green-800 text-sm mt-1">
                          Analysis shows healthy communication patterns and positive engagement behaviors.
                        </p>
                      </div>
                    </div>
                  </Card>
                )}

                {assessment.overallRiskLevel === 'MODERATE' && (
                  <Card className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <div className="flex items-start space-x-3">
                      <AlertTriangle className="w-6 h-6 text-yellow-600 mt-1" />
                      <div>
                        <h4 className="font-semibold text-yellow-900">
                          Further Monitoring Recommended
                        </h4>
                        <p className="text-yellow-800 text-sm mt-1">
                          Some concerning patterns detected. Consider periodic check-ins and continued observation.
                        </p>
                      </div>
                    </div>
                  </Card>
                )}

                {(assessment.overallRiskLevel === 'HIGH' || assessment.overallRiskLevel === 'CRITICAL') && (
                  <Card className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-start space-x-3">
                      <AlertTriangle className="w-6 h-6 text-red-600 mt-1" />
                      <div>
                        <h4 className="font-semibold text-red-900">
                          {assessment.overallRiskLevel === 'CRITICAL' ? 'Immediate Professional Support Required' : 'Clinical Assessment Recommended'}
                        </h4>
                        <p className="text-red-800 text-sm mt-1">
                          {assessment.overallRiskLevel === 'CRITICAL'
                            ? 'Critical indicators detected. Immediate mental health professional consultation strongly advised.'
                            : 'Multiple risk factors identified. Professional mental health evaluation recommended.'
                          }
                        </p>
                      </div>
                    </div>
                  </Card>
                )}
              </>
            )}

            {/* Executive Summary */}
            <Card className="p-6 bg-white border border-gray-200 rounded-lg">
              <h4 className="font-semibold text-gray-900 mb-4">Executive Summary</h4>

              {isAnalyzing ? (
                <div className="flex items-center space-x-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                  <div className="relative h-6 overflow-hidden w-64">
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={loadingTextIndex}
                        initial={{ y: 12, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        exit={{ y: -12, opacity: 0 }}
                        transition={{ duration: 0.5, ease: "easeInOut" }}
                        className="absolute inset-0 flex items-center"
                      >
                        <span className="text-gray-600 text-sm">
                          {loadingTexts[loadingTextIndex]}
                        </span>
                      </motion.div>
                    </AnimatePresence>
                  </div>
                </div>
              ) : (
                <div>
                  <p className="text-gray-700 text-sm leading-relaxed">
                    {streamedText}
                    {aiReport && streamedText.length < aiReport.executiveSummary.length && (
                      <span className="animate-pulse">|</span>
                    )}
                  </p>
                </div>
              )}
            </Card>

            
            {/* Key Points */}
            {assessment && assessment.keyPoints && assessment.keyPoints.length > 0 && (
              <Card className="p-6 bg-white border border-gray-200 rounded-lg">
                <h4 className="font-semibold text-gray-900 mb-4">Key Insights</h4>
                <div className="space-y-3">
                  {assessment.keyPoints.map((point, index) => (
                    <div
                      key={index}
                      className="flex items-start space-x-3"
                    >
                      <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                      <p className="text-gray-700 text-sm font-medium">{point}</p>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        </div>
      </main>

      {/* Post Modal */}
      {selectedPost && (
        <Dialog open={true} onOpenChange={handleCloseModal}>
          <DialogContent className="!max-w-3xl !w-[90vw] mx-auto max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Post Analysis</DialogTitle>
            </DialogHeader>

            <div className="space-y-6">
              {/* Tweet Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <ImageWithFallback
                    src={userProfile?.profileImageUrl || ''}
                    alt="User profile"
                    className="w-12 h-12 rounded-full object-cover"
                  />
                  <div>
                    <h4 className="font-semibold text-gray-900">{userProfile?.displayName}</h4>
                    <p className="text-gray-600 text-sm">u/{userProfile?.username} • {selectedPost.timestamp}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium text-gray-500">#{selectedPost.rank}</span>
                  <Badge className={`${selectedPost.tagColor} text-white text-sm px-3 py-1 rounded-full`}>
                    {selectedPost.tag}
                  </Badge>
                </div>
              </div>

              {/* Post Content */}
              <div className="bg-gray-50 rounded-lg p-6">
                <div className="mb-3">
                  <span className="text-sm text-blue-600 font-medium">{selectedPost.subreddit}</span>
                  {selectedPost.title && (
                    <h3 className="text-lg font-semibold text-gray-900 mt-1">{selectedPost.title}</h3>
                  )}
                </div>
                <p className="text-gray-800 leading-relaxed text-base">{selectedPost.content}</p>
              </div>

              {/* Post Stats */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-6 text-gray-600">
                  <div className="flex items-center space-x-2">
                    <ArrowUp className="w-5 h-5 text-orange-500" />
                    <span className="text-sm font-medium">{selectedPost.score}</span>
                    <ArrowDown className="w-5 h-5" />
                  </div>
                  <div className="flex items-center space-x-2">
                    <MessageCircle className="w-5 h-5" />
                    <span className="text-sm font-medium">{selectedPost.comments}</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-600">Relevance Score</div>
                  <div className="text-lg font-bold text-blue-600">{selectedPost.relevanceScore}%</div>
                </div>
              </div>

              {/* Analysis Insights */}
              <div className="space-y-4">
                <h5 className="font-semibold text-gray-900">AI Analysis Insights</h5>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <Brain className="w-5 h-5 text-blue-600 mt-0.5" />
                    <div>
                      <h6 className="font-medium text-blue-900 mb-2">Mental Health Indicators</h6>
                      <div className="space-y-2 text-sm text-blue-800">
                        {selectedPost.concerns?.map((concern, index) => (
                          <p key={index}>• {concern.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} detected</p>
                        ))}
                        <p>• Mental health risk: {selectedPost.sentimentScore === 0 ? 'No concern detected' : selectedPost.sentimentScore > 0.7 ? 'High risk' : selectedPost.sentimentScore > 0.4 ? 'Moderate risk' : 'Low risk'}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
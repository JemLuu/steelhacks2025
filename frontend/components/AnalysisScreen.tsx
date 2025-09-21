import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence, useInView } from 'motion/react';
import { ArrowLeft, MapPin, Calendar, Users, MessageCircle, ArrowUp, ArrowDown, Share, Brain, AlertTriangle, Activity, X, CheckCircle, Clock, Moon, Sun } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { apiService } from '../services/api';
import { UserProfile, MentalHealthAssessment, AIReport, Post, StreamingResponse, MentalHealthScore } from '../types/api';

interface AnalysisScreenProps {
  redditHandle: string;
  onBack: () => void;
}

export default function AnalysisScreen({ redditHandle, onBack }: AnalysisScreenProps) {
  const { theme, toggleTheme } = useTheme();
  const [searchHandle, setSearchHandle] = useState(redditHandle);
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const [streamedText, setStreamedText] = useState('');
  const [showFindings, setShowFindings] = useState(false);
  const [findingIndex, setFindingIndex] = useState(0);
  const [streamedKeyPoints, setStreamedKeyPoints] = useState<string[]>([]);
  const [isStreamingKeyPoints, setIsStreamingKeyPoints] = useState(false);
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);

  // Animation states for progressive reveal
  const [showUserProfile, setShowUserProfile] = useState(false);
  const [showAnalysisHeader, setShowAnalysisHeader] = useState(false);
  const [showPostsList, setShowPostsList] = useState(false);
  const [showExecutiveSummary, setShowExecutiveSummary] = useState(false);
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

  // Animation refs for viewport triggers
  const categoryAnalysisRef = useRef(null);
  const isCategoryAnalysisInView = useInView(categoryAnalysisRef, { once: false, amount: 0.1 });

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
      setStreamedKeyPoints([]);
      setIsStreamingKeyPoints(false);

      // Reset animation states
      setShowUserProfile(false);
      setShowAnalysisHeader(false);
      setShowPostsList(false);
      setShowExecutiveSummary(false);

      // Step 1: Get user profile
      const userResponse = await apiService.getUserProfile(targetUsername);
      if (signal?.aborted) return;

      if (!userResponse.success) {
        throw new Error(userResponse.error?.message || 'Failed to load user profile');
      }
      setUserProfile(userResponse.data!);

      // Start progressive reveal - User Profile appears first
      setTimeout(() => {
        if (!signal?.aborted) setShowUserProfile(true);
      }, 300);

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

      // Executive Summary shows up during analysis to show loading indicator
      setTimeout(() => {
        if (!signal?.aborted) setShowExecutiveSummary(true);
      }, 500);

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

      // Progressive reveal sequence
      setTimeout(() => {
        if (!signal?.aborted) setShowAnalysisHeader(true);
      }, 600);

      setTimeout(() => {
        if (!signal?.aborted) setShowPostsList(true);
      }, 900);

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
                // Start streaming key points after summary is complete
                if (assessmentResponse.data!.keyPoints && assessmentResponse.data!.keyPoints.length > 0) {
                  setIsStreamingKeyPoints(true);
                }
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

  // Key points streaming effect
  useEffect(() => {
    if (isStreamingKeyPoints && assessment?.keyPoints && assessment.keyPoints.length > 0) {
      console.log('Starting key points streaming. Total points:', assessment.keyPoints.length);
      console.log('Key points to stream:', assessment.keyPoints);

      // Clear any existing streamed points
      setStreamedKeyPoints([]);

      // Stream points one by one with delays
      assessment.keyPoints.forEach((point, index) => {
        setTimeout(() => {
          console.log(`Adding point ${index + 1}:`, point);
          setStreamedKeyPoints(prev => {
            const newPoints = [...prev, point];
            console.log('Current streamed points:', newPoints);
            return newPoints;
          });

          // Stop streaming after the last point
          if (index === assessment.keyPoints.length - 1) {
            setTimeout(() => {
              console.log('All points streamed, stopping');
              setIsStreamingKeyPoints(false);
            }, 100);
          }
        }, index * 800);
      });
    }
  }, [isStreamingKeyPoints, assessment?.keyPoints]);

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
      setStreamedKeyPoints([]);
      setIsStreamingKeyPoints(false);
      setShowFindings(false);
      setFindingIndex(0);
      setSelectedPost(null);
      setIsAnalyzing(true);
      setError(null);

      // Reset animation states for new search
      setShowUserProfile(false);
      setShowAnalysisHeader(false);
      setShowPostsList(false);
      setShowExecutiveSummary(false);

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
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-300">Loading user profile...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <Card className="p-6 max-w-md mx-auto bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
          <div className="text-center">
            <AlertTriangle className="w-12 h-12 text-red-500 dark:text-red-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Analysis Error</h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">{error}</p>
            <Button
              onClick={onBack}
              variant="outline"
              className="border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
            >
              Back to Landing
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-3">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Button
                variant="ghost"
                onClick={onBack}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
              >
                <ArrowLeft className="w-4 h-4 text-gray-700 dark:text-gray-300" />
              </Button>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent">Beacon</h1>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <Button
                variant="ghost"
                onClick={toggleTheme}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
              >
                {theme === 'light' ? (
                  <Moon className="w-4 h-4 text-gray-700 dark:text-gray-300" />
                ) : (
                  <Sun className="w-4 h-4 text-gray-700 dark:text-gray-300" />
                )}
              </Button>

              {isRateLimited && (
                <div className="flex items-center space-x-2 px-3 py-2 bg-orange-50 dark:bg-orange-900/30 border border-orange-200 dark:border-orange-700 rounded-lg">
                  <Clock className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                  <span className="text-sm font-medium text-orange-800 dark:text-orange-300">
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
                  className="w-48 h-9 px-3 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={isRateLimited}
                />
                <Button
                  type="submit"
                  className="px-4 h-9 bg-gray-900 hover:bg-gray-800 dark:bg-gray-100 dark:hover:bg-gray-200 text-white dark:text-gray-900 rounded-lg text-sm disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed"
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
      <main className="max-w-7xl mx-auto px-6 py-4 pb-12">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column - User Profile & Posts */}
          <div className="space-y-6">
            {/* Loading skeleton for user profile */}
            {!showUserProfile && userProfile && (
              <motion.div
                initial={{ opacity: 1 }}
                animate={{ opacity: 1 }}
                className="h-32 bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700 rounded-lg animate-pulse"
              />
            )}
            {/* User Profile Card */}
            {userProfile && showUserProfile && (
              <motion.div
                initial={{ opacity: 0, y: 30, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
              >
                <Card className="p-4 bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg relative overflow-hidden">
                {/* Background decoration */}
                <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-bl from-gray-200/30 dark:from-gray-700/20 to-transparent rounded-full -translate-y-10 translate-x-10"></div>
                <div className="absolute bottom-0 left-0 w-16 h-16 bg-gradient-to-tr from-gray-200/20 dark:from-gray-700/10 to-transparent rounded-full translate-y-8 -translate-x-8"></div>

                <div className="relative z-10 space-y-4">
                  {/* Top section: Picture and Name */}
                  <div className="flex items-start space-x-3">
                    <div className="relative">
                      <ImageWithFallback
                        src={userProfile.profileImageUrl}
                        alt="User profile"
                        className="w-12 h-12 rounded-full object-cover border-2 border-white shadow-sm"
                      />
                      <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white dark:border-gray-800"></div>
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">{userProfile.displayName}</h3>
                      <p className="text-gray-700 dark:text-gray-300 -mt-1">@{userProfile.username}</p>
                      {userProfile.nickname && userProfile.nickname !== userProfile.username && (
                        <p className="text-gray-600 dark:text-gray-400 text-sm font-medium -mt-1">{userProfile.nickname}</p>
                      )}
                    </div>
                  </div>

                  {/* Bio */}
                  <div className="space-y-3">
                    <p className="text-gray-800 dark:text-gray-200 text-sm font-medium">{userProfile.bio}</p>

                    {/* Location and Join Date */}
                    <div className="flex items-center space-x-4 text-sm text-gray-800 dark:text-gray-300">
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

                    {/* Stats */}
                    <div className="flex items-center space-x-6 text-sm">
                      <span><strong className="text-gray-900 dark:text-gray-100">{userProfile.karma.toLocaleString()}</strong> <span className="text-gray-700 dark:text-gray-300">Karma</span></span>
                      <span><strong className="text-gray-900 dark:text-gray-100">{postCount !== null ? postCount.toLocaleString() : '...'}</strong> <span className="text-gray-700 dark:text-gray-300">Posts</span></span>
                      <span><strong className="text-gray-900 dark:text-gray-100">{commentCount !== null ? commentCount.toLocaleString() : '...'}</strong> <span className="text-gray-700 dark:text-gray-300">Comments</span></span>
                    </div>
                  </div>
                </div>
              </Card>
              </motion.div>
            )}

            {/* Relevant Posts */}
            {assessment && showPostsList && (
              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.7, ease: "easeOut" }}
              >
                <Card className="bg-gradient-to-br from-slate-50 to-gray-100 dark:from-slate-800/40 dark:to-gray-800/30 border border-slate-200 dark:border-slate-700 rounded-lg relative overflow-hidden">
                {/* Background decoration */}
                <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-bl from-slate-200/30 dark:from-slate-600/20 to-transparent rounded-full -translate-y-16 translate-x-16"></div>

                <div className="relative z-10">
                  <div className="p-6 border-b border-slate-200 dark:border-slate-700">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center justify-center w-8 h-8 bg-slate-600 dark:bg-slate-500 rounded-lg">
                        <Users className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Relevant Posts</h3>
                        <p className="text-slate-700 dark:text-slate-300 font-medium">{assessment.posts.length} posts analyzed</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div
                  ref={scrollContainerRef}
                  className="h-[440px] overflow-y-auto overflow-x-hidden cursor-pointer no-scrollbar"
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
                      <motion.div
                        key={`${post.id}-${index}`}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4, delay: index * 0.1, ease: "easeOut" }}
                        className="p-4 border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors cursor-pointer min-h-[120px]"
                        onClick={(e) => {
                          e.stopPropagation();
                          handlePostClick(post);
                        }}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <span className="text-sm font-medium text-gray-500 dark:text-gray-400">#{post.rank}</span>
                            <Badge className={`${post.tagColor} text-white text-xs px-2 py-1 rounded`}>
                              {post.tag}
                            </Badge>
                          </div>
                          <span className="text-sm text-gray-500 dark:text-gray-400">{post.timestamp}</span>
                        </div>

                        <div className="mb-2">
                          <span className="text-xs text-blue-600 dark:text-blue-400 font-medium">{post.subreddit}</span>
                          {post.title && (
                            <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mt-1 break-words">{post.title}</h4>
                          )}
                        </div>

                        <p className="text-gray-700 dark:text-gray-300 text-sm mb-3 break-words">{post.content}</p>

                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4 text-gray-500 dark:text-gray-400">
                            <div className="flex items-center space-x-1">
                              <ArrowUp className="w-4 h-4 text-orange-500 dark:text-orange-400" />
                              <span className="text-xs font-medium">{post.score}</span>
                              <ArrowDown className="w-4 h-4" />
                            </div>
                            <div className="flex items-center space-x-1">
                              <MessageCircle className="w-4 h-4" />
                              <span className="text-xs">{post.comments}</span>
                            </div>
                          </div>
                          <span className="text-xs text-blue-600 dark:text-blue-400 font-medium">
                            Relevance: {post.relevanceScore}%
                          </span>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </Card>
              </motion.div>
            )}

            
        {/* Mental Health Category Averages */}
        {assessment?.posts && assessment.posts.length > 0 && assessment.posts.some(post => post.mentalHealthScore) && (
          <motion.div
            ref={categoryAnalysisRef}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.9 }}
            className="mt-6"
          >
            <div className="bg-gradient-to-br from-slate-50 to-gray-100 dark:from-slate-800/40 dark:to-gray-800/30 rounded-2xl border border-slate-200 dark:border-slate-700 p-6 shadow-lg">
              <div className="flex items-center space-x-3 mb-5">
                <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-2.5 rounded-xl shadow-md">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Category Risk Analysis</h3>
                  <p className="text-slate-600 dark:text-slate-400 text-sm">Average scores across posts</p>
                </div>
              </div>

              <div className="space-y-3">
                {(() => {
                  const postsWithScores = assessment.posts.filter(p => p.mentalHealthScore);
                  if (postsWithScores.length === 0) return null;

                  const categories = ['depression', 'anxiety', 'ptsd', 'schizophrenia', 'bipolar', 'eating_disorder', 'adhd'];
                  const averages = categories.map(category => {
                    const sum = postsWithScores.reduce((acc, post) => acc + post.mentalHealthScore![category as keyof typeof post.mentalHealthScore], 0);
                    const avg = sum / postsWithScores.length;
                    let label = category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                    // Hardcode special cases for capitalization
                    if (category === 'adhd') label = 'ADHD';
                    if (category === 'ptsd') label = 'PTSD';

                    return {
                      label: label,
                      value: avg,
                      percentage: Math.round(avg * 100)
                    };
                  });

                  // Sort by value for better visualization
                  averages.sort((a, b) => b.value - a.value);

                  return (
                    <div>
                      {averages.map(({ label, value, percentage }, index) => (
                        <div key={`${label}-${percentage}`} className="space-y-1.5 mb-3">
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">{label}</span>
                            <span className="text-sm font-semibold text-slate-600 dark:text-slate-400">{percentage}%</span>
                          </div>
                          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2 overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${percentage}%` }}
                              transition={{ duration: 1, delay: 1.2 + (0.1 * index), ease: "easeOut" }}
                              className={`h-2 rounded-full ${
                                value < 0.3 ? 'bg-gradient-to-r from-emerald-400 to-emerald-500' :
                                value < 0.6 ? 'bg-gradient-to-r from-amber-400 to-amber-500' :
                                'bg-gradient-to-r from-red-400 to-red-500'
                              }`}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  );
                })()}
              </div>
            </div>
          </motion.div>
        )}
          </div>

          {/* Right Column - AI Analysis */}
          <div className="space-y-6">
            {/* Loading skeletons for right column */}
            {!showAnalysisHeader && assessment && (
              <motion.div
                initial={{ opacity: 1 }}
                animate={{ opacity: 1 }}
                className="h-40 bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/20 rounded-lg animate-pulse"
              />
            )}

            {/* Analysis Header - Redesigned */}
            {assessment && showAnalysisHeader && (
              <motion.div
                initial={{ opacity: 0, x: 50, scale: 0.95 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
              >
                <Card className="p-0 bg-gradient-to-br from-purple-50 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/20 border border-purple-200 dark:border-purple-800 rounded-xl relative overflow-hidden">
                  {/* Background decoration */}
                  <div className="absolute top-0 right-0 w-28 h-28 bg-gradient-to-bl from-purple-200/30 dark:from-purple-700/20 to-transparent rounded-full -translate-y-14 translate-x-14"></div>
                  <div className="absolute bottom-0 left-0 w-20 h-20 bg-gradient-to-tr from-pink-200/20 dark:from-pink-700/10 to-transparent rounded-full translate-y-10 -translate-x-10"></div>

                  <div className="relative z-10">
                    {/* Header Section */}
                    <div className="p-6 pb-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="flex items-center justify-center w-12 h-12 bg-purple-600 dark:bg-purple-500 rounded-xl shadow-lg">
                            <Brain className="w-7 h-7 text-white" />
                          </div>
                          <div>
                            <h3 className="text-xl font-bold text-purple-900 dark:text-purple-100">Mental Health Analysis</h3>
                            <p className="text-sm text-purple-700 dark:text-purple-300 font-medium">AI-Powered Risk Assessment</p>
                          </div>
                        </div>
                        <Badge className={`${getRiskLevelColor(assessment.overallRiskLevel)} px-4 py-2 rounded-xl text-sm font-bold shadow-lg`}>
                          {assessment.overallRiskLevel} RISK
                        </Badge>
                      </div>
                    </div>

                    {/* Risk Level Visual Indicator */}
                    <div className="px-6 pb-4">
                      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-xl p-4 border border-white/60 dark:border-gray-700/60 shadow-sm">
                        <div className="text-center mb-4">
                          <div className="text-3xl font-black text-purple-900 dark:text-purple-100">
                            {assessment.mental_health_score}%
                          </div>
                          <div className="text-sm text-purple-700 dark:text-purple-300 font-semibold">
                            Mental Health Risk Score
                          </div>
                        </div>

                        {/* Visual Risk Bar */}
                        <div className="relative">
                          <div className="h-3 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                            <div
                              className={`h-full transition-all duration-1000 ease-out ${
                                assessment.mental_health_score >= 80 ? 'bg-red-500' :
                                assessment.mental_health_score >= 60 ? 'bg-orange-500' :
                                assessment.mental_health_score >= 40 ? 'bg-yellow-500' : 'bg-green-500'
                              }`}
                              style={{ width: `${assessment.mental_health_score}%` }}
                            />
                          </div>

                        </div>
                      </div>
                    </div>

                    {/* Metrics Section */}
                    <div className="px-6 pb-6">
                        {/* Confidence Score */}
                        <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg p-4 border border-white/50 dark:border-gray-700/50 flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                            <span className="text-sm font-semibold text-purple-800 dark:text-purple-300">Analysis Confidence</span>
                          </div>
                          <div className="flex items-center space-x-3">
                            <div className="w-32">
                              <Progress value={assessment.confidenceScore} className="h-2" />
                            </div>
                            <div className="text-xl font-bold text-purple-900 dark:text-purple-100">
                              {assessment.confidenceScore}%
                            </div>
                          </div>
                        </div>
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}


            {/* Mental Health Evaluation Panel - Redesigned */}
            {assessment && showAnalysisHeader && (
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, ease: "easeOut", delay: 0.3 }}
              >
                {assessment.overallRiskLevel === 'LOW' && (
                  <Card className="p-6 bg-gradient-to-br from-green-50 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/20 border border-green-200 dark:border-green-700 rounded-xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-bl from-green-200/30 dark:from-green-600/20 to-transparent rounded-full -translate-y-10 translate-x-10"></div>

                    <div className="relative z-10">
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="flex items-center justify-center w-12 h-12 bg-green-600 rounded-xl shadow-lg">
                          <CheckCircle className="w-7 h-7 text-white" />
                        </div>
                        <div>
                          <h4 className="font-bold text-green-900 dark:text-green-100 text-lg">
                            Positive Mental Health Status
                          </h4>
                          <p className="text-green-700 dark:text-green-300 text-sm font-medium">
                            Low risk indicators detected
                          </p>
                        </div>
                      </div>

                      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-4 border border-white/60 dark:border-gray-700/60">
                        <p className="text-green-800 dark:text-green-200 font-medium">
                          Analysis shows healthy communication patterns and positive engagement behaviors. Continue monitoring and maintain current support systems.
                        </p>
                      </div>
                    </div>
                  </Card>
                )}

                {assessment.overallRiskLevel === 'MODERATE' && (
                  <Card className="p-6 bg-gradient-to-br from-yellow-50 to-orange-100 dark:from-yellow-900/30 dark:to-orange-900/20 border border-yellow-200 dark:border-yellow-700 rounded-xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-bl from-yellow-200/30 dark:from-yellow-600/20 to-transparent rounded-full -translate-y-10 translate-x-10"></div>

                    <div className="relative z-10">
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="flex items-center justify-center w-12 h-12 bg-yellow-600 rounded-xl shadow-lg">
                          <AlertTriangle className="w-7 h-7 text-white" />
                        </div>
                        <div>
                          <h4 className="font-bold text-yellow-900 dark:text-yellow-100 text-lg">
                            Monitoring Recommended
                          </h4>
                          <p className="text-yellow-700 dark:text-yellow-300 text-sm font-medium">
                            Moderate risk indicators present
                          </p>
                        </div>
                      </div>

                      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-4 border border-white/60 dark:border-gray-700/60">
                        <p className="text-yellow-800 dark:text-yellow-200 font-medium">
                          Some concerning patterns detected. Consider periodic check-ins and continued observation. Preventive interventions may be beneficial.
                        </p>
                      </div>
                    </div>
                  </Card>
                )}

                {(assessment.overallRiskLevel === 'HIGH' || assessment.overallRiskLevel === 'CRITICAL') && (
                  <Card className="p-6 bg-gradient-to-br from-red-50 to-rose-100 dark:from-red-900/30 dark:to-rose-900/20 border border-red-200 dark:border-red-700 rounded-xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-bl from-red-200/30 dark:from-red-600/20 to-transparent rounded-full -translate-y-10 translate-x-10"></div>

                    <div className="relative z-10">
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="flex items-center justify-center w-12 h-12 bg-red-600 rounded-xl shadow-lg">
                          <AlertTriangle className="w-7 h-7 text-white" />
                        </div>
                        <div>
                          <h4 className="font-bold text-red-900 dark:text-red-100 text-lg">
                            {assessment.overallRiskLevel === 'CRITICAL' ? 'Immediate Support Required' : 'Clinical Assessment Needed'}
                          </h4>
                          <p className="text-red-700 dark:text-red-300 text-sm font-medium">
                            {assessment.overallRiskLevel === 'CRITICAL' ? 'Critical risk indicators' : 'High risk indicators detected'}
                          </p>
                        </div>
                      </div>

                      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-4 border border-white/60 dark:border-gray-700/60">
                        <p className="text-red-800 dark:text-red-200 font-medium mb-3">
                          {assessment.overallRiskLevel === 'CRITICAL'
                            ? 'Critical indicators detected. Immediate mental health professional consultation strongly advised.'
                            : 'Multiple risk factors identified. Professional mental health evaluation recommended.'
                          }
                        </p>

                        {assessment.overallRiskLevel === 'CRITICAL' && (
                          <div className="bg-red-100 dark:bg-red-900/40 rounded-lg p-3 border border-red-200 dark:border-red-700">
                            <p className="text-red-900 dark:text-red-100 text-sm font-bold">
                              🚨 Consider immediate intervention or crisis support resources
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </Card>
                )}
              </motion.div>
            )}

            {/* Executive Summary */}
            {(isAnalyzing || (assessment && showExecutiveSummary)) && (
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: 0.6,
                  ease: "easeOut",
                  delay: isAnalyzing ? 0 : 0.2
                }}
              >
                <Card className="p-6 bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-900/30 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 rounded-lg relative overflow-hidden">
              {/* Background decoration */}
              <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-bl from-blue-200/30 dark:from-blue-700/20 to-transparent rounded-full -translate-y-16 translate-x-16"></div>
              <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-indigo-200/20 dark:from-indigo-700/10 to-transparent rounded-full translate-y-12 -translate-x-12"></div>

              <div className="relative z-10">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="flex items-center justify-center w-8 h-8 bg-blue-600 dark:bg-blue-500 rounded-lg">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <h4 className="font-semibold text-blue-900 dark:text-blue-100 text-lg">Executive Summary</h4>
                </div>

                {isAnalyzing ? (
                  <div className="flex items-center space-x-3">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 dark:border-blue-400"></div>
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
                          <span className="text-blue-700 dark:text-blue-300 text-sm font-medium">
                            {loadingTexts[loadingTextIndex]}
                          </span>
                        </motion.div>
                      </AnimatePresence>
                    </div>
                  </div>
                ) : (
                  <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg p-4 border border-white/50 dark:border-gray-700/50">
                    <p className="text-gray-800 dark:text-gray-200 text-base leading-relaxed font-medium">
                      {streamedText}
                      {aiReport && streamedText.length < aiReport.executiveSummary.length && (
                        <span className="animate-pulse text-blue-600 dark:text-blue-400">|</span>
                      )}
                    </p>
                  </div>
                )}
              </div>
            </Card>
              </motion.div>
            )}


            {/* Key Points */}
            {assessment && assessment.keyPoints && assessment.keyPoints.length > 0 && (streamedKeyPoints.length > 0 || isStreamingKeyPoints) && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              >
                <Card className="p-6 bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-emerald-900/30 dark:to-teal-900/20 border border-emerald-200 dark:border-emerald-800 rounded-lg relative overflow-hidden">
                {/* Background decoration */}
                <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-emerald-200/30 dark:from-emerald-700/20 to-transparent rounded-full -translate-y-12 translate-x-12"></div>
                <div className="absolute bottom-0 left-0 w-16 h-16 bg-gradient-to-tr from-teal-200/20 dark:from-teal-700/10 to-transparent rounded-full translate-y-8 -translate-x-8"></div>

                <div className="relative z-10">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="flex items-center justify-center w-8 h-8 bg-emerald-600 dark:bg-emerald-500 rounded-lg">
                      <Activity className="w-5 h-5 text-white" />
                    </div>
                    <h4 className="font-semibold text-emerald-900 dark:text-emerald-100 text-lg">Key Insights</h4>
                  </div>

                  <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg p-4 border border-white/50 dark:border-gray-700/50">
                    <div className="space-y-3">
                      {streamedKeyPoints.map((point, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.5, ease: "easeOut" }}
                          className="flex items-start space-x-3"
                        >
                          <div className="w-2 h-2 bg-emerald-600 dark:bg-emerald-400 rounded-full mt-2 flex-shrink-0"></div>
                          <p className="text-gray-800 dark:text-gray-200 text-sm font-medium">{point}</p>
                        </motion.div>
                      ))}
                      {isStreamingKeyPoints && streamedKeyPoints.length < assessment.keyPoints.length && (
                        <div className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-emerald-400 dark:bg-emerald-300 rounded-full mt-2 flex-shrink-0 animate-pulse"></div>
                          <p className="text-emerald-700 dark:text-emerald-300 text-sm font-medium italic">Loading insights...</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
              </motion.div>
            )}
          </div>
        </div>

      </main>

      {/* Post Modal */}
      {selectedPost && (
        <Dialog open={true} onOpenChange={handleCloseModal}>
          <DialogContent className="!max-w-3xl !w-[90vw] mx-auto max-h-[90vh] overflow-y-auto bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
            <DialogHeader>
              <DialogTitle className="text-gray-900 dark:text-gray-100">Post Analysis</DialogTitle>
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
                    <h4 className="font-semibold text-gray-900 dark:text-gray-100">{userProfile?.displayName}</h4>
                    <p className="text-gray-600 dark:text-gray-400 text-sm">u/{userProfile?.username} • {selectedPost.timestamp}</p>
                    {userProfile?.nickname && userProfile.nickname !== userProfile.username && (
                      <p className="text-gray-500 dark:text-gray-500 text-xs">{userProfile.nickname}</p>
                    )}
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium text-gray-500 dark:text-gray-400">#{selectedPost.rank}</span>
                  <Badge className={`${selectedPost.tagColor} text-white text-sm px-3 py-1 rounded-full`}>
                    {selectedPost.tag}
                  </Badge>
                </div>
              </div>

              {/* Post Content */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
                <div className="mb-3">
                  <span className="text-sm text-blue-600 dark:text-blue-400 font-medium">{selectedPost.subreddit}</span>
                  {selectedPost.title && (
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mt-1">{selectedPost.title}</h3>
                  )}
                </div>
                <p className="text-gray-800 dark:text-gray-200 leading-relaxed text-base">{selectedPost.content}</p>
              </div>

              {/* Post Stats */}
              <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-6 text-gray-600 dark:text-gray-300">
                  <div className="flex items-center space-x-2">
                    <ArrowUp className="w-5 h-5 text-orange-500 dark:text-orange-400" />
                    <span className="text-sm font-medium">{selectedPost.score}</span>
                    <ArrowDown className="w-5 h-5" />
                  </div>
                  <div className="flex items-center space-x-2">
                    <MessageCircle className="w-5 h-5" />
                    <span className="text-sm font-medium">{selectedPost.comments}</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-600 dark:text-gray-300">Relevance Score</div>
                  <div className="text-lg font-bold text-blue-600 dark:text-blue-400">{selectedPost.relevanceScore}%</div>
                </div>
              </div>

              {/* Analysis Insights */}
              <div className="space-y-4">
                <h5 className="font-semibold text-gray-900 dark:text-gray-100">AI Analysis Insights</h5>

                {/* Mental Health Score Bars */}
                {selectedPost.mentalHealthScore && (
                  <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/30 dark:to-purple-900/30 border border-blue-200 dark:border-blue-700 rounded-lg p-4">
                    <div className="flex items-start space-x-3 mb-4">
                      <Brain className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                      <div>
                        <h6 className="font-medium text-blue-900 dark:text-blue-100 mb-1">Mental Health Assessment</h6>
                        <p className="text-xs text-blue-700 dark:text-blue-300">AI-powered analysis of potential mental health indicators</p>
                      </div>
                    </div>

                    <div className="space-y-3">
                      {Object.entries(selectedPost.mentalHealthScore).map(([key, value], index) => {
                        if (key === 'overall_score') return null;
                        let label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                        // Hardcode special cases for capitalization
                        if (key === 'adhd') label = 'ADHD';
                        if (key === 'ptsd') label = 'PTSD';

                        const percentage = Math.round(value * 100);
                        const getBarColor = (score: number) => {
                          if (score < 0.3) return 'bg-gradient-to-r from-green-400 to-green-500';
                          if (score < 0.6) return 'bg-gradient-to-r from-yellow-400 to-yellow-500';
                          return 'bg-gradient-to-r from-red-400 to-red-500';
                        };

                        return (
                          <div key={key} className="space-y-1">
                            <div className="flex justify-between items-center">
                              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</span>
                              <span className="text-xs text-gray-600 dark:text-gray-400">{percentage}%</span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${percentage}%` }}
                                transition={{ duration: 0.8, delay: 0.1 * index, ease: "easeOut" }}
                                className={`h-2 rounded-full ${getBarColor(value)}`}
                              />
                            </div>
                          </div>
                        );
                      })}

                      {/* Overall Score */}
                      <div className="pt-2 mt-3 border-t border-blue-200 dark:border-blue-700">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm font-semibold text-blue-900 dark:text-blue-100">Overall Risk Score</span>
                          <span className="text-sm font-bold text-blue-800 dark:text-blue-200">{Math.round(selectedPost.mentalHealthScore.overall_score * 100)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.round(selectedPost.mentalHealthScore.overall_score * 100)}%` }}
                            transition={{ duration: 1, delay: 0.8, ease: "easeOut" }}
                            className={`h-3 rounded-full ${
                              selectedPost.mentalHealthScore.overall_score < 0.3 ? 'bg-gradient-to-r from-green-400 to-green-500' :
                              selectedPost.mentalHealthScore.overall_score < 0.6 ? 'bg-gradient-to-r from-yellow-400 to-yellow-500' :
                              'bg-gradient-to-r from-red-400 to-red-500'
                            }`}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Indicators */}
                <div className="bg-orange-50 dark:bg-orange-900/30 border border-orange-200 dark:border-orange-700 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <AlertTriangle className="w-5 h-5 text-orange-600 dark:text-orange-400 mt-0.5" />
                    <div>
                      <h6 className="font-medium text-orange-900 dark:text-orange-100 mb-2">Detected Indicators</h6>
                      <div className="space-y-1 text-sm text-orange-800 dark:text-orange-200">
                        {selectedPost.concerns?.map((concern, index) => (
                          <p key={index}>• {concern.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                        ))}
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
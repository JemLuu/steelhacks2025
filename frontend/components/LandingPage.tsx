import React, { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Brain, BarChart3, Shield, Star, Users, TrendingUp, Eye, Menu, X, Moon, Sun } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card } from './ui/card';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface LandingPageProps {
  onAnalyze: (handle: string) => void;
}

export default function LandingPage({ onAnalyze }: LandingPageProps) {
  const { theme, toggleTheme } = useTheme();
  const [heroHandle, setHeroHandle] = useState('');
  const [ctaHandle, setCtaHandle] = useState('');
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleHeroSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (heroHandle.trim()) {
      onAnalyze(heroHandle.trim());
    }
  };

  const handleCtaSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (ctaHandle.trim()) {
      onAnalyze(ctaHandle.trim());
    }
  };

  const features = [
    {
      icon: Brain,
      title: "Real-time Analysis",
      description: "AI-powered analysis of social media patterns for immediate mental health insights"
    },
    {
      icon: BarChart3,
      title: "Clinical Insights",
      description: "Evidence-based reporting that integrates with existing clinical workflows"
    },
    {
      icon: Shield,
      title: "Evidence-Based",
      description: "Built on peer-reviewed research and validated psychological frameworks"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-gray-50 dark:from-slate-900 dark:via-purple-900 dark:to-slate-900">
      {/* Animated Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute top-20 left-20 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, -50, 0],
          }}
          transition={{
            duration: 30,
            repeat: Infinity,
            ease: "easeInOut",
            type: "tween"
          }}
          style={{ willChange: 'transform' }}
        />
        <motion.div
          className="absolute top-1/2 right-20 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"
          animate={{
            x: [0, -80, 0],
            y: [0, 60, 0],
          }}
          transition={{
            duration: 35,
            repeat: Infinity,
            ease: "easeInOut",
            type: "tween"
          }}
          style={{ willChange: 'transform' }}
        />
        <motion.div
          className="absolute bottom-20 left-1/3 w-64 h-64 bg-pink-500/10 rounded-full blur-3xl"
          animate={{
            x: [0, 120, 0],
            y: [0, -40, 0],
          }}
          transition={{
            duration: 40,
            repeat: Infinity,
            ease: "easeInOut",
            type: "tween"
          }}
          style={{ willChange: 'transform' }}
        />
      </div>

      {/* Header */}
      <motion.header
        className="fixed top-0 left-0 right-0 z-50 transition-all duration-300"
        style={{
          backgroundColor: scrollY > 50 ? (theme === 'dark' ? 'rgba(15, 23, 42, 0.9)' : 'rgba(255, 255, 255, 0.9)') : 'transparent',
          backdropFilter: scrollY > 50 ? 'blur(20px)' : 'none'
        }}
      >
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent"
            >
              Mindful
            </motion.div>
            
            <div className="flex items-center space-x-6">
              <nav className="hidden md:flex items-center space-x-8">
                <a href="#home" className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors">Home</a>
                <a href="#about" className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors">About</a>
                <a href="#contact" className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors">Contact</a>
              </nav>

              <Button
                variant="ghost"
                onClick={toggleTheme}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
              >
                {theme === 'light' ? (
                  <Moon className="w-5 h-5 text-gray-700 dark:text-gray-300" />
                ) : (
                  <Sun className="w-5 h-5 text-gray-700 dark:text-gray-300" />
                )}
              </Button>

              <button
                className="md:hidden text-gray-700 dark:text-white"
                onClick={() => setIsMenuOpen(!isMenuOpen)}
              >
                {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
              </button>
            </div>
          </div>

          {/* Mobile Menu */}
          {isMenuOpen && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="md:hidden mt-4 pb-4"
            >
              <nav className="flex flex-col space-y-4">
                <a href="#home" className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors">Home</a>
                <a href="#about" className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors">About</a>
                <a href="#contact" className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors">Contact</a>
              </nav>
            </motion.div>
          )}
        </div>
      </motion.header>

      {/* Hero Section */}
      <section id="home" className="relative min-h-screen flex items-center justify-center px-6">
        <div className="container mx-auto text-center w-1/2 md:w-2/3 sm:w-3/4">
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-4xl md:text-7xl font-black mb-6 bg-gradient-to-r from-gray-900 via-blue-900 to-purple-900 dark:from-white dark:via-blue-100 dark:to-purple-100 bg-clip-text text-transparent leading-tight"
          >
            Predict Mental Health Risks Before They Happen
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-12 max-w-4xl mx-auto"
          >
            AI-powered social media analysis that gives mental health professionals unprecedented patient insights
          </motion.p>

          <motion.form
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            onSubmit={handleHeroSubmit}
            className="max-w-md mx-auto mb-8"
          >
            <div className="relative">
              <Input
                type="text"
                placeholder="@username"
                value={heroHandle}
                onChange={(e) => setHeroHandle(e.target.value)}
                className="w-full h-14 px-6 text-lg bg-gray-100/80 dark:bg-white/10 backdrop-blur-md border border-gray-300 dark:border-white/20 rounded-2xl text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500/50 focus:border-transparent"
              />
              <Button
                type="submit"
                className="absolute right-2 top-2 h-10 px-6 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 rounded-xl font-semibold text-white border-0 shadow-lg"
              >
                Analyze Now
              </Button>
            </div>
          </motion.form>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-sm text-gray-500 dark:text-gray-400"
          >
            For licensed healthcare providers only
          </motion.p>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6">
        <div className="container mx-auto">
          <motion.h2
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl md:text-5xl font-bold text-center mb-16 text-gray-900 dark:text-white"
          >
            Revolutionary AI Technology
          </motion.h2>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
              >
                <Card className="p-8 bg-white/80 dark:bg-white/5 backdrop-blur-md border border-gray-200 dark:border-white/10 hover:bg-white/90 dark:hover:bg-white/10 transition-all duration-300">
                  <div className="flex items-center justify-center w-16 h-16 mb-6 mx-auto bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl">
                    <feature.icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 text-center">{feature.title}</h3>
                  <p className="text-gray-600 dark:text-gray-300 text-center">{feature.description}</p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>


      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="container mx-auto text-center">
          <motion.h2
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl md:text-5xl font-bold mb-6 text-gray-900 dark:text-white"
          >
            Ready to Transform Patient Care?
          </motion.h2>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-xl text-gray-600 dark:text-gray-300 mb-12 max-w-2xl mx-auto"
          >
            Start your free analysis today and discover what social media reveals about mental health
          </motion.p>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.6 }}
            className="space-y-4"
          >
            <Button
              variant="outline"
              className="mr-4 px-8 py-3 bg-transparent border-2 border-gray-300 dark:border-white/30 text-gray-700 dark:text-white hover:bg-gray-100 dark:hover:bg-white/10 rounded-xl"
            >
              Book a Demo
            </Button>
            <p className="text-gray-500 dark:text-gray-400">Start your free analysis today</p>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-gray-200 dark:border-white/10">
        <div className="container mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <div className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent mb-2">
                Mindful
              </div>
              <p className="text-gray-500 dark:text-gray-400 text-sm max-w-md">
                Professional mental health screening tool for licensed healthcare providers.
                Not intended for diagnostic purposes.
              </p>
            </div>

            <div className="flex items-center space-x-6">
              <a href="#privacy" className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-white transition-colors text-sm">Privacy Policy</a>
              <a href="#terms" className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-white transition-colors text-sm">Terms of Service</a>
              <a href="#contact" className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-white transition-colors text-sm">Contact</a>
            </div>
          </div>

          <div className="mt-8 pt-8 border-t border-gray-200 dark:border-white/10 text-center text-gray-500 dark:text-gray-400 text-sm">
            © 2025 Mindful. All rights reserved. For professional medical use only.
          </div>
        </div>
      </footer>
    </div>
  );
}
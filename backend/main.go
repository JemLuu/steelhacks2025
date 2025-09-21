package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/joho/godotenv"
)

const postLimit = 1000
const commentLimit = 1000

// ----- API response types -----

type APIProfileResponse struct {
	Username   string    `json:"username"`
	IconURL    string    `json:"icon_url"`
	TotalKarma int       `json:"total_karma"`
	Bio        string    `json:"bio"`
	CakeDay    time.Time `json:"cake_day"`
}

type APIPostCountResponse struct {
	PostCount int `json:"post_count"`
}

type EnrichedItem struct {
	Type           string    `json:"type"` // "post" | "comment"
	Subreddit      string    `json:"subreddit"`
	Permalink      string    `json:"permalink"`
	Title          string    `json:"title"`           // post title (empty for comments)
	Content        string    `json:"content"`         // post selftext or comment body
	CreatedAt      time.Time `json:"created_at"`      // when the post/comment was created
	Score          MHScore   `json:"score"`           // external classifier scores
	Indicators     []string  `json:"indicators"`      // from Claude
	RelevanceScore float64   `json:"relevance_score"` // from Claude
}

type APIAssessmentResponse struct {
	ExecutiveSummary  string         `json:"executive_summary"`
	ConfidenceScore   float64        `json:"confidence_score"`
	MentalHealthScore float64        `json:"mental_health_score"`
	Items             []EnrichedItem `json:"items"`
}

func main() {
	// Load environment variables from .env file
	root, err := findProjectRoot()
	if err != nil {
		log.Printf("Warning: could not find project root: %v", err)
	} else {
		envPath := filepath.Join(root, ".env")
		if err := godotenv.Overload(envPath); err != nil {
			log.Printf("Warning: Could not load .env file from %s: %v", envPath, err)
			log.Println("Falling back to system environment variables")
		} else {
			log.Printf("Successfully loaded .env file from %s", envPath)
		}
	}

	mux := http.NewServeMux()

	// health check
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	// profile endpoint: /api/reddit/profile/{username}
	mux.HandleFunc("/api/reddit/profile/", func(w http.ResponseWriter, r *http.Request) {
		username := r.URL.Path[len("/api/reddit/profile/"):]
		if username == "" {
			http.Error(w, "username required", http.StatusBadRequest)
			return
		}
		handleProfile(w, r, username)
	})

	// post count endpoint: /api/reddit/postcount/{username}
	mux.HandleFunc("/api/reddit/postcount/", func(w http.ResponseWriter, r *http.Request) {
		username := r.URL.Path[len("/api/reddit/postcount/"):]
		if username == "" {
			http.Error(w, "username required", http.StatusBadRequest)
			return
		}
		handlePostCount(w, r, username)
	})

	// assessment endpoint: /api/reddit/assessment/{username}
	mux.HandleFunc("/api/reddit/assessment/", func(w http.ResponseWriter, r *http.Request) {
		username := r.URL.Path[len("/api/reddit/assessment/"):]
		if username == "" {
			http.Error(w, "username required", http.StatusBadRequest)
			return
		}
		handleAssessment(w, r, username)
	})

	// server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	addr := ":" + port
	fmt.Printf("🚀 listening on http://localhost%s\n", addr)
	if err := http.ListenAndServe(addr, withCORS(mux)); err != nil {
		panic(err)
	}
}

// ----- handlers -----

func handleProfile(w http.ResponseWriter, r *http.Request, username string) {
	ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
	defer cancel()

	prof, err := GetRedditUserProfile(ctx, username)
	if err != nil {
		http.Error(w, "failed to fetch profile: "+err.Error(), http.StatusBadGateway)
		return
	}

	out := APIProfileResponse{
		Username:   prof.Username,
		IconURL:    prof.IconURL,
		TotalKarma: prof.TotalKarma,
		Bio:        prof.Bio,
		CakeDay:    prof.CakeDay,
	}
	writeJSON(w, http.StatusOK, out)
}

func handlePostCount(w http.ResponseWriter, r *http.Request, username string) {
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	posts, err := GetRedditUserPosts(ctx, username, postLimit, commentLimit)
	if err != nil {
		http.Error(w, "failed to fetch posts: "+err.Error(), http.StatusBadGateway)
		return
	}

	out := APIPostCountResponse{
		PostCount: len(posts.Posts),
	}
	writeJSON(w, http.StatusOK, out)
}

func handleAssessment(w http.ResponseWriter, r *http.Request, username string) {
	ctx, cancel := context.WithTimeout(r.Context(), 500*time.Second)
	defer cancel()

	// Orchestrate: scrape -> predict per item -> Claude (permalinks only) -> enrich
	resp, err := OrchestrateAssessment(ctx, username, postLimit, commentLimit)
	if err != nil {
		http.Error(w, "assessment failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// ----- helpers -----

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// findProjectRoot searches for the project root directory by looking for a .git or .env file.
func findProjectRoot() (string, error) {
	currentDir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	dir := currentDir
	for {
		// Check for .git directory
		gitDir := filepath.Join(dir, ".git")
		if stat, err := os.Stat(gitDir); err == nil && stat.IsDir() {
			return dir, nil
		}

		// Check for .env file
		envFile := filepath.Join(dir, ".env")
		if stat, err := os.Stat(envFile); err == nil && !stat.IsDir() {
			return dir, nil
		}

		parentDir := filepath.Dir(dir)
		if parentDir == dir {
			// Reached the root of the filesystem
			break
		}
		dir = parentDir
	}

	return "", fmt.Errorf("project root not found (searched for .git directory or .env file)")
}

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"
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

type APIAssessmentResponse struct {
	Username         string           `json:"username"`
	Model            string           `json:"model"`
	ExecutiveSummary string           `json:"executive_summary"`
	ConfidenceScore  float64          `json:"confidence_score"`
	Items            []AssessmentItem `json:"items"`
}

func main() {
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

	// assessment endpoint: /api/reddit/assessment/{username}
	mux.HandleFunc("/api/reddit/assessment/", func(w http.ResponseWriter, r *http.Request) {
		username := r.URL.Path[len("/api/reddit/assessment/"):]
		if username == "" {
			http.Error(w, "username required", http.StatusBadRequest)
			return
		}
		handleAssessment(w, r, username)
	})

	// prediction endpoint: /api/reddit/predict/{username}
	mux.HandleFunc("/api/reddit/predict/", func(w http.ResponseWriter, r *http.Request) {
		username := r.URL.Path[len("/api/reddit/predict/"):]
		if username == "" {
			http.Error(w, "username required", http.StatusBadRequest)
			return
		}
		handlePredict(w, r, username)
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

func handleAssessment(w http.ResponseWriter, r *http.Request, username string) {
	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()

	content, err := GetRedditUserPosts(ctx, username, postLimit, commentLimit)
	if err != nil {
		http.Error(w, "failed to fetch posts/comments: "+err.Error(), http.StatusBadGateway)
		return
	}

	parsed, err := SendContentToClaude(ctx, content)
	if err != nil {
		http.Error(w, "Claude analysis failed: "+err.Error(), http.StatusBadGateway)
		return
	}

	model := os.Getenv("CLAUDE_MODEL")
	if model == "" {
		model = "claude-sonnet-4-20250514"
	}

	writeJSON(w, http.StatusOK, APIAssessmentResponse{
		Username:         username,
		Model:            model,
		ExecutiveSummary: parsed.ExecutiveSummary,
		ConfidenceScore:  parsed.ConfidenceScore,
		Items:            parsed.Items,
	})
}

func handlePredict(w http.ResponseWriter, r *http.Request, username string) {
	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()

	content, err := GetRedditUserPosts(ctx, username, postLimit, commentLimit)
	if err != nil {
		http.Error(w, "failed to fetch posts/comments: "+err.Error(), http.StatusBadGateway)
		return
	}

	items, err := PredictSequential(ctx, content)
	if err != nil {
		http.Error(w, "prediction failed: "+err.Error(), http.StatusBadGateway)
		return
	}

	writeJSON(w, http.StatusOK, struct {
		Username string           `json:"username"`
		Count    int              `json:"count"`
		Items    []ClassifiedItem `json:"items"`
	}{
		Username: username,
		Count:    len(items),
		Items:    items,
	})
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

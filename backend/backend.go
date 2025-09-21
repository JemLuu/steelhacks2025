package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

var httpClient = &http.Client{Timeout: 180 * time.Second}

// ---------- Reddit scraping ----------

type RedditAbout struct {
	Data struct {
		Name       string  `json:"name"`
		IconImg    string  `json:"icon_img"`
		TotalKarma int     `json:"total_karma"`
		CreatedUTC float64 `json:"created_utc"`
		Subreddit  struct {
			PublicDescription string `json:"public_description"` // bio
		} `json:"subreddit"`
	} `json:"data"`
}

type Listing struct {
	Data struct {
		Children []struct {
			Kind string `json:"kind"`
			Data struct {
				Subreddit   string  `json:"subreddit"`
				Title       string  `json:"title"`    // posts
				SelfText    string  `json:"selftext"` // posts text
				Body        string  `json:"body"`     // comments
				Permalink   string  `json:"permalink"`
				URL         string  `json:"url"`
				Score       int     `json:"score"`
				NumComments int     `json:"num_comments"`
				CreatedUTC  float64 `json:"created_utc"`
			} `json:"data"`
		}
	} `json:"data"`
}

type ProfileInformation struct {
	Username   string
	IconURL    string
	TotalKarma int
	Bio        string
	CakeDay    time.Time
}

type CommentsAndPosts struct {
	Posts    []RedditPost
	Comments []RedditComment
}

type RedditPost struct {
	Subreddit   string
	Title       string
	URL         string
	Permalink   string
	Score       int
	NumComments int
	CreatedAt   time.Time
	Content     string
}

type RedditComment struct {
	Subreddit string
	Body      string
	Permalink string
	Score     int
	CreatedAt time.Time
}

// ClaudeInput is what we send to Claude (we’ll compose a compact form later)
type ClaudeInput struct {
	// not used directly anymore, but kept for compatibility if needed
	Posts    []RedditPost    `json:"posts"`
	Comments []RedditComment `json:"comments"`
}

type TextBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ClaudeMessage struct {
	Role    string        `json:"role"`
	Content []interface{} `json:"content"`
}

type ClaudeRequest struct {
	Model     string          `json:"model"`
	MaxTokens int             `json:"max_tokens"`
	System    string          `json:"system"`
	Messages  []ClaudeMessage `json:"messages"`
}

type ClaudeResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
}

func doGET(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; GoRedditProfile/1.0; +https://example.com)")

	for attempt := 0; attempt < 3; attempt++ {
		resp, err := httpClient.Do(req)
		if err != nil {
			time.Sleep(time.Duration(1<<attempt) * 300 * time.Millisecond)
			continue
		}
		if resp.StatusCode == http.StatusTooManyRequests {
			ra := resp.Header.Get("Retry-After")
			_ = drainAndClose(resp.Body)
			if secs, err := strconv.Atoi(ra); err == nil && secs > 0 {
				time.Sleep(time.Duration(secs) * time.Second)
				continue
			}
			time.Sleep(2 * time.Second)
			continue
		}
		return resp, nil
	}
	return httpClient.Do(req)
}

func drainAndClose(rc io.ReadCloser) error {
	_, _ = io.Copy(io.Discard, rc)
	return rc.Close()
}

func GetRedditUserPosts(ctx context.Context, username string, postLimit, commentLimit int) (*CommentsAndPosts, error) {
	if postLimit <= 0 {
		postLimit = 10
	}
	if commentLimit <= 0 {
		commentLimit = 10
	}

	base := "https://www.reddit.com"
	submittedURL := fmt.Sprintf("%s/user/%s/submitted.json?limit=%d", base, username, postLimit)
	commentsURL := fmt.Sprintf("%s/user/%s/comments.json?limit=%d", base, username, commentLimit)

	out := &CommentsAndPosts{
		Posts:    []RedditPost{},
		Comments: []RedditComment{},
	}

	// Posts
	resp, err := doGET(ctx, submittedURL)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return out, fmt.Errorf("submitted.json: %s - %s", resp.Status, string(body))
	}
	var posts Listing
	if err := json.NewDecoder(resp.Body).Decode(&posts); err != nil {
		return out, err
	}
	for _, ch := range posts.Data.Children {
		d := ch.Data
		content := ""
		if s := strings.TrimSpace(d.SelfText); s != "" {
			if content != "" {
				content += "\n\n" + s
			} else {
				content = s
			}
		}
		out.Posts = append(out.Posts, RedditPost{
			Subreddit:   d.Subreddit,
			Title:       d.Title,
			URL:         d.URL,
			Permalink:   "https://www.reddit.com" + d.Permalink,
			Score:       d.Score,
			NumComments: d.NumComments,
			CreatedAt:   time.Unix(int64(d.CreatedUTC), 0).UTC(),
			Content:     content,
		})
	}

	// Comments
	resp, err = doGET(ctx, commentsURL)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return out, fmt.Errorf("comments.json: %s - %s", resp.Status, string(body))
	}
	var comments Listing
	if err := json.NewDecoder(resp.Body).Decode(&comments); err != nil {
		return out, err
	}
	for _, ch := range comments.Data.Children {
		d := ch.Data
		out.Comments = append(out.Comments, RedditComment{
			Subreddit: d.Subreddit,
			Body:      d.Body,
			Permalink: "https://www.reddit.com" + d.Permalink,
			Score:     d.Score,
			CreatedAt: time.Unix(int64(d.CreatedUTC), 0).UTC(),
		})
	}

	return out, nil
}

func GetRedditUserProfile(ctx context.Context, username string) (*ProfileInformation, error) {
	base := "https://www.reddit.com"
	aboutURL := fmt.Sprintf("%s/user/%s/about.json", base, username)

	resp, err := doGET(ctx, aboutURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("about.json: %s - %s", resp.Status, string(body))
	}
	var about RedditAbout
	if err := json.NewDecoder(resp.Body).Decode(&about); err != nil {
		return nil, err
	}

	return &ProfileInformation{
		Username:   about.Data.Name,
		IconURL:    about.Data.IconImg,
		TotalKarma: about.Data.TotalKarma,
		Bio:        about.Data.Subreddit.PublicDescription,
		CakeDay:    time.Unix(int64(about.Data.CreatedUTC), 0).UTC(),
	}, nil
}

// ---------- External mental-health model (single-call per item) ----------

func externalAPIBase() string {
	base := os.Getenv("MENTAL_API_BASE")
	if base == "" {
		base = "https://jluu196--mental-health-api-fastapi-app.modal.run"
	}
	return strings.TrimRight(base, "/")
}

type MHScore struct {
	Depression     float64 `json:"depression"`
	Anxiety        float64 `json:"anxiety"`
	PTSD           float64 `json:"ptsd"`
	Schizophrenia  float64 `json:"schizophrenia"`
	Bipolar        float64 `json:"bipolar"`
	EatingDisorder float64 `json:"eating_disorder"`
	ADHD           float64 `json:"adhd"`
	Overall        float64 `json:"overall_score"`
}

type ClassifiedItem struct {
	Type      string    `json:"type"` // "post" | "comment"
	Subreddit string    `json:"subreddit"`
	Permalink string    `json:"permalink"`
	Title     string    `json:"title"`      // post title (empty for comments)
	Content   string    `json:"content"`    // post selftext or comment body
	CreatedAt time.Time `json:"created_at"` // when the post/comment was created
	Score     MHScore   `json:"score"`
}

func callPredict(ctx context.Context, texts []string) ([]MHScore, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	batchPath := strings.TrimSpace(os.Getenv("MENTAL_API_BATCH_PATH"))
	if batchPath == "" {
		batchPath = "/predict/batch"
	}

	const maxBatch = 100
	out := make([]MHScore, 0, len(texts))

	for i := 0; i < len(texts); i += maxBatch {
		j := i + maxBatch
		if j > len(texts) {
			j = len(texts)
		}
		part := texts[i:j]

		payload := struct {
			Texts []string `json:"texts"`
		}{Texts: part}
		body, _ := json.Marshal(payload)

		url := externalAPIBase() + batchPath
		req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		resp, err := httpClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("batch request failed: %w", err)
		}
		respBytes, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return nil, fmt.Errorf("batch API status %s: %s", resp.Status, string(respBytes))
		}

		// Primary: {"predictions":[...]}
		var wrap struct {
			Predictions []MHScore `json:"predictions"`
		}
		if err := json.Unmarshal(respBytes, &wrap); err == nil && wrap.Predictions != nil {
			if len(wrap.Predictions) != len(part) {
				return nil, fmt.Errorf("batch length mismatch: got %d, want %d", len(wrap.Predictions), len(part))
			}
			out = append(out, wrap.Predictions...)
			continue
		}

		// Fallbacks (just in case the server changes shape)
		var arr []MHScore
		if err := json.Unmarshal(respBytes, &arr); err == nil && len(arr) == len(part) {
			out = append(out, arr...)
			continue
		}
		var withScores struct {
			Scores []MHScore `json:"scores"`
		}
		if err := json.Unmarshal(respBytes, &withScores); err == nil && len(withScores.Scores) == len(part) {
			out = append(out, withScores.Scores...)
			continue
		}
		var withResults struct {
			Results []MHScore `json:"results"`
		}
		if err := json.Unmarshal(respBytes, &withResults); err == nil && len(withResults.Results) == len(part) {
			out = append(out, withResults.Results...)
			continue
		}

		return nil, fmt.Errorf("unrecognized batch response: %s", string(respBytes))
	}

	return out, nil
}

func Predict(ctx context.Context, cp *CommentsAndPosts) ([]ClassifiedItem, error) {
	if cp == nil {
		return nil, errors.New("nil CommentsAndPosts")
	}

	// Build texts + metadata
	type work struct {
		Type      string
		Subreddit string
		Permalink string
		Text      string
		Title     string
	}
	var queue []work
	for _, p := range cp.Posts {
		t := strings.TrimSpace(p.Content)
		if t == "" {
			t = "(empty post)"
		}
		queue = append(queue, work{"post", p.Subreddit, p.Permalink, t, p.Title})
	}
	for _, c := range cp.Comments {
		t := strings.TrimSpace(c.Body)
		if t == "" {
			t = "(empty comment)"
		}
		queue = append(queue, work{"comment", c.Subreddit, c.Permalink, t, ""})
	}

	// Collect texts
	texts := make([]string, len(queue))
	for i, w := range queue {
		texts[i] = w.Text
	}

	/*
		// Single batch call
		scores, err := callPredict(ctx, texts)
		if err != nil {
			// On error, fill zeros but keep order
			scores = make([]MHScore, len(queue))
		}
	*/

	// Stitch results
	results := make([]ClassifiedItem, len(queue))
	for i, w := range queue {
		results[i] = ClassifiedItem{
			Type:      w.Type,
			Subreddit: w.Subreddit,
			Permalink: w.Permalink,
			Title:     w.Title,
			Content:   w.Text,
			Score:     MHScore{},
		}
	}

	return results, nil
}

// ---------- Claude: return permalinks only, no post bodies ----------

// What Claude should return (ONLY permalinks + indicators + relevance)
type ClaudePermalinkItem struct {
	Permalink      string   `json:"permalink"`
	Indicators     []string `json:"indicators"`
	RelevanceScore float64  `json:"relevance_score"`
}
type ClaudePermalinkAssessment struct {
	ExecutiveSummary  string                `json:"executive_summary"`
	ConfidenceScore   float64               `json:"confidence_score"`
	MentalHealthScore float64               `json:"mental_health_score"`
	Items             []ClaudePermalinkItem `json:"items"`
}

func trimForClaude(s string) string {
	// Rune-safe trim to avoid cutting UTF-8 sequences
	r := []rune(s)
	if len(r) <= 500 {
		return s
	}
	return string(r[:500]) + "…"
}

func SendPermalinksToClaude(ctx context.Context, classified []ClassifiedItem) (*ClaudePermalinkAssessment, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("missing ANTHROPIC_API_KEY")
	}
	model := os.Getenv("CLAUDE_MODEL")
	if model == "" {
		model = "claude-sonnet-4-20250514"
	}

	// Build a trimmed view just for Claude (do not mutate originals)
	type slimItem struct {
		Type      string  `json:"type"`
		Subreddit string  `json:"subreddit"`
		Permalink string  `json:"permalink"`
		Content   string  `json:"content"`
		Score     MHScore `json:"score"`
	}
	trimmed := make([]slimItem, 0, len(classified))
	for _, it := range classified {
		trimmed = append(trimmed, slimItem{
			Type:      it.Type,
			Subreddit: it.Subreddit,
			Permalink: it.Permalink,
			Content:   trimForClaude(it.Content), // <-- only pass a substring if it's long
			Score:     it.Score,
		})
	}

	inputJSON, _ := json.Marshal(trimmed)

	system := `You are a mental health analyst. You will receive a list of items with fields:
- permalink (identifier)
- type ("post"|"comment")
- subreddit
- content (text to consider)
- score (per-category scores + overall_score from an external classifier). Note these are ranked from 0-1, where the mental health gets worse as it goes to 1.

Task:
1) Produce an executive_summary (string) of the user's mental health state.
2) Produce a confidence_score (0..100).
3) A mental_health_score (0..100). Note that 0 is good mental health and 100 is bad mental health.
4) Select at most 5 notable items and RETURN ONLY their permalinks plus:
   - indicators: short bullet-like phrases that are a max of 3 words (strings). These can be positive or negative. These have to make sense in the context of the scores given to the post.
   - relevance_score (0..100)

IMPORTANT:
- OUTPUT STRICT JSON ONLY, with keys: executive_summary, confidence_score, mental_health_score, items.
- Each item MUST have: permalink, indicators, relevance_score.
- DO NOT include the raw content in your output.`

	userPrompt := "Analyze the following items and return ONLY the JSON as specified. Input follows:\n\n" + string(inputJSON)

	payload := ClaudeRequest{
		Model:     model,
		MaxTokens: 4096,
		System:    system,
		Messages: []ClaudeMessage{
			{
				Role: "user",
				Content: []interface{}{
					TextBlock{Type: "text", Text: userPrompt},
				},
			},
		},
	}

	body, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBytes, _ := io.ReadAll(resp.Body)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("claude api error: %s - %s", resp.Status, string(respBytes))
	}

	var cResp ClaudeResponse
	if err := json.Unmarshal(respBytes, &cResp); err != nil {
		return nil, fmt.Errorf("claude decode error: %w", err)
	}
	if len(cResp.Content) == 0 {
		return nil, fmt.Errorf("claude response had no content")
	}

	var parsed ClaudePermalinkAssessment
	raw := cResp.Content[0].Text
	// strict parse first, then graceful trim
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		if start := strings.IndexByte(raw, '{'); start >= 0 {
			if end := strings.LastIndexByte(raw, '}'); end > start {
				if err2 := json.Unmarshal([]byte(raw[start:end+1]), &parsed); err2 == nil {
					return &parsed, nil
				}
			}
		}
		return nil, fmt.Errorf("failed to parse Claude JSON: %w; raw: %s", err, raw)
	}
	return &parsed, nil
}

// ---------- Orchestrator used by /assessment ----------

func OrchestrateAssessment(ctx context.Context, username string, postLimit, commentLimit int) (*APIAssessmentResponse, error) {
	// 1) Scrape
	cp, err := GetRedditUserPosts(ctx, username, postLimit, commentLimit)
	if err != nil {
		return nil, fmt.Errorf("scrape failed: %w", err)
	}

	// 2) Predict sequentially (one request per item)
	classified, err := Predict(ctx, cp)
	if err != nil {
		return nil, fmt.Errorf("prediction failed: %w", err)
	}

	// 3) Feed to Claude; expect only permalinks + indicators + relevance
	claudeOut, err := SendPermalinksToClaude(ctx, classified)
	if err != nil {
		return nil, fmt.Errorf("claude analysis failed: %w", err)
	}

	// 4) Enrich Claude-selected permalinks with our local content + scores
	byPermalink := make(map[string]ClassifiedItem, len(classified))
	for _, it := range classified {
		byPermalink[it.Permalink] = it
	}

	items := make([]EnrichedItem, 0, len(claudeOut.Items))
	for _, it := range claudeOut.Items {
		if src, ok := byPermalink[it.Permalink]; ok {
			items = append(items, EnrichedItem{
				Type:           src.Type,
				Subreddit:      src.Subreddit,
				Permalink:      src.Permalink,
				Title:          src.Title,
				Content:        src.Content,
				CreatedAt:      src.CreatedAt,
				Score:          src.Score,
				Indicators:     it.Indicators,
				RelevanceScore: it.RelevanceScore,
			})
		}
	}

	return &APIAssessmentResponse{
		ExecutiveSummary:  claudeOut.ExecutiveSummary,
		ConfidenceScore:   claudeOut.ConfidenceScore,
		MentalHealthScore: claudeOut.MentalHealthScore,
		Items:             items,
	}, nil
}

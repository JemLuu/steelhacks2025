package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

var httpClient = &http.Client{Timeout: 60 * time.Second}

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
		} `json:"children"`
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
}

type RedditComment struct {
	Subreddit string
	Body      string
	Permalink string
	Score     int
	CreatedAt time.Time
}

// ClaudeInput is what we send to Claude (only posts + comments)
type ClaudeInput struct {
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

type AssessmentItem struct {
	Type           string   `json:"type"` // "post" | "comment"
	Subreddit      string   `json:"subreddit"`
	Permalink      string   `json:"permalink"`
	Content        string   `json:"content"`         // post title/body or comment body
	Indicators     []string `json:"indicators"`      // short bullet-style phrases
	RelevanceScore float64  `json:"relevance_score"` // 0..10
}

type ParsedAssessment struct {
	ExecutiveSummary string           `json:"executive_summary"`
	ConfidenceScore  float64          `json:"confidence_score"`
	Items            []AssessmentItem `json:"items"`
}

func doGET(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	// IMPORTANT: Set a descriptive UA as Reddit requests
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; GoRedditProfile/1.0; +https://example.com)")

	// Simple 429 handling (Retry-After)
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

	content := &CommentsAndPosts{
		Posts:    []RedditPost{},
		Comments: []RedditComment{},
	}

	// Posts
	resp, err := doGET(ctx, submittedURL)
	if err != nil {
		return content, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return content, fmt.Errorf("submitted.json: %s - %s", resp.Status, string(body))
	}
	var posts Listing
	if err := json.NewDecoder(resp.Body).Decode(&posts); err != nil {
		return content, err
	}
	for _, ch := range posts.Data.Children {
		d := ch.Data
		content.Posts = append(content.Posts, RedditPost{
			Subreddit:   d.Subreddit,
			Title:       d.Title,
			URL:         d.URL,
			Permalink:   "https://www.reddit.com" + d.Permalink,
			Score:       d.Score,
			NumComments: d.NumComments,
			CreatedAt:   time.Unix(int64(d.CreatedUTC), 0).UTC(),
		})
	}

	// Comments
	resp, err = doGET(ctx, commentsURL)
	if err != nil {
		return content, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return content, fmt.Errorf("comments.json: %s - %s", resp.Status, string(body))
	}
	var comments Listing
	if err := json.NewDecoder(resp.Body).Decode(&comments); err != nil {
		return content, err
	}
	for _, ch := range comments.Data.Children {
		d := ch.Data
		content.Comments = append(content.Comments, RedditComment{
			Subreddit: d.Subreddit,
			Body:      d.Body,
			Permalink: "https://www.reddit.com" + d.Permalink,
			Score:     d.Score,
			CreatedAt: time.Unix(int64(d.CreatedUTC), 0).UTC(),
		})
	}

	return content, nil
}

func GetRedditUserProfile(ctx context.Context, username string) (*ProfileInformation, error) {
	base := "https://www.reddit.com"
	aboutURL := fmt.Sprintf("%s/user/%s/about.json", base, username)

	// About
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

	prof := &ProfileInformation{
		Username:   about.Data.Name,
		IconURL:    about.Data.IconImg,
		TotalKarma: about.Data.TotalKarma,
		Bio:        about.Data.Subreddit.PublicDescription,
		CakeDay:    time.Unix(int64(about.Data.CreatedUTC), 0).UTC(),
	}

	return prof, nil
}

func SendContentToClaude(ctx context.Context, content *CommentsAndPosts) (*ParsedAssessment, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("missing ANTHROPIC_API_KEY")
	}
	model := os.Getenv("CLAUDE_MODEL")
	if model == "" {
		model = "claude-sonnet-4-20250514"
	}

	// Limit payload size
	const maxPosts = 1000
	const maxComments = 1000
	posts := content.Posts
	if len(posts) > maxPosts {
		posts = posts[:maxPosts]
	}
	comments := content.Comments
	if len(comments) > maxComments {
		comments = comments[:maxComments]
	}

	// Only send posts and comments
	in := ClaudeInput{
		Posts:    posts,
		Comments: comments,
	}
	inputJSON, err := json.Marshal(in)
	if err != nil {
		return nil, err
	}

	payload := ClaudeRequest{
		Model:     model,
		MaxTokens: 1024,
		System: `You are a mental health analyst. Return ONLY strict JSON with this schema:
{
  "executive_summary": string,
  "confidence_score": number,  // 0..100
  "items": [
    {
      "type": "post"|"comment",
      "subreddit": string,
      "permalink": string,
      "content": string,
      "indicators": [string],
      "relevance_score": number // 0..10
    }
  ]
}
Limit to at most 5 items and include permalinks. No additional prose.`,
		Messages: []ClaudeMessage{
			{
				Role: "user",
				Content: []interface{}{
					TextBlock{
						Type: "text",
						Text: "Analyze this Reddit user's posts and comments and return a brief JSON summary with keys: topics, writing_style, activity_patterns, notable_posts (permalinks). Input follows:\n\n" + string(inputJSON),
					},
				},
			},
		},
	}

	body, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{Timeout: 20 * time.Second}
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

	// Parse the model output as strict JSON; if wrapped with prose, trim to outermost braces.
	raw := cResp.Content[0].Text
	var parsed ParsedAssessment
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

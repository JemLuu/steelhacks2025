package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

var httpClient = &http.Client{Timeout: 60 * time.Second}

// ----- Raw Reddit API structs -----

type RedditAbout struct {
	Data struct {
		Name       string  `json:"name"`
		IconImg    string  `json:"icon_img"`
		TotalKarma int     `json:"total_karma"`
		CreatedUTC float64 `json:"created_utc"`
		Subreddit  struct {
			PublicDescription string `json:"public_description"`
		} `json:"subreddit"`
	} `json:"data"`
}

type Listing struct {
	Data struct {
		Children []struct {
			Kind string `json:"kind"`
			Data struct {
				Subreddit   string   `json:"subreddit"`
				Title       string   `json:"title"`
				SelfText    string   `json:"selftext"`
				Body        string   `json:"body"`
				Permalink   string   `json:"permalink"`
				URL         string   `json:"url"`
				Score       int      `json:"score"`
				NumComments int      `json:"num_comments"`
				CreatedUTC  float64  `json:"created_utc"`
				UpvoteRatio *float64 `json:"upvote_ratio,omitempty"`
				Ups         *int     `json:"ups,omitempty"`
				Downs       *int     `json:"downs,omitempty"`
			} `json:"data"`
		} `json:"children"`
	} `json:"data"`
}

// ----- Internal types -----

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
	Upvotes     *int
	Downvotes   *int
}

type RedditComment struct {
	Subreddit string
	Body      string
	Permalink string
	Score     int
	CreatedAt time.Time
	Upvotes   *int
	Downvotes *int
}

// What we send to Claude
type ClaudeInput struct {
	Posts    []RedditPost    `json:"posts"`
	Comments []RedditComment `json:"comments"`
}

// Parsed assessment
type AssessmentItem struct {
	Type           string   `json:"type"`
	Subreddit      string   `json:"subreddit"`
	Permalink      string   `json:"permalink"`
	Content        string   `json:"content"`
	Indicators     []string `json:"indicators"`
	RelevanceScore float64  `json:"relevance_score"`
}

type ParsedAssessment struct {
	ExecutiveSummary string           `json:"executive_summary"`
	ConfidenceScore  float64          `json:"confidence_score"`
	Items            []AssessmentItem `json:"items"`
}

// ----- Scraper -----

func doGET(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; GoRedditProfile/1.0)")

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

func GetRedditUserProfile(ctx context.Context, username string) (*ProfileInformation, error) {
	url := fmt.Sprintf("https://www.reddit.com/user/%s/about.json", username)
	resp, err := doGET(ctx, url)
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

func estimateVotes(score int, ratio *float64) (up *int, down *int) {
	if ratio == nil {
		return nil, nil
	}
	r := *ratio
	if r <= 0 || r >= 1 || math.Abs(2*r-1) < 1e-6 {
		return nil, nil
	}
	T := float64(score) / (2*r - 1)
	if T < 0 {
		return nil, nil
	}
	u := int(math.Round(r * T))
	d := int(math.Round((1 - r) * T))
	return &u, &d
}

func GetRedditUserPosts(ctx context.Context, username string, postLimit, commentLimit int) (*CommentsAndPosts, error) {
	base := "https://www.reddit.com"
	submittedURL := fmt.Sprintf("%s/user/%s/submitted.json?limit=%d", base, username, postLimit)
	commentsURL := fmt.Sprintf("%s/user/%s/comments.json?limit=%d", base, username, commentLimit)

	out := &CommentsAndPosts{}

	// posts
	resp, err := doGET(ctx, submittedURL)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	var posts Listing
	if err := json.NewDecoder(resp.Body).Decode(&posts); err != nil {
		return out, err
	}
	for _, ch := range posts.Data.Children {
		d := ch.Data
		up, down := estimateVotes(d.Score, d.UpvoteRatio)
		if d.Ups != nil {
			up = d.Ups
			if d.Score != 0 {
				downVal := *d.Ups - d.Score
				if downVal >= 0 {
					down = &downVal
				}
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
			Upvotes:     up,
			Downvotes:   down,
		})
	}

	// comments
	resp, err = doGET(ctx, commentsURL)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	var comments Listing
	if err := json.NewDecoder(resp.Body).Decode(&comments); err != nil {
		return out, err
	}
	for _, ch := range comments.Data.Children {
		d := ch.Data
		var up, down *int
		if d.Ups != nil {
			up = d.Ups
			downVal := *d.Ups - d.Score
			if downVal >= 0 {
				down = &downVal
			}
		}
		out.Comments = append(out.Comments, RedditComment{
			Subreddit: d.Subreddit,
			Body:      d.Body,
			Permalink: "https://www.reddit.com" + d.Permalink,
			Score:     d.Score,
			CreatedAt: time.Unix(int64(d.CreatedUTC), 0).UTC(),
			Upvotes:   up,
			Downvotes: down,
		})
	}

	return out, nil
}

// ----- Claude -----

func SendContentToClaude(ctx context.Context, content *CommentsAndPosts) (*ParsedAssessment, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("missing ANTHROPIC_API_KEY")
	}
	model := os.Getenv("CLAUDE_MODEL")
	if model == "" {
		model = "claude-sonnet-4-20250514"
	}

	in := ClaudeInput{Posts: content.Posts, Comments: content.Comments}
	inputJSON, _ := json.Marshal(in)

	// build request
	type TextBlock struct{ Type, Text string }
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

	system := `You are a mental health analyst. Return ONLY strict JSON with this schema:
{
  "executive_summary": string,
  "confidence_score": number,
  "items": [
    {
      "type": "post"|"comment",
      "subreddit": string,
      "permalink": string,
      "content": string,
      "indicators": [string],
      "relevance_score": number
    }
  ]
}`

	userPrompt := "Analyze the following Reddit posts and comments:\n\n" + string(inputJSON)

	payload := ClaudeRequest{
		Model:     model,
		MaxTokens: 2000,
		System:    system,
		Messages: []ClaudeMessage{
			{Role: "user", Content: []interface{}{TextBlock{"text", userPrompt}}},
		},
	}

	body, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBytes, _ := io.ReadAll(resp.Body)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("claude api error: %s - %s", resp.Status, string(respBytes))
	}

	var claudeResp ClaudeResponse
	if err := json.Unmarshal(respBytes, &claudeResp); err != nil {
		return nil, err
	}
	if len(claudeResp.Content) == 0 {
		return nil, fmt.Errorf("empty Claude response")
	}

	// parse model output
	var parsed ParsedAssessment
	raw := claudeResp.Content[0].Text
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		if jStart := strings.IndexByte(raw, '{'); jStart >= 0 {
			if jEnd := strings.LastIndexByte(raw, '}'); jEnd > jStart {
				if err2 := json.Unmarshal([]byte(raw[jStart:jEnd+1]), &parsed); err2 == nil {
					return &parsed, nil
				}
			}
		}
		return nil, fmt.Errorf("failed to parse Claude JSON: %w; raw: %s", err, raw)
	}

	return &parsed, nil
}

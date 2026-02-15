package anthropic

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/docker/cagent/pkg/chat"
)

func TestDetectMimeType(t *testing.T) {
	tests := []struct {
		path     string
		expected string
	}{
		{"image.jpg", "image/jpeg"},
		{"image.jpeg", "image/jpeg"},
		{"image.png", "image/png"},
		{"image.gif", "image/gif"},
		{"image.webp", "image/webp"},
		{"document.pdf", "application/pdf"},
		{"readme.txt", "text/plain"},
		{"readme.md", "text/plain"},
		{"readme.markdown", "text/plain"},
		// json and csv are treated as text/plain for provider compatibility
		{"data.json", "text/plain"},
		{"data.csv", "text/plain"},
		{"unknown.xyz", "application/octet-stream"},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			result := chat.DetectMimeType(tt.path)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsImageMime(t *testing.T) {
	tests := []struct {
		mimeType string
		expected bool
	}{
		{"image/jpeg", true},
		{"image/png", true},
		{"image/gif", true},
		{"image/webp", true},
		{"application/pdf", false},
		{"text/plain", false},
		{"application/octet-stream", false},
	}

	for _, tt := range tests {
		t.Run(tt.mimeType, func(t *testing.T) {
			result := IsImageMime(tt.mimeType)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsDocumentMime(t *testing.T) {
	tests := []struct {
		mimeType string
		expected bool
	}{
		{"application/pdf", true},
		{"text/plain", true},
		{"text/markdown", false},
		{"image/jpeg", false},
		{"image/png", false},
		{"application/octet-stream", false},
	}

	for _, tt := range tests {
		t.Run(tt.mimeType, func(t *testing.T) {
			result := IsAnthropicDocumentMime(tt.mimeType)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsSupportedMime(t *testing.T) {
	tests := []struct {
		mimeType string
		expected bool
	}{
		{"image/jpeg", true},
		{"image/png", true},
		{"image/gif", true},
		{"image/webp", true},
		{"application/pdf", true},
		{"text/plain", true},
		{"text/markdown", false},
		{"application/json", false},
		{"application/octet-stream", false},
	}

	for _, tt := range tests {
		t.Run(tt.mimeType, func(t *testing.T) {
			result := IsSupportedMime(tt.mimeType)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestHashFile(t *testing.T) {
	// Create a temporary file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.txt")
	content := []byte("test content for hashing")
	err := os.WriteFile(testFile, content, 0o644)
	require.NoError(t, err)

	// Hash should be consistent for same content
	hash1, err := hashFile(testFile)
	require.NoError(t, err)
	assert.NotEmpty(t, hash1)

	hash2, err := hashFile(testFile)
	require.NoError(t, err)
	assert.Equal(t, hash1, hash2)

	// Different content should produce different hash
	testFile2 := filepath.Join(tmpDir, "test2.txt")
	err = os.WriteFile(testFile2, []byte("different content"), 0o644)
	require.NoError(t, err)

	hash3, err := hashFile(testFile2)
	require.NoError(t, err)
	assert.NotEqual(t, hash1, hash3)
}

func TestHashFile_NotFound(t *testing.T) {
	_, err := hashFile("/nonexistent/path/to/file.txt")
	assert.Error(t, err)
}

func TestNewFileManager(t *testing.T) {
	fm := NewFileManager(nil)
	require.NotNil(t, fm)
	assert.Equal(t, 0, fm.CachedCount())
}

func TestUploadedFile_TTL(t *testing.T) {
	old := &UploadedFile{
		FileID:     "file_old",
		UploadedAt: time.Now().Add(-25 * time.Hour),
	}
	recent := &UploadedFile{
		FileID:     "file_recent",
		UploadedAt: time.Now().Add(-1 * time.Hour),
	}

	cutoff := time.Now().Add(-24 * time.Hour)

	assert.True(t, old.UploadedAt.Before(cutoff), "old file should be before cutoff")
	assert.False(t, recent.UploadedAt.Before(cutoff), "recent file should not be before cutoff")
}

func TestFileManager_Deduplication(t *testing.T) {
	// This test verifies the deduplication logic structure
	// Actual upload testing would require mocking the Anthropic client

	fm := NewFileManager(nil)
	require.NotNil(t, fm)

	// Manually populate the cache to test deduplication logic
	testHash := "abc123"
	testFile := &UploadedFile{
		FileID:      "file_test",
		Filename:    "test.png",
		MimeType:    "image/png",
		ContentHash: testHash,
		UploadedAt:  time.Now(),
		LocalPath:   "/path/to/test.png",
	}

	fm.mu.Lock()
	fm.uploads[testHash] = testFile
	fm.paths["/path/to/test.png"] = testHash
	fm.mu.Unlock()

	// Check that the file is cached
	assert.Equal(t, 1, fm.CachedCount())

	// Verify the path mapping exists
	fm.mu.RLock()
	hash, ok := fm.paths["/path/to/test.png"]
	fm.mu.RUnlock()
	assert.True(t, ok)
	assert.Equal(t, testHash, hash)

	// Verify the upload exists
	fm.mu.RLock()
	upload, ok := fm.uploads[testHash]
	fm.mu.RUnlock()
	assert.True(t, ok)
	assert.Equal(t, "file_test", upload.FileID)
}

func TestCreateFileContentBlock_NotSupported(t *testing.T) {
	// Standard API doesn't support file references - Files API is Beta-only
	_, err := createFileContentBlock("file_123", "image/png")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Beta API")
}

func TestCreateBetaFileContentBlock_Image(t *testing.T) {
	block, err := createBetaFileContentBlock("file_beta_123", "image/jpeg")
	require.NoError(t, err)
	assert.NotNil(t, block.OfImage)
	assert.Nil(t, block.OfDocument)
	assert.Equal(t, "file_beta_123", block.OfImage.Source.OfFile.FileID)
}

func TestCreateBetaFileContentBlock_Document(t *testing.T) {
	block, err := createBetaFileContentBlock("file_beta_456", "application/pdf")
	require.NoError(t, err)
	assert.NotNil(t, block.OfDocument)
	assert.Nil(t, block.OfImage)
	assert.Equal(t, "file_beta_456", block.OfDocument.Source.OfFile.FileID)
}

func TestCreateBetaFileContentBlock_Unsupported(t *testing.T) {
	_, err := createBetaFileContentBlock("file_beta_000", "video/mp4")
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrUnsupportedFileType)
}

func TestFileManager_CleanupAll_Empty(t *testing.T) {
	fm := NewFileManager(nil)
	err := fm.CleanupAll(t.Context())
	require.NoError(t, err)
	assert.Equal(t, 0, fm.CachedCount())
}

func TestIsRetryableUploadError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{"nil error", nil, false},
		{"generic error", fmt.Errorf("something went wrong"), false},
		{"400 bad request", &anthropic.Error{StatusCode: 400}, false},
		{"401 unauthorized", &anthropic.Error{StatusCode: 401}, false},
		{"403 forbidden", &anthropic.Error{StatusCode: 403}, false},
		{"404 not found", &anthropic.Error{StatusCode: 404}, false},
		{"413 file too large", &anthropic.Error{StatusCode: 413}, false},
		{"429 rate limit", &anthropic.Error{StatusCode: 429}, false},
		{"500 internal server error", &anthropic.Error{StatusCode: 500}, true},
		{"502 bad gateway", &anthropic.Error{StatusCode: 502}, true},
		{"503 service unavailable", &anthropic.Error{StatusCode: 503}, true},
		{"504 gateway timeout", &anthropic.Error{StatusCode: 504}, true},
		{"wrapped 500", fmt.Errorf("upload failed: %w", &anthropic.Error{StatusCode: 500}), true},
		{"wrapped 400", fmt.Errorf("upload failed: %w", &anthropic.Error{StatusCode: 400}), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableUploadError(tt.err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsRetryableUploadError_WithRequest(t *testing.T) {
	// Simulate the real error shape from the Anthropic SDK
	req, _ := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/files?beta=true", nil)
	resp := &http.Response{StatusCode: 500}
	apiErr := &anthropic.Error{
		StatusCode: 500,
		Request:    req,
		Response:   resp,
		RequestID:  "req_011CYAF8kKRHegwiS7522zF5",
	}

	assert.True(t, isRetryableUploadError(apiErr))
	assert.True(t, isRetryableUploadError(fmt.Errorf("failed to upload file: %w", apiErr)))
}
